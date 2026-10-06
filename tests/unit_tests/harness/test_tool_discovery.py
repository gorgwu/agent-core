# coding: utf-8
"""Tests for automatic tool discovery selection and authorization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openjiuwen.core.foundation.llm.schema.message import UserMessage
from openjiuwen.core.foundation.tool import ToolCard, ToolExposure, ToolInfo
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    ModelCallInputs,
    ToolCallInputs,
)
from openjiuwen.harness.prompts.builder import PromptSection, SystemPromptBuilder
from openjiuwen.harness.prompts.tools.tool_call import DESCRIPTION, TOOL_CALL_PARAMS
from openjiuwen.harness.rails.progressive_tool_rail import ProgressiveToolRail
from openjiuwen.harness.schema.config import DeepAgentConfig
from openjiuwen.harness.schema.deep_agent_spec import DeepAgentSpec, ProgressiveToolSpec
from openjiuwen.harness.tools.tool_discovery import decisions_api
from openjiuwen.harness.tools.tool_discovery.tool_call import ToolCallTool
from openjiuwen.harness.tools.tool_discovery.tool_search import ToolSearchTool


class _Session:
    def __init__(self):
        self.state = {}

    def get_state(self, key):
        return self.state.get(key)

    def update_state(self, values):
        self.state.update(values)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_jev_credential_presence_is_checked_at_rail_startup(monkeypatch):
    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.has_decisions_api_key",
        lambda _api_key=None: False,
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(tool_discovery_backend="jev")
    )

    rail.init(SimpleNamespace())

    assert rail._jev_disabled is True


def test_progressive_tool_spec_passes_discovery_settings_to_agent_config():
    spec = ProgressiveToolSpec(
        tool_discovery_backend="jev",
        tool_discovery_model="typesafe/jev-1.13",
        tool_discovery_max_tools=10,
    )

    deep_agent_spec = DeepAgentSpec.model_construct(progressive_tool=spec)

    assert deep_agent_spec._progressive_tool_kwargs() == {
        "progressive_tool_enabled": True,
        "tool_search_limit": 5,
        "tool_discovery_backend": "jev",
        "tool_discovery_api_key": None,
        "tool_discovery_api_base": None,
        "tool_discovery_model": "typesafe/jev-1.13",
        "tool_discovery_max_tools": 10,
    }


@pytest.mark.asyncio
async def test_decisions_request_returns_top_scored_tools(monkeypatch):
    tools = [
        ToolInfo(name="calendar_create", description="Create calendar events"),
        ToolInfo(name="mail_send", description="Send email"),
        ToolInfo(name="calendar_search", description="Search calendar"),
        ToolInfo(name="calendar_delete", description="Delete calendar event"),
        ToolInfo(name="mail_search", description="Search email"),
        ToolInfo(name="mail_draft", description="Draft email"),
    ]
    captured = {}

    def post(url, *, json, headers, timeout):
        captured.update(url=url, body=json, headers=headers, timeout=timeout)
        return _Response(
            {
                "answers": {
                    "tool_group_0000": {
                        "type": "choice",
                        "choice": "tool_000000",
                        "confidence": 0.8,
                        "probabilities": {
                            "tool_000000": 0.8,
                            "tool_000001": 0.1,
                            "tool_000002": 0.55,
                            "tool_000003": 0.4,
                            "tool_000004": 0.7,
                            "tool_000005": 0.6,
                            "no_relevant_tool": 0.1,
                            "untrusted_name": 1.0,
                        },
                    }
                }
            }
        )

    monkeypatch.setattr(decisions_api.requests, "post", post)
    result = await decisions_api.select_deferred_tools(
        query="create a meeting",
        tools=tools,
        api_key="test-key",
        max_tools=4,
    )

    assert result == [tools[0], tools[4], tools[5], tools[2]]
    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["state"] == {"user_request": "create a meeting"}
    question = body["questions"]["tool_group_0000"]
    assert question["type"] == "choice"
    assert question["criteria"]["tool_000000"] == "calendar_create: Create calendar events"
    assert "no_relevant_tool" in question["criteria"]


@pytest.mark.asyncio
async def test_uses_configured_decisions_api_base(monkeypatch):
    captured = {}

    def post(url, **_kwargs):
        captured["url"] = url
        return _Response({"answers": {"tool_group_0000": {"type": "choice", "probabilities": {}}}})

    monkeypatch.setattr(decisions_api.requests, "post", post)
    await decisions_api.select_deferred_tools(
        query="find a calendar tool",
        tools=[ToolInfo(name="calendar_search", description="Search calendar")],
        api_key="test-key",
        api_base="https://jev.example/api/decisions",
    )

    assert captured["url"] == "https://jev.example/api/decisions"


@pytest.mark.asyncio
async def test_discards_zero_scores_and_caps_at_configured_max(monkeypatch):
    tools = [ToolInfo(name=f"tool_{i}", description="candidate") for i in range(12)]

    def post(_url, *, json, **_kwargs):
        return _Response({"answers": {"tool_group_0000": {
            "type": "choice",
            "probabilities": {
                f"tool_{i:06d}": (0.0 if i == 0 else 0.1)
                for i in range(12)
            },
        }}})

    monkeypatch.setattr(decisions_api.requests, "post", post)
    result = await decisions_api.select_deferred_tools(
        query="find relevant tools", tools=tools, api_key="test-key", max_tools=10,
    )

    assert len(result) == 10
    assert tools[0] not in result
    assert {tool.name for tool in result} == {
        f"tool_{i}" for i in range(1, 12)
    } - {"tool_9"}


@pytest.mark.asyncio
async def test_loads_tool_discovery_key_from_dotenv_on_demand(monkeypatch):
    monkeypatch.delenv("TOOL_DISCOVERY_API_KEY", raising=False)
    headers_seen = {}

    def load_local_env(*, override):
        assert override is False
        monkeypatch.setenv("TOOL_DISCOVERY_API_KEY", "dotenv-key")

    def post(_url, *, headers, **_kwargs):
        headers_seen.update(headers)
        return _Response({"answers": {"tool_group_0000": {"type": "choice", "probabilities": {}}}})

    monkeypatch.setattr(decisions_api, "load_dotenv", load_local_env)
    monkeypatch.setattr(decisions_api.requests, "post", post)
    await decisions_api.select_deferred_tools(
        query="look up a calendar",
        tools=[ToolInfo(name="calendar_search", description="Search calendar events")],
    )

    assert headers_seen["Authorization"] == "Bearer dotenv-key"


@pytest.mark.asyncio
async def test_does_not_accept_openrouter_key_as_discovery_credential(monkeypatch):
    monkeypatch.delenv("TOOL_DISCOVERY_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def load_local_env(*, override):
        assert override is False
        monkeypatch.setenv("OPENROUTER_API_KEY", "old-key")

    monkeypatch.setattr(decisions_api, "load_dotenv", load_local_env)
    with pytest.raises(ValueError, match="TOOL_DISCOVERY_API_KEY is required"):
        await decisions_api.select_deferred_tools(
            query="look up a calendar",
            tools=[ToolInfo(name="calendar_search", description="Search calendar events")],
        )


@pytest.mark.asyncio
async def test_choice_option_groups_respect_openrouter_limit(monkeypatch):
    tools = [ToolInfo(name=f"tool_{index}", description="candidate") for index in range(256)]
    captured = {}

    def post(_url, *, json, **_kwargs):
        captured["body"] = json
        return _Response({"answers": {}})

    monkeypatch.setattr(decisions_api.requests, "post", post)
    await decisions_api.select_deferred_tools(
        query="find something",
        tools=tools,
        api_key="test-key",
    )

    questions = captured["body"]["questions"]
    assert len(questions) == 2
    assert all(len(question["criteria"]) <= 255 for question in questions.values())
    assert questions["tool_group_0001"]["criteria"]["tool_000255"] == "tool_255: candidate"


@pytest.mark.asyncio
async def test_rail_automatically_authorizes_and_exposes_selected_schema(monkeypatch):
    candidate = ToolInfo(
        name="calendar_create",
        description="Create a calendar event",
        parameters={"type": "object", "properties": {"title": {"type": "string"}}},
    )
    selected = []

    async def rank(**kwargs):
        selected.append(kwargs)
        return [decisions_api.ScoredTool(tool=candidate, score=0.91)]

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", rank
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(
            progressive_tool_enabled=True,
            tool_discovery_backend="jev",
            tool_discovery_max_tools=10,
        )
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [candidate]
    session = _Session()
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    manager = SimpleNamespace(list=lambda: [], registry_revision=1)
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(system_prompt_builder=builder, ability_manager=manager),
        inputs=ModelCallInputs(
            messages=[UserMessage(content="schedule a planning meeting")],
            tools=[ToolInfo(name="tool_search"), ToolInfo(name="tool_call")],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)

    assert selected[0]["query"] == "schedule a planning meeting"
    assert session.get_state("__progressive_discovered_tool_names__") == ["calendar_create"]
    assert [tool.name for tool in ctx.inputs.tools] == ["tool_call"]
    prompt = builder.build()
    assert "calendar_create" in prompt
    assert '"title"' in prompt
    assert selected[0]["max_tools"] == 10
    assert "tool_search" in prompt  # It is explicitly disallowed in automatic discovery mode.
    assert "selected by automatic retrieval" in prompt
    assert "always invoke a selected deferred tool by calling the `tool_call` wrapper" in prompt
    assert "Never issue a tool call whose function name is the deferred tool name itself" in prompt
    assert "selected by automatic discovery" in DESCRIPTION["en"]
    assert "never invoke their names directly" in DESCRIPTION["en"]
    assert "selected by discovery" in TOOL_CALL_PARAMS["name"]["en"]


@pytest.mark.asyncio
async def test_rail_discovers_once_per_user_turn_and_keeps_result_order(monkeypatch, caplog):
    first = ToolInfo(name="first_match", description="First candidate")
    second = ToolInfo(name="second_match", description="Second candidate")
    rank_calls = []

    async def rank(**kwargs):
        rank_calls.append(kwargs)
        return [
            decisions_api.ScoredTool(tool=first, score=0.9),
            decisions_api.ScoredTool(tool=second, score=0.4),
        ]

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", rank
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(progressive_tool_enabled=True, tool_discovery_backend="jev")
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [first, second]
    session = _Session()
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(
            system_prompt_builder=builder,
            ability_manager=SimpleNamespace(list=lambda: [], registry_revision=1),
        ),
        inputs=ModelCallInputs(
            messages=[UserMessage(content="compare these tools")],
            tools=[ToolInfo(name="tool_search"), ToolInfo(name="tool_call")],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)
    ctx.inputs.react_iteration = 2
    await rail.before_model_call(ctx)

    assert len(rank_calls) == 1
    trace = session.get_state("__progressive_tool_discovery_trace__")
    assert len(trace) == 1
    assert trace[0]["matched"] == ["first_match", "second_match"]
    assert caplog.text.count("[ToolDiscovery] deferred tool routing | candidate_count=") == 1
    prompt = builder.build()
    assert prompt.index("### first_match") < prompt.index("### second_match")

    ctx.inputs.messages.append(UserMessage(content="now a new user turn"))
    await rail.before_model_call(ctx)
    assert len(rank_calls) == 2
    assert len(session.get_state("__progressive_tool_discovery_trace__")) == 2


@pytest.mark.asyncio
async def test_discovery_authorization_is_replaced_each_user_turn(monkeypatch):
    first = ToolInfo(name="first_turn_tool", description="First turn tool")
    second = ToolInfo(name="second_turn_tool", description="Second turn tool")
    executed = []

    async def rank(*, query, **_kwargs):
        candidate = second if "second" in query else first
        return [decisions_api.ScoredTool(tool=candidate, score=0.9)]

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", rank
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(progressive_tool_enabled=True, tool_discovery_backend="jev")
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [first, second]
    session = _Session()
    manager = SimpleNamespace(
        list=lambda: [],
        registry_revision=1,
        get=lambda _name: None,
        execute=lambda **kwargs: executed.append(kwargs),
    )
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(system_prompt_builder=builder, ability_manager=manager),
        inputs=ModelCallInputs(
            messages=[UserMessage(content="first request")],
            tools=[ToolInfo(name="tool_search"), ToolInfo(name="tool_call")],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)
    assert session.get_state("__progressive_discovered_tool_names__") == [
        "first_turn_tool"
    ]

    ctx.inputs.messages.append(UserMessage(content="second request"))
    await rail.before_model_call(ctx)
    assert session.get_state("__progressive_discovered_tool_names__") == [
        "second_turn_tool"
    ]

    result = await rail._call_discovered_tool(
        "first_turn_tool",
        {},
        session,
        SimpleNamespace(
            agent=SimpleNamespace(ability_manager=manager),
            inputs=SimpleNamespace(tool_call=SimpleNamespace(id="call")),
            extra={},
        ),
    )
    assert result.success is False
    assert "must be selected by discovery" in result.error
    assert executed == []


@pytest.mark.asyncio
async def test_discovery_unselected_deferred_tool_is_not_callable_directly_or_through_wrapper(
    monkeypatch,
):
    selected = ToolInfo(name="selected_tool", description="Selected this turn")
    unselected = ToolInfo(name="unselected_tool", description="Not selected")
    cards = {
        name: ToolCard(
            id=name,
            name=name,
            description=description,
            input_params={"type": "object", "properties": {}},
            exposure=ToolExposure.DEFERRED,
        )
        for name, description in (
            (selected.name, selected.description),
            (unselected.name, unselected.description),
        )
    }
    executed = []

    class _AbilityManager:
        def list(self):
            return list(cards.values())

        def get(self, name):
            return cards.get(name)

        async def execute(self, **kwargs):
            executed.append(kwargs)
            return []

    async def rank(**_kwargs):
        return [decisions_api.ScoredTool(tool=selected, score=0.9)]

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", rank
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(progressive_tool_enabled=True, tool_discovery_backend="jev")
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [selected, unselected]
    session = _Session()
    manager = _AbilityManager()
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    agent = SimpleNamespace(system_prompt_builder=builder, ability_manager=manager)
    ctx = AgentCallbackContext(
        agent=agent,
        inputs=ModelCallInputs(
            messages=[UserMessage(content="use the selected tool")],
            tools=[
                ToolInfo(name="tool_search"),
                ToolInfo(name="tool_call"),
                selected,
                unselected,
            ],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)
    assert session.get_state("__progressive_discovered_tool_names__") == [
        "selected_tool"
    ]

    direct_ctx = AgentCallbackContext(
        agent=agent,
        inputs=ToolCallInputs(
            tool_call=SimpleNamespace(id="direct-unselected"),
            tool_name="unselected_tool",
            tool_args={},
        ),
        session=session,
    )
    await rail.before_tool_call(direct_ctx)
    assert direct_ctx.extra.get("_skip_tool") is True

    wrapper_ctx = AgentCallbackContext(
        agent=agent,
        inputs=ToolCallInputs(
            tool_call=SimpleNamespace(id="wrapper-unselected"),
            tool_name="tool_call",
            tool_args={"name": "unselected_tool", "args": {}},
        ),
        session=session,
    )
    wrapper = ToolCallTool(call_tool=rail._call_discovered_tool)
    output = await wrapper.invoke(
        wrapper_ctx.inputs.tool_args,
        session=session,
        _tool_callback_context=wrapper_ctx,
    )

    assert output.success is False
    assert "must be selected by discovery" in (output.error or "")
    assert executed == []


@pytest.mark.asyncio
async def test_before_invoke_resets_authorization_only_when_user_turn_changes():
    class _AbilityManager:
        async def list_tool_info(self):
            return []

        def list(self):
            return []

    rail = ProgressiveToolRail(DeepAgentConfig(progressive_tool_enabled=True))
    session = _Session()
    rail._set_discovered_tools(
        session,
        ["previous_turn_tool"],
        fingerprints={"previous_turn_tool": "old-fingerprint"},
    )
    session.update_state({"__progressive_discovered_tool_turn_index__": 1})
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(ability_manager=_AbilityManager()),
        inputs=ModelCallInputs(messages=[UserMessage(content="same turn resumed")]),
        session=session,
    )

    await rail.before_invoke(ctx)
    assert session.get_state("__progressive_discovered_tool_names__") == [
        "previous_turn_tool"
    ]

    ctx.inputs.messages.append(UserMessage(content="new user turn"))
    await rail.before_invoke(ctx)

    assert session.get_state("__progressive_discovered_tool_names__") == []
    assert session.get_state("__progressive_discovered_tool_fingerprints__") == {}


@pytest.mark.asyncio
async def test_rail_enables_model_directed_search_and_caches_after_discovery_failure(monkeypatch):
    candidate = ToolInfo(name="private_tool", description="A deferred tool")
    calls = []

    async def fail(**_kwargs):
        calls.append(_kwargs)
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", fail
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(progressive_tool_enabled=True, tool_discovery_backend="jev")
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [candidate]
    rail._cached_all_tool_infos = [candidate]
    session = _Session()
    rail._set_discovered_tools(
        session,
        ["previous_turn_tool"],
        fingerprints={"previous_turn_tool": "old-fingerprint"},
    )
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(
            system_prompt_builder=builder,
            ability_manager=SimpleNamespace(registry_revision=1),
        ),
        inputs=ModelCallInputs(
            messages=[UserMessage(content="use private_tool")],
            tools=[ToolInfo(name="tool_search"), ToolInfo(name="tool_call")],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)
    assert session.get_state("__progressive_discovered_tool_names__") == []
    assert [tool.name for tool in ctx.inputs.tools] == ["tool_search", "tool_call"]
    prompt = builder.build()
    assert "Deferred tools available through tool_search" in prompt
    assert "private_tool" in prompt
    assert "Automatically discovered deferred tools" not in prompt
    discovery_state = session.get_state("__progressive_tool_discovery__")
    assert discovery_state["selection_backend"] == "bm25_tool_search_fallback"
    assert discovery_state["manual_search_fallback"] is True
    assert "API unavailable" in discovery_state["fallback_error"]

    search_tool = ToolSearchTool(search_tools=rail._search_tools)
    result = await search_tool.invoke(
        {"query": "private_tool"},
        session=session,
    )
    assert result.success is True
    assert [item["name"] for item in result.data["results"]] == ["private_tool"]
    assert session.get_state("__progressive_discovered_tool_names__") == ["private_tool"]

    ctx.inputs.react_iteration = 2
    await rail.before_model_call(ctx)
    assert len(calls) == 1
    assert session.get_state("__progressive_discovered_tool_names__") == [
        "private_tool"
    ]
    assert [tool.name for tool in ctx.inputs.tools] == ["tool_search", "tool_call"]

    ctx.inputs.messages.append(UserMessage(content="another user turn"))
    await rail.before_model_call(ctx)
    assert len(calls) == 1
    assert [tool.name for tool in ctx.inputs.tools] == ["tool_search", "tool_call"]


@pytest.mark.asyncio
async def test_successful_discovery_no_match_does_not_trigger_bm25_fallback(monkeypatch):
    candidate = ToolInfo(name="private_tool", description="A deferred tool")
    calls = []

    async def no_match(**_kwargs):
        calls.append(_kwargs)
        return []

    monkeypatch.setattr(
        "openjiuwen.harness.rails.progressive_tool_rail.rank_decisions_api_tools", no_match
    )
    rail = ProgressiveToolRail(
        DeepAgentConfig(progressive_tool_enabled=True, tool_discovery_backend="jev")
    )
    rail._meta_tool_names = {"tool_search", "tool_call"}
    rail._list_registered_deferred_tool_infos = lambda _agent: [candidate]
    session = _Session()
    builder = SystemPromptBuilder(language="en")
    builder.add_section(PromptSection(name="identity", content={"en": "Agent"}))
    ctx = AgentCallbackContext(
        agent=SimpleNamespace(
            system_prompt_builder=builder,
            ability_manager=SimpleNamespace(list=lambda: [], registry_revision=1),
        ),
        inputs=ModelCallInputs(
            messages=[UserMessage(content="use private_tool")],
            tools=[ToolInfo(name="tool_search"), ToolInfo(name="tool_call")],
        ),
        session=session,
    )

    await rail.before_model_call(ctx)

    assert len(calls) == 1
    assert session.get_state("__progressive_discovered_tool_names__") == []
    discovery_state = session.get_state("__progressive_tool_discovery__")
    assert discovery_state["selection_backend"] == "jev"
    assert discovery_state["fallback_error"] == ""
