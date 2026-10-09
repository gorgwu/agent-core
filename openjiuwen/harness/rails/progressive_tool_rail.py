# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ProgressiveToolRail for large-scale tool usage with progressive disclosure."""

from __future__ import annotations

import inspect
import json
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from openjiuwen.core.common.logging import logger
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.foundation.llm.schema.message import UserMessage
from openjiuwen.core.foundation.tool import ToolCard, ToolExposure, ToolInfo
from openjiuwen.core.single_agent.interrupt.exception import ToolInterruptException
from openjiuwen.core.single_agent.prompts.builder import PromptSection
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.prompts.builder import SystemPromptBuilder
from openjiuwen.harness.prompts.prompt_attachment_manager import PromptAttachmentKind
from openjiuwen.harness.prompts.sections.progressive_tool_rail import (
    build_multilingual_progressive_tool_rules_section,
    render_deferred_tool_catalog_delta,
    render_deferred_tool_catalog_snapshot,
)
from openjiuwen.harness.rails.base import DeepAgentRail
from openjiuwen.harness.schema.config import DeepAgentConfig
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.tool_discovery.bm25 import BM25ToolIndex
from openjiuwen.harness.tools.tool_discovery.decisions_api import (
    has_decisions_api_key,
    rank_decisions_api_tools,
)
from openjiuwen.harness.tools.tool_discovery.tool_call import RelayedToolOutput, ToolCallTool
from openjiuwen.harness.tools.tool_discovery.tool_search import (
    DEFAULT_TOOL_SEARCH_LIMIT,
    ToolSearchTool,
)

_DISCOVERED_TOOLS_KEY = "__progressive_discovered_tool_names__"
_DISCOVERED_TOOL_FINGERPRINTS_KEY = "__progressive_discovered_tool_fingerprints__"
_DISCOVERY_TRACE_KEY = "__progressive_tool_discovery_trace__"
_DISCOVERED_TOOLS_TURN_KEY = "__progressive_discovered_tool_turn_index__"
_TOOL_DISCOVERY_STATE_KEY = "__progressive_tool_discovery__"
_TOOL_DISCOVERY_TOOLS_SECTION = "progressive_tool_discovery_selected_tools"
_DEFERRED_TOOL_ATTACHMENT_SECTION = "progressive_deferred_tools"
_DEFERRED_TOOL_ATTACHMENT_SOURCE = "progressive_tool_rail"
_DEFERRED_TOOL_CATALOG_STATE_KEY = "__progressive_deferred_tool_catalog__"
_DEFERRED_TOOL_CATALOG_METADATA_KEY = "catalog_tools"
_DEFERRED_TOOL_CATALOG_FINGERPRINTS_METADATA_KEY = "catalog_fingerprints"
_DEFERRED_TOOL_CATALOG_INITIAL_FINGERPRINTS_STATE_KEY = "initial_fingerprints"
_DEFERRED_TOOL_CATALOG_FINGERPRINTS_STATE_KEY = "fingerprints"
_DEFERRED_TOOL_REGISTRY_REVISION_METADATA_KEY = "registry_revision"
_DEFERRED_TOOL_FULL_SNAPSHOT_INTERVAL = 10
_NESTED_TOOL_CALL_KEY = "__progressive_nested_tool_call__"
_JIUWENSWARM_USER_MESSAGE_PREFIX = "You receive a new message:"


class ProgressiveToolRail(DeepAgentRail):
    """Rail that enables progressive tool discovery and callable-tool filtering."""

    priority = 90

    def __init__(self, config: DeepAgentConfig):
        """Initialize ProgressiveToolRail.

        Args:
            config: DeepAgentConfig containing progressive tool settings.
        """
        super().__init__()
        self._config = config

        self._meta_tool_names: Set[str] = set()
        # Meta tools this rail actually registered, mapped from tool name to the
        # exact card that was stored. The name is the ability-manager key, while
        # the card identity tells uninit whether this rail is still the owner or
        # another rail has since taken the name over.
        self._owned_tool_cards: Dict[str, ToolCard] = {}
        self._cached_all_tool_infos: List[ToolInfo] = []
        self._tool_search_index: Optional[BM25ToolIndex] = None
        self._tool_search_index_revision: Optional[int] = None
        self._tool_search_registry: Any = None
        self._jev_disabled = False

    def init(self, agent) -> None:
        """Register progressive meta tools to resource manager and ability manager."""
        if str(
            getattr(self._config, "tool_discovery_backend", "bm25")
        ).strip().lower() == "jev":
            self._jev_disabled = not has_decisions_api_key(
                getattr(self._config, "tool_discovery_api_key", None)
            )
        language = getattr(self._config, "language", "cn") or "cn"
        agent_id = getattr(getattr(agent, "card", None), "id", None)

        tools = [
            ToolSearchTool(
                search_tools=self._search_tools,
                language=language,
                agent_id=agent_id,
                result_limit=getattr(
                    self._config,
                    "tool_search_limit",
                    DEFAULT_TOOL_SEARCH_LIMIT,
                ),
            ),
            ToolCallTool(
                call_tool=self._call_discovered_tool,
                language=language,
                agent_id=agent_id,
            ),
        ]

        # Meta tools are always directly callable.  Mark this explicitly on
        # the card so a progressive registration policy cannot accidentally
        # hide the wrapper that is needed to execute search results.
        for tool in tools:
            tool.card.exposure = ToolExposure.DIRECT
            tool.card.set_exposure_declared(True)

        self._meta_tool_names = {tool.card.name for tool in tools}

        if hasattr(agent, "ability_manager"):
            for tool in tools:
                try:
                    result = agent.ability_manager.add_ability(tool.card, tool)
                    if result.added:
                        self._owned_tool_cards[tool.card.name] = tool.card
                except Exception as exc:
                    logger.warning(
                        f"[ProgressiveToolRail] failed to register tool '{tool.card.name}': {exc}"
                    )

        # Keep the registry reference, but defer the initial BM25 build until
        # DeepAgent has initialized every startup-stage rail.  Tool exposure is
        # assigned by AbilityManager at registration time; this rail only
        # consumes that decision for visibility and indexing.
        self._tool_search_registry = getattr(agent, "ability_manager", None)

    def finalize_startup(self, agent: Any) -> None:
        """Build the initial BM25 catalog after startup registrations finish."""
        self._tool_search_registry = getattr(agent, "ability_manager", None)
        self._rebuild_tool_search_index(agent)

    async def finalize_startup_async(self, agent: Any) -> None:
        """Materialize lazy tool schemas, then build the startup catalog.

        MCP-backed cards are materialized by ``AbilityManager.list_tool_info``;
        calling it here ensures those startup tools are included before the
        first BM25 snapshot instead of waiting for the first model request.
        """
        ability_manager = getattr(agent, "ability_manager", None)
        list_tool_info = getattr(ability_manager, "list_tool_info", None)
        if callable(list_tool_info):
            try:
                await list_tool_info()
            except Exception as exc:
                logger.warning(
                    "[ProgressiveToolRail] failed to materialize startup tool schemas: %s",
                    exc,
                )
        self.finalize_startup(agent)

    def uninit(self, agent) -> None:
        """Remove the meta tools this rail still owns."""
        if hasattr(agent, "ability_manager"):
            for tool_name, tool_card in list(self._owned_tool_cards.items()):
                if agent.ability_manager.get(tool_name) is not tool_card:
                    # Another owner re-registered the name after this rail did
                    # and now holds both the card and the live instance; tearing
                    # it down here would unregister that owner's tool.
                    continue
                try:
                    agent.ability_manager.remove_ability(tool_name)
                except Exception as exc:
                    logger.warning(
                        f"[ProgressiveToolRail] failed to remove tool '{tool_name}' "
                        f"from ability_manager: {exc}"
                    )

        self._owned_tool_cards.clear()
        self._meta_tool_names.clear()
        self._cached_all_tool_infos = []
        self._tool_search_index = None
        self._tool_search_index_revision = None
        self._tool_search_registry = None

    async def before_invoke(self, ctx: AgentCallbackContext) -> None:
        """Cache the full registered tool inventory for discovery authorization."""
        session = getattr(ctx, "session", None)
        if session is not None:
            turn_index = self._user_message_count(getattr(ctx, "inputs", None))
            if session.get_state(_DISCOVERED_TOOLS_TURN_KEY) != turn_index:
                self._set_discovered_tools(session, [])
                session.update_state({_DISCOVERED_TOOLS_TURN_KEY: turn_index})
        self._cached_all_tool_infos = await self._list_tool_infos(ctx.agent)
        self._ensure_initial_deferred_catalog(ctx)

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        _ = ctx

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        """Update builder sections and filter callable tools for the current turn."""
        # --------------------------------------------------
        # DEBUG 1: ctx.agent 到底是谁，builder 挂没挂上
        # --------------------------------------------------
        logger.info(
            "[ProgressiveToolRail][DEBUG] before_model_call | agent_type=%s | has_system_prompt_builder=%s",
            type(getattr(ctx, "agent", None)).__name__,
            hasattr(getattr(ctx, "agent", None), "system_prompt_builder"),
        )

        builder = self._get_prompt_builder(ctx)
        agent = getattr(ctx, "agent", None)
        current_deferred_tools = self._collect_deferred_tool_descriptions(agent)
        initial_deferred_tools = self._ensure_initial_deferred_catalog(
            ctx,
            current_tools=current_deferred_tools,
        )

        backend = str(
            getattr(self._config, "tool_discovery_backend", "bm25")
        ).strip().lower()
        routed_tools: List[ToolInfo] = []
        manual_search_fallback = backend == "jev" and self._jev_disabled
        if backend == "jev" and not self._jev_disabled:
            routed_tools, manual_search_fallback = await self._route_deferred_tools(ctx)
            if manual_search_fallback:
                self._jev_disabled = True
        use_automatic_discovery = backend == "jev" and not manual_search_fallback
        prompt_backend = "bm25" if manual_search_fallback else backend

        rules_section = self._build_progressive_tool_rules_section(
            deferred_tool_descriptions=initial_deferred_tools,
            tool_discovery_backend=prompt_backend,
        )

        builder.add_section(rules_section)
        if use_automatic_discovery:
            rendered_tools = self._render_routed_tools(routed_tools)
            english = (
                "The configured retrieval backend automatically discovers deferred tools. "
                "Tools listed below, if any, were selected by automatic retrieval. "
                "always invoke a selected deferred tool by calling the `tool_call` wrapper "
                "using its exact name and complete schema. Never issue a tool call whose "
                "function name is the deferred tool name itself. If none fit, do not use `tool_search`."
            )
            chinese = (
                "以下工具（如果有）由自动检索选出。必须通过 `tool_call` 使用准确名称和完整 schema 调用。"
                "如果没有合适工具，不要使用 `tool_search`。"
            )
            builder.add_section(
                PromptSection(
                    name=_TOOL_DISCOVERY_TOOLS_SECTION,
                    content={
                        "en": f"## Automatically discovered deferred tools\n{english}\n\n{rendered_tools}",
                        "cn": f"## 自动发现的延迟工具\n{chinese}\n\n{rendered_tools}",
                    },
                    priority=74,
                    category="system_prompt",
                )
            )
        else:
            builder.remove_section(_TOOL_DISCOVERY_TOOLS_SECTION)
            await self._sync_deferred_tool_attachment(
                ctx,
                current_tools=current_deferred_tools,
            )

        inputs = getattr(ctx, "inputs", None)
        tools = getattr(inputs, "tools", None)
        if not isinstance(tools, list):
            logger.info(
                "[ProgressiveToolRail][DEBUG] before_model_call | inputs.tools is not a list: %s",
                type(tools).__name__,
            )
            return

        # --------------------------------------------------
        # DEBUG 2: 过滤前工具数量
        # --------------------------------------------------
        original_tool_names = [
            str(getattr(tool, "name", "") or "")
            for tool in tools
            if str(getattr(tool, "name", "") or "")
        ]
        logger.info(
            "[ProgressiveToolRail][DEBUG] before filter | tool_count=%s | first_20=%s",
            len(original_tool_names),
            original_tool_names[:20],
        )

        meta_visible_tools = set(self._meta_tool_names)
        direct_visible_tools = self._get_direct_tool_names(ctx.agent)
        if use_automatic_discovery:
            meta_visible_tools.discard("tool_search")
            direct_visible_tools.discard("tool_search")

        logger.info(
            "[ProgressiveToolRail][DEBUG] visibility | meta=%s | direct=%s",
            sorted(meta_visible_tools),
            sorted(direct_visible_tools),
        )

        filtered_tools: List[ToolInfo] = []

        for tool in tools:
            tool_name = str(getattr(tool, "name", "") or "")
            if not tool_name:
                continue

            if tool_name in meta_visible_tools:
                filtered_tools.append(tool)
                continue

            if tool_name in direct_visible_tools:
                filtered_tools.append(tool)
                continue

        filtered_tool_names = [
            str(getattr(tool, "name", "") or "")
            for tool in filtered_tools
            if str(getattr(tool, "name", "") or "")
        ]

        # --------------------------------------------------
        # DEBUG 3: 过滤后工具数量
        # --------------------------------------------------
        logger.info(
            "[ProgressiveToolRail][DEBUG] after filter | tool_count=%s | tools=%s",
            len(filtered_tool_names),
            filtered_tool_names,
        )
        inputs.tools = filtered_tools

    async def _route_deferred_tools(
        self,
        ctx: AgentCallbackContext,
    ) -> tuple[List[ToolInfo], bool]:
        """Select deferred tools for the latest user request."""
        backend = str(
            getattr(self._config, "tool_discovery_backend", "bm25")
        ).strip().lower()
        query = self._latest_user_request(getattr(ctx, "inputs", None))
        session = getattr(ctx, "session", None)
        if not query:
            self._set_discovered_tools(session, [])
            if session is not None:
                session.update_state({_TOOL_DISCOVERY_STATE_KEY: None})
            return [], False

        candidates = self._list_registered_deferred_tool_infos(
            getattr(ctx, "agent", None)
        )
        if not candidates:
            self._set_discovered_tools(session, [])
            if session is not None:
                session.update_state({_TOOL_DISCOVERY_STATE_KEY: None})
            return [], False

        fingerprints = {
            str(tool.name): BM25ToolIndex.tool_fingerprint(tool)
            for tool in candidates
        }
        catalog = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
        model = str(
            getattr(self._config, "tool_discovery_model", "typesafe/jev-1.13")
        )
        max_tools = min(
            10,
            max(1, int(getattr(self._config, "tool_discovery_max_tools", 10))),
        )
        try:
            min_score = float(getattr(self._config, "tool_discovery_min_score", 0.01))
        except (TypeError, ValueError):
            min_score = 0.01
        if not math.isfinite(min_score):
            min_score = 0.01
        min_score = min(1.0, max(0.0, min_score))
        turn_index = self._user_message_count(getattr(ctx, "inputs", None))
        previous = session.get_state(_TOOL_DISCOVERY_STATE_KEY) if session is not None else None
        cache_hit = (
            isinstance(previous, dict)
            and previous.get("query") == query
            and previous.get("turn_index") == turn_index
            and previous.get("catalog") == catalog
            and previous.get("backend") == backend
            and previous.get("model") == model
            and previous.get("max_tools") == max_tools
            and previous.get("min_score") == min_score
            and previous.get("api_base")
            == getattr(self._config, "tool_discovery_api_base", None)
        )
        scores: Dict[str, float] = {}
        selection_backend = backend
        fallback_error = ""
        manual_search_fallback = False
        conversation: List[Dict[str, Any]] = []
        by_name = {str(tool.name): tool for tool in candidates}
        if cache_hit:
            selected_names = previous.get("names", [])
            selected = [by_name[name] for name in selected_names if name in by_name]
            previous_scores = previous.get("scores", {})
            if isinstance(previous_scores, dict):
                scores = {
                    str(name): float(score)
                    for name, score in previous_scores.items()
                    if name in selected_names
                }
            selection_backend = str(previous.get("selection_backend", backend))
            fallback_error = str(previous.get("fallback_error", ""))
            manual_search_fallback = bool(previous.get("manual_search_fallback"))
        else:
            conversation = self._conversation_context(getattr(ctx, "inputs", None))
            try:
                ranked = await rank_decisions_api_tools(
                    query=query,
                    tools=candidates,
                    conversation=conversation,
                    model=model,
                    api_key=getattr(self._config, "tool_discovery_api_key", None),
                    api_base=getattr(self._config, "tool_discovery_api_base", None),
                    max_tools=max_tools,
                    min_score=min_score,
                    include_no_tool=bool(
                        getattr(self._config, "tool_discovery_include_no_tool", True)
                    ),
                )
                selected = [item.tool for item in ranked]
                scores = {str(item.tool.name): item.score for item in ranked}
            except Exception as exc:
                fallback_error = str(exc)
                selection_backend = "bm25_tool_search_fallback"
                manual_search_fallback = True
                self._jev_disabled = True
                if backend != "jev":
                    logger.warning(
                        "[ProgressiveToolRail] discovery backend %s failed; enabling model-directed tool_search fallback: %s",
                        backend,
                        exc,
                    )
                selected = []

        selected_names = [str(tool.name) for tool in selected]
        selected_fingerprints = {
            name: fingerprints[name] for name in selected_names if name in fingerprints
        }
        if not (cache_hit and manual_search_fallback):
            self._set_discovered_tools(
                session,
                selected_names,
                fingerprints=selected_fingerprints,
            )
        if session is not None and not cache_hit:
            session.update_state(
                {
                    _TOOL_DISCOVERY_STATE_KEY: {
                        "query": query,
                        "turn_index": turn_index,
                        "catalog": catalog,
                        "backend": backend,
                        "model": model,
                        "max_tools": max_tools,
                        "min_score": min_score,
                        "api_base": getattr(self._config, "tool_discovery_api_base", None),
                        "names": selected_names,
                        "scores": scores,
                        "selection_backend": selection_backend,
                        "fallback_error": fallback_error,
                        "manual_search_fallback": manual_search_fallback,
                    }
                }
            )
            self._append_trace(
                session,
                {
                    "action": "automatic_discovery",
                    "query": query,
                    "backend": selection_backend,
                    "candidate_count": len(candidates),
                    "max_tools": max_tools,
                    "min_score": min_score,
                    "matched": selected_names,
                    "scores": scores,
                    "fallback_error": fallback_error,
                },
            )
        if not cache_hit and backend == "jev":
            logger.info(
                "[ToolDiscovery] JEV routing: %s",
                json.dumps(
                    {
                        "timestamp": datetime.now().astimezone().isoformat(
                            timespec="milliseconds"
                        ),
                        "session_id": self._get_attachment_session_id(ctx),
                        "deferred_tool_count": len(candidates),
                        "max_tools": max_tools,
                        "min_score": min_score,
                        "user_prompt": query,
                        "jev_context": conversation,
                        "selected_tools": [
                            {"name": name, "score": scores.get(name)}
                            for name in selected_names
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        elif not cache_hit:
            logger.info(
                "[ToolDiscovery] deferred tool routing | candidate_count=%s | backend=%s | matched=%s",
                len(candidates),
                selection_backend,
                selected_names,
            )
        return selected, manual_search_fallback

    @staticmethod
    def _latest_user_request(inputs: Any) -> str:
        messages = getattr(inputs, "messages", None) or []
        for message in reversed(messages):
            if not isinstance(message, UserMessage):
                continue
            content, _ = ProgressiveToolRail._clean_jev_user_content(
                getattr(message, "content", "")
            )
            if content:
                return content
        return ""

    @staticmethod
    def _clean_jev_user_content(raw_content: Any) -> tuple[str, bool]:
        """Strip JiuwenSwarm transport envelopes and injected system reminders."""
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            content = "\n".join(
                part if isinstance(part, str) else part["text"]
                for part in raw_content
                if isinstance(part, str)
                or (isinstance(part, dict) and isinstance(part.get("text"), str))
            )
        else:
            return "", False

        was_wrapped = False
        stripped = content.strip()
        if stripped.startswith(_JIUWENSWARM_USER_MESSAGE_PREFIX):
            payload = stripped[len(_JIUWENSWARM_USER_MESSAGE_PREFIX):].strip()
            try:
                envelope = json.loads(payload)
            except (TypeError, ValueError):
                envelope = None
            if (
                isinstance(envelope, dict)
                and envelope.get("type") == "user input"
                and isinstance(envelope.get("content"), str)
            ):
                content = envelope["content"]
                was_wrapped = True

        content = re.sub(
            r"<system-reminder>.*?</system-reminder>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        return content, was_wrapped

    @staticmethod
    def _conversation_context(inputs: Any) -> List[Dict[str, Any]]:
        """Return user dialog for the JEV router.

        System instructions and message metadata are intentionally omitted;
        the router needs conversation context, not the agent's private prompt.
        Reasoning content and non-text multimodal payloads are also excluded.
        """
        conversation: List[Dict[str, Any]] = []
        messages = getattr(inputs, "messages", None) or []
        for message in messages:
            role = str(getattr(message, "role", "") or "").strip().lower()
            if role != "user":
                continue

            content, was_wrapped = ProgressiveToolRail._clean_jev_user_content(
                getattr(message, "content", "")
            )
            if not content:
                continue
            if (
                was_wrapped
                and conversation
                and conversation[-1]["content"] == content
            ):
                continue
            conversation.append({"role": role, "content": content})
        return conversation

    @staticmethod
    def _user_message_count(inputs: Any) -> int:
        return sum(
            isinstance(message, UserMessage)
            for message in (getattr(inputs, "messages", None) or [])
        )

    @staticmethod
    def _render_routed_tools(tools: List[ToolInfo]) -> str:
        if not tools:
            return "(No deferred tool was selected.)"
        rendered = []
        for tool in tools:
            parameters = tool.parameters
            if inspect.isclass(parameters) and issubclass(parameters, BaseModel):
                parameters = parameters.model_json_schema()
            rendered.append(
                f"### {tool.name}\n{tool.description}\n"
                f"Parameters: {json.dumps(parameters, ensure_ascii=False, default=str)}"
            )
        return "\n\n".join(rendered)

    @staticmethod
    def _get_direct_tool_names(agent: Any) -> Set[str]:
        """Return registered cards whose exposure policy is direct."""
        ability_manager = getattr(agent, "ability_manager", None)
        list_abilities = getattr(ability_manager, "list", None)
        if not callable(list_abilities):
            return set()
        try:
            abilities = list_abilities() or []
        except Exception:
            return set()
        if not isinstance(abilities, (list, tuple)):
            return set()
        direct_names: Set[str] = set()
        for card in abilities:
            if not isinstance(card, ToolCard):
                continue
            name = str(getattr(card, "name", "") or "")
            exposure = getattr(card, "exposure", ToolExposure.DIRECT)
            if name and exposure == ToolExposure.DIRECT:
                direct_names.add(name)
        return direct_names

    async def _list_tool_infos(self, agent) -> List[ToolInfo]:
        """List all tool infos currently registered on the agent."""
        if not hasattr(agent, "ability_manager"):
            return []
        try:
            tool_infos = await agent.ability_manager.list_tool_info()
            return list(tool_infos or [])
        except Exception as exc:
            logger.warning(f"[ProgressiveToolRail] failed to list tool infos: {exc}")
            return []

    async def _list_all_tool_infos(self) -> List[ToolInfo]:
        """Return cached full tool inventory."""
        return list(self._cached_all_tool_infos or [])

    def _list_registered_deferred_tool_infos(self, agent: Any) -> List[ToolInfo]:
        """Build index documents from registered deferred ToolCards."""
        ability_manager = getattr(agent, "ability_manager", None) or self._tool_search_registry
        list_abilities = getattr(ability_manager, "list", None)
        if not callable(list_abilities):
            return []

        documents: List[ToolInfo] = []
        try:
            abilities = list_abilities() or []
        except Exception as exc:
            logger.warning(
                f"[ProgressiveToolRail] failed to list registered cards for BM25: {exc}"
            )
            return []
        if not isinstance(abilities, (list, tuple)):
            return []

        for card in abilities:
            if not isinstance(card, ToolCard):
                continue
            if str(getattr(card, "name", "") or "") in self._meta_tool_names:
                continue

            exposure = getattr(card, "exposure", ToolExposure.DIRECT)
            if exposure != ToolExposure.DEFERRED:
                continue

            try:
                documents.append(card.tool_info())
            except Exception as exc:
                logger.warning(
                    "[ProgressiveToolRail] failed to convert deferred card '%s' "
                    "to ToolInfo: %s",
                    getattr(card, "name", ""),
                    exc,
                )
        return documents

    def _rebuild_tool_search_index(
        self,
        agent: Any = None,
        tool_infos: Optional[List[ToolInfo]] = None,
    ) -> None:
        """Build and store a new immutable BM25 index."""
        if tool_infos is None:
            tool_infos = self._list_registered_deferred_tool_infos(agent)

        self._tool_search_index = BM25ToolIndex.build(tool_infos)
        ability_manager = getattr(agent, "ability_manager", None) or self._tool_search_registry
        revision = getattr(ability_manager, "registry_revision", None)
        self._tool_search_index_revision = (
            int(revision) if isinstance(revision, int) else None
        )

    def _get_current_tool_search_documents(self, agent: Any = None) -> List[ToolInfo]:
        """Read the current deferred catalog used by BM25 and invalidation."""
        ability_manager = (
            getattr(agent, "ability_manager", None) or self._tool_search_registry
        )
        if callable(getattr(ability_manager, "list", None)):
            return self._list_registered_deferred_tool_infos(agent)

        return [
            tool
            for tool in self._cached_all_tool_infos
            if str(getattr(tool, "name", "") or "") not in self._meta_tool_names
        ]

    def _ensure_tool_search_index(self, agent: Any = None) -> None:
        """Reuse the index only while its complete deferred catalog is unchanged."""
        ability_manager = (
            getattr(agent, "ability_manager", None) or self._tool_search_registry
        )
        revision = getattr(ability_manager, "registry_revision", None)
        normalized_revision = int(revision) if isinstance(revision, int) else None
        current_documents = self._get_current_tool_search_documents(agent)
        current_fingerprint = BM25ToolIndex.fingerprint(current_documents)

        if self._tool_search_index is not None and (
            self._tool_search_index.source_fingerprint == current_fingerprint
            and normalized_revision == self._tool_search_index_revision
        ):
            # The registry revision covers normal registration changes; the
            # metadata fingerprint additionally catches in-place schema edits
            # that do not advance that revision.
            return

        self._rebuild_tool_search_index(agent, tool_infos=current_documents)

    async def _get_real_tool_infos(self) -> List[ToolInfo]:
        """Return non-meta tools from the cached inventory."""
        infos = await self._list_all_tool_infos()
        return [
            tool
            for tool in infos
            if getattr(tool, "name", "") not in self._meta_tool_names
        ]

    async def _search_tools(
        self,
        query: str,
        limit: int = 5,
        session: Any = None,
    ) -> List[Dict[str, Any]]:
        """Search BM25 unless JEV failure enabled model-directed fallback."""
        backend = str(
            getattr(self._config, "tool_discovery_backend", "bm25")
        ).strip().lower()
        discovery_state = (
            session.get_state(_TOOL_DISCOVERY_STATE_KEY)
            if session is not None
            else None
        )
        if backend == "jev" and not (
            self._jev_disabled
            or (
                isinstance(discovery_state, dict)
                and discovery_state.get("manual_search_fallback")
            )
        ):
            raise RuntimeError(
                "tool_search is disabled while JEV discovery is active."
            )

        query = (query or "").strip().lower()
        if not query:
            return []

        self._ensure_tool_search_index()
        matched = (
            self._tool_search_index.search(query, limit=max(1, limit))
            if self._tool_search_index is not None
            else []
        )

        matched_names = [str(getattr(tool, "name", "") or "") for tool in matched]
        matched_fingerprints = {
            name: BM25ToolIndex.tool_fingerprint(tool)
            for name, tool in zip(matched_names, matched)
            if name
        }
        self._authorize_discovered_tools(
            session,
            matched_names,
            fingerprints=matched_fingerprints,
        )
        self._append_trace(
            session,
            {
                "action": "tool_search",
                "query": query,
                "limit": max(1, limit),
                "matched": matched_names,
            },
        )

        return [self._build_tool_schema(tool) for tool in matched]

    async def _call_discovered_tool(
        self,
        name: str,
        args: Dict[str, Any],
        session: Any,
        callback_context: AgentCallbackContext,
    ) -> ToolOutput:
        """Execute one searched deferred tool through AbilityManager.

        ``tool_call`` is a stable model-visible wrapper.  The target name is
        resolved only after checking the session's search authorization, then
        dispatched through the normal AbilityManager execution path so target
        tool rails, permissions, timeouts, and error handling remain active.
        """
        target_name = str(name or "").strip()
        if not target_name:
            return ToolOutput(success=False, error="tool_call requires a non-empty tool name.")

        if target_name not in self._get_discovered_tools(session):
            return ToolOutput(
                success=False,
                error=(
                    f"Deferred tool '{target_name}' must be selected by discovery "
                    "before it can be called."
                ),
            )

        ability_manager = getattr(callback_context.agent, "ability_manager", None)
        card = ability_manager.get(target_name) if ability_manager is not None else None
        if card is None:
            return ToolOutput(
                success=False,
                error=f"Deferred tool '{target_name}' is no longer registered.",
            )

        if isinstance(card, ToolCard) and getattr(card, "exposure", ToolExposure.DIRECT) != ToolExposure.DEFERRED:
            return ToolOutput(
                success=False,
                error=f"Tool '{target_name}' is not a deferred search result.",
            )

        discovered_fingerprints = self._get_discovered_tool_fingerprints(session)
        if discovered_fingerprints is not None:
            expected_fingerprint = discovered_fingerprints.get(target_name)
            if not expected_fingerprint:
                return ToolOutput(
                    success=False,
                    error=(
                        f"Deferred tool '{target_name}' must be searched again "
                        "before it can be called."
                    ),
                )
            try:
                current_fingerprint = BM25ToolIndex.tool_fingerprint(card.tool_info())
            except Exception as exc:
                self._invalidate_discovered_tools(session, {target_name})
                return ToolOutput(
                    success=False,
                    error=(
                        f"Deferred tool '{target_name}' schema is unavailable; "
                        f"search it again before calling it: {exc}"
                    ),
                )
            if current_fingerprint != expected_fingerprint:
                self._invalidate_discovered_tools(session, {target_name})
                return ToolOutput(
                    success=False,
                    error=(
                        f"Deferred tool '{target_name}' has changed; search it "
                        "again before calling it."
                    ),
                )

        self._append_trace(
            session,
            {
                "action": "tool_call",
                "name": target_name,
            },
        )

        outer_call = getattr(getattr(callback_context, "inputs", None), "tool_call", None)
        outer_call_id = str(getattr(outer_call, "id", "tool_call") or "tool_call")
        target_call = ToolCall(
            id=f"{outer_call_id}:target",
            type="function",
            name=target_name,
            arguments=json.dumps(args or {}, ensure_ascii=False),
        )

        callback_context.extra[_NESTED_TOOL_CALL_KEY] = True
        try:
            results = await ability_manager.execute(
                ctx=callback_context,
                tool_call=target_call,
                session=session,
                parallel_tool_calls=False,
            )
        except ToolInterruptException:
            raise
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))
        finally:
            callback_context.extra.pop(_NESTED_TOOL_CALL_KEY, None)

        if not results:
            return ToolOutput(
                success=False,
                error=f"Deferred tool '{target_name}' returned no execution result.",
            )

        target_result, target_message = results[0]
        if isinstance(target_result, ToolInterruptException):
            raise target_result

        # The target's message already holds its model-facing rendering (plus
        # any AFTER_TOOL_CALL rewrite). The wrapper relays that text to the
        # model out of band; the structured fields stay exactly what upper
        # layers stream as the tool result.
        target_message_content = str(getattr(target_message, "content", "") or "")
        target_success = getattr(target_result, "success", None)
        if target_success is False:
            return RelayedToolOutput(
                success=False,
                error=str(
                    getattr(target_result, "error", None)
                    or getattr(target_message, "content", None)
                    or target_result
                ),
                rendered_text=target_message_content,
            )

        if target_result is None and target_message_content.startswith(
            ("Ability execution error:", "Tool execution error:", "[Interrupted]")
        ):
            return ToolOutput(success=False, error=target_message_content)

        return RelayedToolOutput(
            success=True,
            data={
                "name": target_name,
                "result": target_result,
            },
            rendered_text=target_message_content,
        )

    def _authorize_discovered_tools(
        self,
        session: Any,
        names: List[str],
        *,
        fingerprints: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record search hits as callable through ``tool_call`` for this session."""
        if session is None:
            return
        current = self._get_discovered_tools(session)
        self._set_discovered_tools(
            session,
            list(dict.fromkeys(current + names)),
            fingerprints=fingerprints,
        )

    def _get_discovered_tools(self, session: Any) -> List[str]:
        if session is None:
            return []
        state = session.get_state(_DISCOVERED_TOOLS_KEY)
        if isinstance(state, list):
            return [str(item).strip() for item in state if str(item).strip()]
        return []

    def _get_discovered_tool_fingerprints(
        self,
        session: Any,
    ) -> Optional[Dict[str, str]]:
        if session is None:
            return None
        state = session.get_state(_DISCOVERED_TOOL_FINGERPRINTS_KEY)
        if not isinstance(state, dict):
            return None
        return {
            str(name).strip(): str(fingerprint).strip()
            for name, fingerprint in state.items()
            if str(name).strip() and str(fingerprint).strip()
        }

    def _set_discovered_tools(
        self,
        session: Any,
        names: List[str],
        *,
        fingerprints: Optional[Dict[str, str]] = None,
    ) -> None:
        if session is None:
            return
        normalized_names = list(
            dict.fromkeys(str(name).strip() for name in names if str(name).strip())
        )
        discovered_fingerprints = self._get_discovered_tool_fingerprints(session) or {}
        if fingerprints is not None:
            discovered_fingerprints.update(
                {
                    str(name).strip(): str(fingerprint).strip()
                    for name, fingerprint in fingerprints.items()
                    if str(name).strip() and str(fingerprint).strip()
                }
            )
        session.update_state(
            {
                _DISCOVERED_TOOLS_KEY: normalized_names,
                _DISCOVERED_TOOL_FINGERPRINTS_KEY: {
                    name: discovered_fingerprints[name]
                    for name in normalized_names
                    if name in discovered_fingerprints
                },
            }
        )

    def _invalidate_discovered_tools(self, session: Any, names: Set[str]) -> None:
        """Forget search authorizations for changed or removed deferred tools."""

        if session is None or not names:
            return
        current = self._get_discovered_tools(session)
        retained = [tool_name for tool_name in current if tool_name not in names]
        if retained != current:
            self._set_discovered_tools(session, retained)

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        """Allow direct tools and only wrapper-dispatched deferred tools."""
        inputs = getattr(ctx, "inputs", None)
        tool_name = str(getattr(inputs, "tool_name", "") or "")
        if not tool_name or tool_name in self._meta_tool_names:
            return

        session = getattr(ctx, "session", None)
        ability_manager = getattr(ctx.agent, "ability_manager", None)
        card = ability_manager.get(tool_name) if ability_manager is not None else None
        if not isinstance(card, ToolCard):
            if (
                ctx.extra.get(_NESTED_TOOL_CALL_KEY)
                and tool_name in self._get_discovered_tools(session)
                and tool_name in {
                    str(getattr(tool, "name", "") or "")
                    for tool in self._cached_all_tool_infos
                }
            ):
                return
            self._reject_tool_call(ctx, f"Unknown tool '{tool_name}'.")
            return

        exposure = getattr(card, "exposure", ToolExposure.DIRECT)
        if exposure == ToolExposure.DIRECT:
            return

        if (
            ctx.extra.get(_NESTED_TOOL_CALL_KEY)
            and tool_name in self._get_discovered_tools(session)
        ):
            # Isolated rail tests and lightweight adapters may expose only the
            # cached ToolInfo inventory. The real AbilityManager path still
            # requires a registered card above; the fallback only confirms
            # that the name came from this rail's search catalog.
            return

        self._reject_tool_call(
            ctx,
            f"Deferred tool '{tool_name}' must be selected by discovery before it can be called.",
        )

    @staticmethod
    def _reject_tool_call(ctx: AgentCallbackContext, message: str) -> None:
        """Prevent execution and provide a normal tool result to the model."""
        inputs = getattr(ctx, "inputs", None)
        tool_call = getattr(inputs, "tool_call", None)
        tool_call_id = str(getattr(tool_call, "id", "") or "")
        if not tool_call_id:
            tool_call_id = "unknown-tool-call"
        from openjiuwen.core.foundation.llm.schema.message import ToolMessage

        if hasattr(ctx, "extra"):
            ctx.extra["_skip_tool"] = True
        if inputs is not None:
            inputs.tool_result = {"error": message}
            inputs.tool_msg = ToolMessage(content=message, tool_call_id=tool_call_id)

    @staticmethod
    def _build_tool_schema(tool: ToolInfo) -> Dict[str, Any]:
        """Return the exact model-callable schema represented by ToolInfo."""
        return {
            "name": str(getattr(tool, "name", "") or ""),
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": ProgressiveToolRail._safe_serialize_parameters(
                getattr(tool, "parameters", None)
            ),
        }

    def _build_progressive_tool_rules_section(
        self,
        agent: Any = None,
        session: Any = None,
        deferred_tool_descriptions: Optional[Dict[str, str]] = None,
        tool_discovery_backend: Optional[str] = None,
    ):
        """Build stable rules plus the session's initial deferred catalog."""
        if deferred_tool_descriptions is None:
            deferred_tool_descriptions = self._collect_deferred_tool_descriptions(agent)
        _ = session
        return build_multilingual_progressive_tool_rules_section(
            deferred_tool_descriptions=deferred_tool_descriptions,
            tool_discovery_backend=tool_discovery_backend
            or str(getattr(self._config, "tool_discovery_backend", "bm25")),
        )

    async def _sync_deferred_tool_attachment(
        self,
        ctx: AgentCallbackContext,
        *,
        current_tools: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish only changes after the initial static catalog.

        The initial deferred-tool directory is captured in the stable system
        prompt for the session.  This attachment section is therefore empty on
        the first unchanged turn and is created only when the registered
        deferred catalog changes.  A full attachment snapshot is still used
        after context compaction or a bounded number of deltas.
        """

        session = getattr(ctx, "session", None)
        if session is None:
            return

        agent = getattr(ctx, "agent", None)
        manager = getattr(agent, "prompt_attachment_manager", None)
        add_section = getattr(manager, "add_section", None)
        if not callable(add_section):
            return

        session_id = self._get_attachment_session_id(ctx)
        if not session_id:
            return

        language = getattr(
            self._get_prompt_builder(ctx),
            "language",
            getattr(self._config, "language", "cn") or "cn",
        )
        current_tools = dict(
            current_tools
            if current_tools is not None
            else self._collect_deferred_tool_descriptions(agent)
        )
        current_tools = self._normalize_deferred_catalog(current_tools) or {}
        current_fingerprints = self._collect_deferred_tool_fingerprints(agent)
        registry_revision = self._get_registry_revision(agent)
        state = session.get_state(_DEFERRED_TOOL_CATALOG_STATE_KEY)
        if not isinstance(state, dict):
            state = {}

        initial_tools = state.get("initial_tools")
        if not isinstance(initial_tools, dict):
            initial_tools = dict(current_tools)
        initial_tools = {
            str(name): str(description or "")
            for name, description in initial_tools.items()
            if str(name).strip()
        }
        initial_fingerprints = self._normalize_deferred_catalog(
            state.get(_DEFERRED_TOOL_CATALOG_INITIAL_FINGERPRINTS_STATE_KEY)
        )
        if initial_fingerprints is None:
            initial_fingerprints = dict(current_fingerprints)

        previous_tools = state.get("tools")
        if not isinstance(previous_tools, dict):
            previous_tools = {}
        previous_tools = {
            str(name): str(description or "")
            for name, description in previous_tools.items()
            if str(name).strip()
        }
        previous_fingerprints = self._normalize_deferred_catalog(
            state.get(_DEFERRED_TOOL_CATALOG_FINGERPRINTS_STATE_KEY)
        )
        if previous_fingerprints is None:
            previous_fingerprints = {}
        initialized = bool(state.get("initialized"))
        previous_language = str(state.get("language") or "")
        try:
            # This is the last directory version, not the next attachment
            # write number.  It is incremented only after a real catalog diff
            # is found below.
            version = int(state.get("version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        version = max(version, 1)
        try:
            delta_count = int(state.get("delta_count", 0) or 0)
        except (TypeError, ValueError):
            delta_count = 0

        # Session state is normally persistent, but a hot reload, adapter
        # recreation, or a lightweight host can hand the rail a fresh Session
        # object.  The attachment manager still owns the latest materialized
        # directory in that case.  Recover that directory before calculating
        # the diff; otherwise the same removal/addition can be emitted again
        # with a new catalog version on every turn.
        existing_attachment = await self._get_deferred_tool_attachment(
            manager,
            session_id,
        )
        attachment_metadata = getattr(existing_attachment, "metadata", {}) or {}
        attachment_tools = self._normalize_deferred_catalog(
            attachment_metadata.get(_DEFERRED_TOOL_CATALOG_METADATA_KEY)
            if isinstance(attachment_metadata, dict)
            else None
        )
        if attachment_tools is not None:
            previous_tools = attachment_tools
            initialized = True
            attachment_fingerprints = self._normalize_deferred_catalog(
                attachment_metadata.get(_DEFERRED_TOOL_CATALOG_FINGERPRINTS_METADATA_KEY)
                if isinstance(attachment_metadata, dict)
                else None
            )
            if attachment_fingerprints is not None:
                previous_fingerprints = attachment_fingerprints
            attachment_language = (
                str(attachment_metadata.get("catalog_language") or "")
                if isinstance(attachment_metadata, dict)
                else ""
            )
            if attachment_language:
                previous_language = attachment_language
            try:
                attachment_version = int(
                    attachment_metadata.get("catalog_version", 0) or 0
                )
            except (TypeError, ValueError):
                attachment_version = 0
            if attachment_version:
                # The attachment stores the last materialized directory.  A
                # recovery snapshot must keep its directory version; only a
                # real diff below advances it.
                version = max(version, attachment_version)
            try:
                attachment_delta_count = int(
                    attachment_metadata.get("delta_count", 0) or 0
                )
            except (TypeError, ValueError):
                attachment_delta_count = 0
            if attachment_delta_count:
                delta_count = attachment_delta_count

        has_snapshot = False
        has_history_snapshot = getattr(manager, "has_history_snapshot", None)
        context = getattr(ctx, "context", None)
        if callable(has_history_snapshot) and context is not None:
            try:
                has_snapshot = bool(has_history_snapshot(context, session_id))
            except Exception as exc:
                logger.debug(
                    "[ProgressiveToolRail] failed to inspect attachment history: %s",
                    exc,
                )

        baseline_tools = previous_tools if initialized else initial_tools
        baseline_fingerprints = (
            previous_fingerprints if initialized else initial_fingerprints
        )
        added = {
            name: description
            for name, description in current_tools.items()
            if name not in baseline_tools
        }
        updated: Dict[str, str] = {}
        for name, description in current_tools.items():
            if name not in baseline_tools:
                continue
            if baseline_tools[name] != description:
                updated[name] = description
                continue
            if baseline_fingerprints.get(name) != current_fingerprints.get(name):
                updated[name] = description
        removed = sorted(name for name in baseline_tools if name not in current_tools)
        changed = bool(added or updated or removed)
        next_version = version + 1 if changed else version

        # Search authorizations are tied to the catalog entry that was
        # returned. Keep authorizations for unchanged tools, but invalidate
        # entries whose deferred card was modified or removed before the next
        # model call. This is the runtime counterpart of the directory update
        # wording: an old search result must not revive a changed/deleted tool.
        invalidated_names = set(updated) | set(removed)
        if invalidated_names:
            self._invalidate_discovered_tools(session, invalidated_names)

        if not initialized:
            # The initial static prompt is the baseline.  If registration
            # changed between startup and the first model call, report that
            # difference as the first attachment; otherwise emit nothing.
            if not changed:
                session.update_state(
                    {
                        _DEFERRED_TOOL_CATALOG_STATE_KEY: {
                            **state,
                            "initialized": True,
                            "version": version,
                            "delta_count": 0,
                            "language": str(language),
                            "initial_tools": dict(initial_tools),
                            "tools": dict(current_tools),
                            _DEFERRED_TOOL_CATALOG_INITIAL_FINGERPRINTS_STATE_KEY: dict(
                                initial_fingerprints
                            ),
                            _DEFERRED_TOOL_CATALOG_FINGERPRINTS_STATE_KEY: dict(
                                current_fingerprints
                            ),
                            _DEFERRED_TOOL_REGISTRY_REVISION_METADATA_KEY: registry_revision,
                        }
                    }
                )
                return

            force_snapshot = False
            version = next_version
            next_delta_count = 1
            mode = "delta"
            content = render_deferred_tool_catalog_delta(
                added,
                updated,
                removed,
                language=str(language),
                version=version,
            )
        else:
            force_snapshot = (
                not has_snapshot
                or previous_language != str(language)
                or delta_count >= _DEFERRED_TOOL_FULL_SNAPSHOT_INTERVAL
            )
            if initialized and not force_snapshot and not changed:
                return

            if force_snapshot:
                version = next_version
                content = render_deferred_tool_catalog_snapshot(
                    current_tools,
                    language=str(language),
                    version=version,
                )
                mode = "snapshot"
                next_delta_count = 0
            else:
                version = next_version
                content = render_deferred_tool_catalog_delta(
                    added,
                    updated,
                    removed,
                    language=str(language),
                    version=version,
                )
                mode = "delta"
                next_delta_count = delta_count + 1

        try:
            await add_section(
                session_id=session_id,
                section=_DEFERRED_TOOL_ATTACHMENT_SECTION,
                content=content,
                kind=PromptAttachmentKind.TOOL,
                source=_DEFERRED_TOOL_ATTACHMENT_SOURCE,
                priority=75,
                metadata={
                    "catalog_version": version,
                    "catalog_mode": mode,
                    "catalog_language": str(language),
                    "delta_count": next_delta_count,
                    _DEFERRED_TOOL_CATALOG_METADATA_KEY: dict(current_tools),
                    _DEFERRED_TOOL_CATALOG_FINGERPRINTS_METADATA_KEY: dict(
                        current_fingerprints
                    ),
                    _DEFERRED_TOOL_REGISTRY_REVISION_METADATA_KEY: registry_revision,
                },
                content_kind="text/markdown",
            )
        except Exception as exc:
            logger.warning(
                "[ProgressiveToolRail] failed to update deferred-tool attachment: %s",
                exc,
            )
            return

        session.update_state(
            {
                _DEFERRED_TOOL_CATALOG_STATE_KEY: {
                    "initialized": True,
                    "version": version,
                    "delta_count": next_delta_count,
                    "language": str(language),
                    "initial_tools": dict(initial_tools),
                    "tools": dict(current_tools),
                    _DEFERRED_TOOL_CATALOG_INITIAL_FINGERPRINTS_STATE_KEY: dict(
                        initial_fingerprints
                    ),
                    _DEFERRED_TOOL_CATALOG_FINGERPRINTS_STATE_KEY: dict(
                        current_fingerprints
                    ),
                    _DEFERRED_TOOL_REGISTRY_REVISION_METADATA_KEY: registry_revision,
                }
            }
        )

    @staticmethod
    def _normalize_deferred_catalog(value: Any) -> Optional[Dict[str, str]]:
        """Normalize a serialized deferred-tool directory.

        ``None`` means that no recoverable catalog was supplied.  An empty
        dictionary is a valid catalog: it represents a session whose current
        deferred registry is empty and must not be confused with missing
        state.
        """

        if value is None:
            return None
        if not isinstance(value, dict):
            return None
        return dict(sorted({
            str(name): str(description or "")
            for name, description in value.items()
            if str(name).strip()
        }.items()))

    @staticmethod
    def _get_registry_revision(agent: Any = None) -> Optional[int]:
        """Return the ability-registry revision when the host provides it."""

        ability_manager = getattr(agent, "ability_manager", None)
        revision = getattr(ability_manager, "registry_revision", None)
        return int(revision) if isinstance(revision, int) else None

    @staticmethod
    async def _get_deferred_tool_attachment(
        manager: Any,
        session_id: str,
    ) -> Any:
        """Read the currently materialized deferred catalog attachment."""

        list_by_filter = getattr(manager, "list_by_filter", None)
        if not callable(list_by_filter):
            return None
        try:
            result = list_by_filter(
                session_id=session_id,
                section=_DEFERRED_TOOL_ATTACHMENT_SECTION,
                source=_DEFERRED_TOOL_ATTACHMENT_SOURCE,
            )
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, (list, tuple)) and result:
                return result[0]
        except Exception as exc:
            logger.debug(
                "[ProgressiveToolRail] failed to recover deferred-tool attachment: %s",
                exc,
            )
        return None

    @staticmethod
    def _get_attachment_session_id(ctx: AgentCallbackContext) -> Optional[str]:
        """Resolve a session id without requiring every callback test double."""

        session = getattr(ctx, "session", None)
        get_session_id = getattr(session, "get_session_id", None)
        if callable(get_session_id):
            try:
                session_id = get_session_id()
                if session_id:
                    return str(session_id)
            except Exception as exc:
                logger.debug(
                    "[ProgressiveToolRail] failed to resolve session id from "
                    "session.get_session_id: %s",
                    exc,
                )

        session_id = getattr(session, "session_id", None)
        if session_id:
            return str(session_id)

        context = getattr(ctx, "context", None)
        get_context_session_id = getattr(context, "session_id", None)
        if callable(get_context_session_id):
            try:
                session_id = get_context_session_id()
                if session_id:
                    return str(session_id)
            except Exception as exc:
                logger.debug(
                    "[ProgressiveToolRail] failed to resolve session id from "
                    "context.session_id: %s",
                    exc,
                )
        elif get_context_session_id:
            return str(get_context_session_id)
        return None

    def _collect_deferred_tool_descriptions(self, agent: Any = None) -> Dict[str, str]:
        """Read the current deferred-tool directory in deterministic order."""
        descriptions = {
            str(getattr(tool, "name", "") or ""): str(
                getattr(tool, "description", "") or ""
            )
            for tool in self._list_registered_deferred_tool_infos(agent)
            if str(getattr(tool, "name", "") or "")
        }
        return dict(sorted(descriptions.items()))

    def _collect_deferred_tool_fingerprints(self, agent: Any = None) -> Dict[str, str]:
        """Read schema-aware versions of the current deferred-tool directory."""
        fingerprints = {
            str(getattr(tool, "name", "") or ""): BM25ToolIndex.tool_fingerprint(tool)
            for tool in self._list_registered_deferred_tool_infos(agent)
            if str(getattr(tool, "name", "") or "")
        }
        return dict(sorted(fingerprints.items()))

    def _ensure_initial_deferred_catalog(
        self,
        ctx: AgentCallbackContext,
        *,
        current_tools: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Capture the session's immutable initial deferred-tool directory."""
        session = getattr(ctx, "session", None)
        agent = getattr(ctx, "agent", None)
        if session is None:
            return dict(
                current_tools
                if current_tools is not None
                else self._collect_deferred_tool_descriptions(agent)
            )

        state = session.get_state(_DEFERRED_TOOL_CATALOG_STATE_KEY)
        if not isinstance(state, dict):
            state = {}

        initial_tools = state.get("initial_tools")
        if isinstance(initial_tools, dict):
            return dict(sorted({
                str(name): str(description or "")
                for name, description in initial_tools.items()
                if str(name).strip()
            }.items()))

        if current_tools is None:
            current_tools = self._collect_deferred_tool_descriptions(agent)
        seed_tools = (
            state.get("tools")
            if state.get("initialized") and isinstance(state.get("tools"), dict)
            else current_tools
        )
        initial_tools = dict(sorted({
            str(name): str(description or "")
            for name, description in seed_tools.items()
            if str(name).strip()
        }.items()))
        session.update_state(
            {
                _DEFERRED_TOOL_CATALOG_STATE_KEY: {
                    **state,
                    "initial_tools": dict(initial_tools),
                }
            }
        )
        return initial_tools

    def _append_trace(self, session: Any, event: Dict[str, Any]) -> None:
        """Append progressive-tool discovery trace into session state."""
        if session is None:
            return
        trace = session.get_state(_DISCOVERY_TRACE_KEY)
        if not isinstance(trace, list):
            trace = []
        trace.append(event)
        session.update_state({_DISCOVERY_TRACE_KEY: trace})

    @staticmethod
    def _build_tool_summary(tool: ToolInfo, *, detail_level: int = 1) -> Dict[str, Any]:
        """Build structured tool summary payload."""
        name = str(getattr(tool, "name", "") or "")
        description = str(getattr(tool, "description", "") or "")
        parameters = getattr(tool, "parameters", None)

        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
        }

        if detail_level >= 2:
            payload["parameter_summary"] = ProgressiveToolRail._parameters_summary(parameters)

        if detail_level >= 3:
            payload["parameters"] = ProgressiveToolRail._safe_serialize_parameters(parameters)

        return payload

    @staticmethod
    def _safe_serialize_parameters(parameters: Any) -> Any:
        """Safely serialize tool parameter schema."""
        try:
            if inspect.isclass(parameters) and issubclass(parameters, BaseModel):
                try:
                    return parameters.model_json_schema()
                except Exception:
                    return str(parameters)
            if isinstance(parameters, dict):
                return parameters
            return str(parameters)
        except Exception as exc:
            logger.warning(f"[ProgressiveToolRail] failed to serialize parameters: {exc}")
            return str(parameters)

    @staticmethod
    def _parameters_summary(parameters: Any) -> str:
        """Build a short textual summary of parameters."""
        try:
            if inspect.isclass(parameters) and issubclass(parameters, BaseModel):
                fields = getattr(parameters, "model_fields", None)
                if isinstance(fields, dict):
                    names = list(fields.keys())
                    return f"fields: {', '.join(names)}" if names else "no declared fields"

            if isinstance(parameters, dict):
                props = parameters.get("properties")
                if isinstance(props, dict) and props:
                    return f"fields: {', '.join(props.keys())}"
                if parameters:
                    return f"schema keys: {', '.join(parameters.keys())}"
                return "empty schema"

            if parameters is None:
                return "no parameters"

            return str(parameters)
        except Exception as exc:
            logger.warning(f"[ProgressiveToolRail] failed to summarize parameters: {exc}")
            return "parameter summary unavailable"

    @staticmethod
    def _parameters_to_text(parameters: Any) -> str:
        """Flatten parameter summary and raw schema into searchable text."""
        summary = ProgressiveToolRail._parameters_summary(parameters)
        raw = ProgressiveToolRail._safe_serialize_parameters(parameters)
        return f"{summary} {raw}"

    @staticmethod
    def _get_prompt_builder(ctx: AgentCallbackContext) -> SystemPromptBuilder:
        """Fetch persistent SystemPromptBuilder from agent."""
        agent = getattr(ctx, "agent", None)
        if agent is None:
            raise RuntimeError("ProgressiveToolRail requires ctx.agent to exist.")

        builder = getattr(agent, "system_prompt_builder", None)
        if not isinstance(builder, SystemPromptBuilder):
            raise RuntimeError(
                "ProgressiveToolRail requires agent.system_prompt_builder "
                "to be an instance of SystemPromptBuilder."
            )
        return builder

__all__ = [
    "ProgressiveToolRail",
]
