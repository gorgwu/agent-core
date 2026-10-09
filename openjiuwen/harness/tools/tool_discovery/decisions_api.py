# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JEV Decisions API adapter used by automatic tool discovery."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# Tool discovery credentials use the same public name as the runtime config.
_API_KEY_ENV = "TOOL_DISCOVERY_API_KEY"
_API_BASE_ENV = "TOOL_DISCOVERY_API_BASE"
_MAX_CHOICE_OPTIONS = 255


def has_decisions_api_key(api_key: str | None = None) -> bool:
    """Load dotenv once at rail startup and report whether JEV credentials exist."""
    load_dotenv(override=False)
    return bool(str(api_key or os.getenv(_API_KEY_ENV) or "").strip())


@dataclass(frozen=True)
class ScoredTool:
    """A registered deferred tool and its discovery relevance score."""

    tool: Any
    score: float


def _choice_request(
    *,
    query: str,
    tools: List[Any],
    model: str,
    conversation: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build one typed Choice per bounded group and retain local key mappings."""
    questions: Dict[str, Any] = {}
    key_to_tool: Dict[str, Any] = {}
    chunk_size = _MAX_CHOICE_OPTIONS

    for group_index, offset in enumerate(range(0, len(tools), chunk_size)):
        question_id = f"tool_group_{group_index:04d}"
        criteria: Dict[str, str] = {}
        for item_index, tool in enumerate(tools[offset : offset + chunk_size]):
            option_key = f"tool_{offset + item_index:06d}"
            name = str(getattr(tool, "name", "") or "")
            description = str(getattr(tool, "description", "") or "")
            criteria[option_key] = f"{name}: {description}".strip()
            key_to_tool[option_key] = tool
        questions[question_id] = {
            "type": "choice",
            "instructions": "Which deferred tool in this group best enables the user request? Choose the most relevant tool.",
            "criteria": criteria,
        }

    state: Dict[str, Any] = {"user_request": query}
    if conversation:
        state["conversation"] = conversation

    request = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    return request, key_to_tool


def _post_decisions(
    request: Dict[str, Any], api_key: str, timeout: float, api_base: str
) -> Dict[str, Any]:
    response = requests.post(
        api_base,
        json=request,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError("Decisions API response has no answers object")
    return payload


async def rank_decisions_api_tools(
    *,
    query: str,
    tools: List[Any],
    conversation: List[Dict[str, Any]] | None = None,
    model: str = "typesafe/jev-1.13",
    api_key: str | None = None,
    api_base: str | None = None,
    max_tools: int = 10,
    min_score: float = 0.0,
    request_timeout: float = 15.0,
) -> List[ScoredTool]:
    """Return the highest-scoring registered tools selected by JEV.

    Option keys are generated locally and mapped back to the original ToolInfo;
    names emitted outside those keys are never accepted as tool identities.
    Scores from each typed Choice probability distribution are ranked globally,
    and at most ``max_tools`` candidates are returned. The Choice request
    contains only registered tools, so the router must select a tool.
    """
    if not query.strip() or not tools:
        return []
    max_tools = min(10, max(1, int(max_tools)))
    min_score = float(min_score)
    if not math.isfinite(min_score) or not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be a finite number between 0 and 1")
    # Load a project-local .env on demand so users can configure discovery without
    # adding dotenv setup code to every application entrypoint.
    load_dotenv(override=False)
    token = api_key or os.getenv(_API_KEY_ENV)
    if not token:
        raise ValueError(f"{_API_KEY_ENV} is required for the JEV tool-discovery backend")

    endpoint = (
        api_base
        or os.getenv(_API_BASE_ENV)
        or _DECISIONS_URL
    )
    request, key_to_tool = _choice_request(
        query=query,
        tools=tools,
        model=model,
        conversation=conversation,
    )
    response = await asyncio.to_thread(
        _post_decisions, request, token, request_timeout, endpoint
    )
    scored: Dict[str, ScoredTool] = {}
    for question_id in request["questions"]:
        answer = response["answers"].get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            probabilities = {}
        choice = answer.get("choice")
        for key, probability in probabilities.items():
            if key not in key_to_tool:
                continue
            try:
                score = float(probability)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            tool = key_to_tool[key]
            name = str(getattr(tool, "name", "") or "")
            if name and (name not in scored or score > scored[name].score):
                scored[name] = ScoredTool(tool=tool, score=score)

        # A valid choice may arrive without a probability map; use its typed
        # choice and confidence as the available score in that case.
        if not probabilities and choice in key_to_tool:
            try:
                confidence = float(answer.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if math.isfinite(confidence) and 0.0 <= confidence <= 1.0:
                tool = key_to_tool[choice]
                name = str(getattr(tool, "name", "") or "")
                if name and (name not in scored or confidence > scored[name].score):
                    scored[name] = ScoredTool(tool=tool, score=confidence)

    ranked = sorted(
        (item for item in scored.values() if item.score >= min_score),
        key=lambda item: (-item.score, str(item.tool.name)),
    )
    return ranked[:max_tools]


async def select_deferred_tools(
    *,
    query: str,
    tools: List[Any],
    conversation: List[Dict[str, Any]] | None = None,
    model: str = "typesafe/jev-1.13",
    api_key: str | None = None,
    api_base: str | None = None,
    max_tools: int = 10,
    min_score: float = 0.0,
    request_timeout: float = 15.0,
) -> List[Any]:
    """Return the tools in the top JEV relevance scores."""
    ranked = await rank_decisions_api_tools(
        query=query,
        tools=tools,
        conversation=conversation,
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tools=max_tools,
        min_score=min_score,
        request_timeout=request_timeout,
    )
    return [item.tool for item in ranked]


__all__ = ["ScoredTool", "rank_decisions_api_tools", "select_deferred_tools"]
