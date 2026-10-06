# coding: utf-8
"""Bilingual description and input schema for the ``tool_call`` tool."""

from __future__ import annotations

from typing import Any, Dict

from openjiuwen.harness.prompts.tools.base import ToolMetadataProvider


DESCRIPTION: Dict[str, str] = {
    "cn": (
        "通过固定的 tool_call wrapper 执行已由自动发现或 tool_search 搜索到的 deferred 工具。"
        "必须将发现结果中的准确工具名称放入 name，将符合该工具完整 parameters schema 的参数放入 args。"
        "deferred 工具不会加入顶层 tools；绝不能直接调用 deferred 工具名称，必须调用本 wrapper。"
        "工具未变化时可以复用之前的搜索结果；如果工具已修改，应重新搜索获取最新 schema；"
        "如果当前目录后来显示该工具已删除，之前的搜索结果和授权立即失效，不能继续调用，也不能改用 task_tool 或子代理间接调用。"
    ),
    "en": (
        "Execute a deferred tool selected by automatic discovery or returned by tool_search. "
        "Always call this wrapper: put the exact discovered tool name in name and its schema-compatible "
        "arguments in args. Deferred tools are not top-level callable tools; never invoke their names directly. "
        "An unchanged tool may reuse a previous result; if a later "
        "directory update changes its schema, search again, and if it marks the tool as "
        "removed, the previous result and authorization are invalid; do not invoke it "
        "through task_tool or a subagent."
    ),
}


TOOL_CALL_PARAMS: Dict[str, Dict[str, str]] = {
    "name": {
        "cn": "自动发现或 tool_search 搜索结果中的准确 deferred 工具名称",
        "en": "Exact deferred tool name selected by discovery or returned by tool_search",
    },
    "args": {
        "cn": "按照自动发现或 tool_search 结果中的完整 parameters schema 填写工具参数；无参数工具使用空对象",
        "en": (
            "Arguments matching the complete parameters schema provided by discovery or tool_search; "
            "use an empty object for no-argument tools"
        ),
    },
}


def get_tool_call_input_params(language: str = "cn") -> Dict[str, Any]:
    params = TOOL_CALL_PARAMS
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": params["name"].get(language, params["name"]["cn"]),
            },
            "args": {
                "type": "object",
                "description": params["args"].get(language, params["args"]["cn"]),
                "additionalProperties": True,
            },
        },
        "required": ["name", "args"],
        "additionalProperties": False,
    }


class ToolCallMetadataProvider(ToolMetadataProvider):
    """Metadata provider used to build the model-visible ``tool_call`` card."""

    def get_name(self) -> str:
        return "tool_call"

    def get_description(self, language: str = "cn") -> str:
        return DESCRIPTION.get(language, DESCRIPTION["cn"])

    def get_input_params(self, language: str = "cn") -> Dict[str, Any]:
        return get_tool_call_input_params(language)
