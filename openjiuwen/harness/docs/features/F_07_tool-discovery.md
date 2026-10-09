# Tool Discovery: Deferred Tool Selection

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-10-01 |
| Scope | BM25 and System 1 tool selection for deferred tools |
| Specs | S_05 |
| Baseline | Existing BM25 `tool_search` and fixed `tool_call` contract tests |
| Refs | None |

## Background

Tool discovery selects relevant deferred tools before the main model call. It can
use a compatible System 1 tool-selection endpoint with a supported model. BM25
remains the default and is also the failure fallback for automatic discovery.

## Decisions

- Run retrieval at `ProgressiveToolRail.before_model_call` using the active
  user-message history, the latest user message, and current deferred
  `ToolInfo` inventory. Send only user text as conversation state; omit system
  messages, assistant messages, tool results, metadata, and non-text multimodal
  payloads. Cache the result for that
  user-message count so further ReAct iterations reuse it without another discovery
  request or duplicate discovery log; the next user message triggers discovery.
- Send the latest user text and active conversation in `state`, along with one
  or more typed `choice` questions. Each question offers locally generated keys
  mapped to current tool records. There is no no-tool choice. Choice groups
  contain no more than 255 total options.
- Rank registered tools from the response scores, keep scores greater than or
  equal to `tool_discovery_min_score` (default `0.00`), and expose up to
  `tool_discovery_max_tools` candidates (default and maximum 10). Only keys
  included in the request map to registered tools; only requested tool options
  can be selected.
- Authorize selected tools through the existing session name/fingerprint state
  and show their full parameter schemas in a prompt section. Tool execution
  remains the existing `tool_call` → `AbilityManager.execute()` path.
- Configure `tool_discovery_backend` as `bm25` (default) or `jev` to select System 1
  routing.
  `tool_discovery_model` is passed as the model ID to the System 1 endpoint;
  `tool_discovery_max_tools` caps results and `tool_discovery_min_score` is the
  configurable inclusive score floor. Set the floor to `0.00` to retain zero
  and positive scores, or `0.05` to keep scores of at least `0.05`.
- System 1 discovery uses a tool-selection API, not a chat-completions API.
  `TOOL_DISCOVERY_API_KEY` is sent as a Bearer credential. The optional
  `TOOL_DISCOVERY_API_BASE` is the full endpoint URL. Its service must accept the
  typed request (`model`, `state`, and `questions` containing `choice` criteria)
  and return an `answers` object with choice probabilities or a selected choice
  and confidence. A ChatGPT model ID or ordinary chat-completions URL alone is
  not sufficient; a compatible System 1 endpoint may route to any supported
  model.
- In System 1 mode, expose `tool_call` and only the selected deferred tools; hide
  `tool_search`. At rail startup, check whether `TOOL_DISCOVERY_API_KEY` is
  present. If it is missing, use model-directed BM25 search without attempting
  System 1. If the first request fails, disable System 1 for that rail instance
  and switch to the normal BM25 `tool_search` workflow for subsequent turns. The
  model chooses its own search queries; the rail does not automatically select
  BM25 results. Restart the app/agent after fixing the credential or endpoint to
  try System 1 again. A successful System 1 response with no usable tool scores
  remains empty and does not trigger fallback. Select the `bm25` backend to use
  `tool_search` as the primary discovery mechanism.
- Record selected tool names and their probabilities in the discovery trace and
  one debug log record so a developer can inspect the retrieval result. That
  record contains a timestamp, deferred candidate count, user prompt, user-message
  context sent to System 1, selected tool names, and their scores; it omits
  candidate choice descriptions.

## Rejected Alternatives

- Do not let a router backend return arbitrary tool names or execute tools.
- Do not create a second registration or execution system.
- Do not remove the BM25 path or change default behavior.

## Verification and Limits

Unit tests cover typed request construction, custom endpoint selection, local
key mapping, grouped option ceiling, score filtering, automatic prompt exposure,
model-directed search after API failure, and empty-result handling.
