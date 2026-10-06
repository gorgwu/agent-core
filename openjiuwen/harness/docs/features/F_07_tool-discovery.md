# Tool Discovery: Deferred Tool Selection

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-10-01 |
| Scope | BM25 and JEV System 1 tool selection for deferred tools |
| Specs | S_05 |
| Baseline | Existing BM25 `tool_search` and fixed `tool_call` contract tests |
| Refs | None |

## Background

Tool discovery selects relevant deferred tools before the main model call. It can use
a compatible System 1 tool-selection endpoint with a JEV model. BM25 remains the
default and is also the failure fallback for automatic discovery.

## Decisions

- Run retrieval at `ProgressiveToolRail.before_model_call` using the latest user
  message and current deferred `ToolInfo` inventory. Cache the result for that
  user-message count so further ReAct iterations reuse it without another discovery
  request or duplicate discovery log; the next user message triggers discovery.
- Send user text in `state` and one or more typed `choice` questions. Each
  question offers locally generated keys mapped to current tool records, plus a
  no-match choice. Choice groups contain no more than 255 total options.
- Rank registered tools from the response scores, discard zero-score results,
  and expose up to `tool_discovery_max_tools` candidates
  (default and maximum 10). Only keys included in the request map to registered
  tools; the no-match option is never exposed. If a group selects the no-match
  option, suppress every real-tool candidate from that group.
- Authorize selected tools through the existing session name/fingerprint state
  and show their full parameter schemas in a prompt section. Tool execution
  remains the existing `tool_call` → `AbilityManager.execute()` path.
- Configure `tool_discovery_backend` as `bm25` (default) or `jev`.
  `tool_discovery_model` is passed as the model ID to the System 1 endpoint;
  `tool_discovery_max_tools` caps results.
- JEV discovery uses a System 1 tool-selection API, not a chat-completions API.
  `TOOL_DISCOVERY_API_KEY` is sent as a Bearer credential. The optional
  `TOOL_DISCOVERY_API_BASE` is the full endpoint URL. Its service must accept the
  typed request (`model`, `state`, and `questions` containing `choice` criteria)
  and return an `answers` object with choice probabilities or a selected choice
  and confidence. A ChatGPT model ID or ordinary chat-completions URL alone is
  not sufficient; a compatible System 1 endpoint may route to any supported model.
- In JEV mode, expose `tool_call` and only the selected deferred tools; hide
  `tool_search`. At rail startup, check whether a JEV credential is present. If
  it is missing, use model-directed BM25 search without attempting JEV. If the
  first JEV request fails, disable JEV for that rail instance and switch to the
  normal BM25 `tool_search` workflow for subsequent turns. The model chooses
  its own search queries; the rail does not automatically select BM25 results.
  Restart the app/agent after fixing the credential or endpoint to try JEV again.
  A successful JEV response with no positive matches remains empty and does not
  trigger fallback.
  Select the `bm25` backend to use `tool_search` as the primary discovery mechanism.
- Record selected tool names and their probabilities in the discovery trace and
  debug log so a developer can inspect the retrieval result.

## Rejected Alternatives

- Do not let a router backend return arbitrary tool names or execute tools.
- Do not create a second registration or execution system.
- Do not remove the BM25 path or change default behavior.

## Verification and Limits

Unit tests cover typed request construction, custom endpoint selection, local
key mapping, grouped option ceiling, score filtering, automatic prompt exposure,
model-directed search after API failure, and valid no-match behavior.
