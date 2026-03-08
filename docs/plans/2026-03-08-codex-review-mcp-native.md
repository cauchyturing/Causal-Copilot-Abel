# Codex Review: MCP-Native Redesign (2026-03-08)

> Context for future AI sessions. 2-round Claude↔Codex 5.4 review of v0.3.1 plan.
> Codex thread: 019ccf41-cd44-71a3-b1d9-663aed6ec455

## Starting Position (Claude's plan)
- 9 tools → 4 tools: discover, inspect_graph (new), diagnose_data, run_algorithm
- Delete: analyze, explain_graph, explain_result, estimate_effects, refine_graph, list_algorithms
- discover = hero tool (product core), not God tool anti-pattern

## Round 1: Codex Findings

**High:**
1. inspect_graph (was classify_graph) needs data_diagnosis for CPDAG branch.
   adj+names alone can't answer IDA eligibility. Must accept run_id or csv_data.
2. discover output lacks human-ready explanation. Weak LLMs can't interpret
   structured CPDAG/PAG data. Need summary, key_findings, limitations.
3. explain_graph logic is scientifically unsafe on ambiguous graphs (labels
   "root causes" on CPDAGs). Delete, don't port.

**Medium:**
4. run_id architecture exists but isn't real — tools just echo it, no cache retrieval.
5. Prompts still teach client to use deleted tools + manual selection workflow.
6. run_algorithm silently overrides user HP via resolvers — not honest for expert mode.

**Codex alternative suggestions:**
- Rename classify_graph → inspect_graph (broader than just classification)
- discover summary: deterministic from structured state, NOT from GlobalState.logging
- inspect_graph: return needs_more_input for CPDAG without diagnosis, not a guess
- run_algorithm: expose requested/effective/resolver_adjustments, add override flag
- No fake fields (edge_confidence=null, effect_estimate=null) — omit entirely

## Round 2: JSON Schema Contract

Codex drafted full JSON Schema (draft 2020-12) for all 4 tools:

**discover output (ok):** status, run_id, summary, key_findings, limitations,
algorithm_rationale, adjacency_matrix, node_names, edges, graph_kind,
identifiability, data_diagnosis, provenance, warnings

**inspect_graph output (ok | needs_more_input | error):**
- ok: graph_kind, graph_stats, identifiability, inference_policy
  (eligibility/method/reason/assumptions_used), query_assessment (optional),
  summary, key_findings, limitations
- needs_more_input: missing_inputs list, next_step instruction
- Input: run_id preferred, adj+names+diagnosis fallback

**run_algorithm output:** status, run_id, adjacency_matrix, edges, graph_kind,
identifiability, data_diagnosis, provenance (with requested_hyperparameters,
effective_hyperparameters, resolver_adjustments), warnings.
Input adds: allow_resolver_overrides (bool, default true)

**diagnose_data output:** status, diagnosis (DataDiagnosis), feature_names, warnings

**Runtime rules:**
- treatment + outcome are all-or-none, must be distinct, must exist in node_names
- CPDAG without diagnosis → needs_more_input (not a guess)
- run_id and explicit graph input are mutually exclusive in inspect_graph
- Expired/unknown run_id → error

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| inspect_graph not classify_graph | Does more than classify — also inference policy, query assessment |
| Deterministic summary, no LLM call | Cheap, stable, honest. GlobalState.logging is prompt artifacts |
| needs_more_input status | Honest > complete. Don't fake CPDAG answers |
| allow_resolver_overrides flag | Expert mode means actual control, not silent mutation |
| No fake fields | If you can't compute it, don't include the field |
| Collapse fake tools now, reintroduce when real | estimate_effects/refine_graph return when they do real work |
