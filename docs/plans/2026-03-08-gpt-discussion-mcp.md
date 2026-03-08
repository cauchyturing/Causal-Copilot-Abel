# GPT Discussion: MCP Server v0.3 Architecture (2026-03-08)

> Context for future AI sessions. 3-round Claude↔GPT discussion on Causal-Copilot MCP server design.

## Starting Position (Claude's v4 plan)

- Wrap main branch pipeline (28 algos, full LLM, Docker, causal inference) as MCP server
- 8 tools, 55+ resources, 2 prompts
- Bridge layer for sys.path management
- Phase 0 pipeline hardening (CI resolver, score resolver, LiNGAM bootstrap, 4 new wrappers)

## Round 1: GPT Pushback

**Accepted:**
- Option A (bridge from main) correct for v0.3. Clean package is v0.4+.
- CI/score resolvers are high-value, low-risk.
- Resources + prompts design is strong.

**Challenged:**
- **8 tools → 6 tools.** `list_algorithms` should be an MCP resource, not a tool. `select_algorithm` should be internal to `discover`, not exposed. Final 6: `discover`, `diagnose_data`, `run_algorithm`, `refine_graph`, `estimate_effects`, `explain_result`.
- **PDAG→DAG is scientific risk.** Can't just call `pdag2dag()` and pretend it's a true DAG. Need policy: DAG→full inference, CPDAG→IDA when linear-Gaussian else reject_ambiguous, PAG→never coerce.
- **Single Docker image unacceptable.** 6-8GB pytorch image as default is a non-starter for MCP ecosystem. Need three tiers: core-cpu (~800MB, no torch/LaTeX), full-cpu (~4GB), gpu (~8GB).
- **Offline mode too binary.** Three tiers: client_sampling (server asks client LLM), server_key (server's own LLM), off (rule-based).

## Round 2: Architecture Refinement

**New consensus items:**
- **Run artifacts with run_id + TTL.** Multi-step workflows (diagnose → run → refine → explain) need state. Each `discover`/`run_algorithm` returns a `run_id`. Other tools accept `run_id` to reference cached results instead of re-parsing CSV.
- **graph_kind + identifiability in every result.** Output must declare: "This is a CPDAG, not a DAG. Edges X→Y and Y→Z are identifiable. Edge A—B is ambiguous."
- **JSON Schema snapshots + golden tests.** Each tool has a frozen schema. Golden tests catch regressions. Schema changes require explicit version bump.

## Round 3: Final Priorities

**Minimum viable v0.3.0 (10 items, in order):**

1. Bridge layer from main (sys.path + CWD management)
2. 6 MCP tools: discover, diagnose_data, run_algorithm, refine_graph, estimate_effects, explain_result
3. Algorithm listing as MCP resources (not a tool)
4. CI test resolver + score function resolver + missing-data policy (Phase 0 hardening)
5. PDAG policy: DAG→full, CPDAG→IDA/reject, PAG→never coerce
6. graph_kind + identifiability + provenance + warnings in every tool result
7. Offline three-tier: client_sampling → server_key → off
8. Run artifacts: run_id + TTL for multi-step workflows
9. Three Docker images: core-cpu, full-cpu, gpu
10. JSON Schema snapshots + golden tests per tool

**Deferred to v0.4:**
- Clean pip-installable package (from feat/v0.1-m0 work)
- PyPI publishing
- Client-side sampling MCP protocol integration
- 4 new algorithm wrappers (GIN, BOSS, ExactSearch, RCD)
- LiNGAM native bootstrap
- Background knowledge auto-population
- LaTeX report generation via MCP

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Bridge from main, not clean package | Ship fast. Clean package is v0.4. |
| 6 tools not 8 | Fewer tools = less cognitive load for LLMs |
| PDAG policy module | Scientific credibility > convenience |
| Three Docker images | MCP ecosystem expects lightweight defaults |
| Run artifacts | Multi-step workflows need state |
| JSON Schema snapshots | Contract stability for downstream consumers |
