# Causal-Copilot MCP Server

Give any LLM or AI agent **causal superpowers** — discover cause-effect relationships, estimate treatment effects, test robustness, run counterfactuals — all from CSV data.

## Quickstart (30 seconds)

```bash
pip install causal-copilot[mcp]
causal-copilot mcp
```

That's it. The MCP server is running on stdio. Connect your AI client.

## Connect to Your AI Client

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "causal-copilot": {
      "command": "causal-copilot",
      "args": ["mcp"]
    }
  }
}
```

### Claude Code

Add to `.claude/settings.json` in your project:

```json
{
  "mcpServers": {
    "causal-copilot": {
      "command": "causal-copilot",
      "args": ["mcp"]
    }
  }
}
```

### Cursor / VS Code (Copilot)

Add to `.cursor/mcp.json` or VS Code MCP settings:

```json
{
  "servers": {
    "causal-copilot": {
      "command": "causal-copilot",
      "args": ["mcp"]
    }
  }
}
```

### Any MCP-Compatible Client

The server uses **stdio transport** (stdin/stdout JSON-RPC). Any MCP client works:

```bash
causal-copilot mcp
```

### Python (programmatic)

```python
from causal_copilot.mcp.server import mcp
mcp.run()  # stdio
# or
mcp.run(transport="sse", port=8000)  # HTTP SSE
```

## What Your AI Gets

### 12 Tools

| Tool | What It Does |
|------|-------------|
| **diagnose_data** | Understand your data: linearity, gaussianity, missingness, per-column types, descriptive stats, knowledge prompt |
| **discover** | Full autonomous causal discovery: data → algorithm selection → execution → graph refinement → causal graph |
| **run_algorithm** | Run a specific algorithm (PC, GES, LiNGAM, etc.) with explicit hyperparameters |
| **inspect_graph** | Classify graph (DAG/CPDAG/PAG), check identifiability, assess causal queries |
| **estimate_effect** | Estimate causal effects (ATE/ATT with CIs) — auto-selects method from 6 options |
| **refute_estimate** | Test robustness with 3 sensitivity methods |
| **estimate_counterfactual** | "What if X were different?" — counterfactual reasoning |
| **attribute_anomaly** | Root cause analysis: which variables drive anomalous values? |
| **attribute_distribution_change** | Why did Y's distribution shift between two datasets? |
| **simulate_intervention** | What happens if we intervene on X? Simulate the causal effect |
| **compute_feature_importance** | SHAP-based causal feature importance |
| **validate_graph** | Statistical test: is the discovered graph consistent with data? |

### 3 Prompts (Workflow Guidance)

| Prompt | Purpose |
|--------|---------|
| **causal_expert** | Expert identity: algorithm families, methodology, interpretation rules |
| **analyze_dataset** | Quick vs recommended vs expert workflow paths |
| **causal_analysis** | Complete 6-step knowledge-first workflow |

### 4 Resource Types

| Resource | Content |
|----------|---------|
| `causal://algorithms` | Index of all 28+ algorithms |
| `causal://algorithms/{name}` | Detailed profile per algorithm |
| `causal://hyperparameters/{name}` | Hyperparameter specs per algorithm |
| `causal://guides/{name}` | Methodology guides (ci-tests, score-functions, interpreting-graphs) |

## Recommended Workflow

The best results come from a **knowledge-first** approach:

```
User: "Analyze my data and find what causes customer churn"
                    ↓
AI calls: diagnose_data(csv)
  → Gets data characteristics + knowledge_prompt tailored to YOUR dataset
                    ↓
AI generates domain knowledge from the prompt
  (variable descriptions, known relationships, forbidden edges)
                    ↓
AI calls: discover(csv, domain_knowledge=..., query="What causes churn?")
  → Knowledge flows through 4 pipeline stages:
    Filter (algorithm selection) → Reranker (algorithm ranking)
    → HP Selector (hyperparameter tuning) → Judge (graph refinement)
  → Returns: causal graph + edge confidence + LLM pruning decisions
                    ↓
AI calls: inspect_graph(run_id, treatment="price", outcome="churn")
  → Checks: is the effect identifiable? What method to use?
                    ↓
AI calls: estimate_effect(run_id, treatment="price", outcome="churn")
  → ATE = -0.15 (95% CI: [-0.22, -0.08], p < 0.001)
  → "Increasing price by 1 unit decreases churn by 0.15"
                    ↓
AI calls: refute_estimate(run_id, treatment="price", outcome="churn")
  → Robust: ✓ all 3 sensitivity tests pass
```

## Install Extras for Full Power

```bash
# Core discovery (always included)
pip install causal-copilot[mcp]

# Add causal inference (estimate_effect, refute, counterfactual, etc.)
pip install causal-copilot[mcp,inference]

# Add LLM-powered algorithm selection (dramatically better results)
pip install causal-copilot[mcp,agent]

# Everything
pip install causal-copilot[mcp,inference,agent]
```

Without `[inference]`: discovery + graph analysis work; estimation tools return errors.
Without `[agent]`: discovery uses rule-based algorithm selection (5 algorithms) instead of LLM-powered selection (28+ algorithms).

## Key Features

### Honest Gate
The system won't lie about what it can identify. Before estimating effects:
- **DAG** → proceed (all effects identifiable)
- **CPDAG + linear Gaussian** → proceed (IDA method)
- **CPDAG + nonlinear** → reject (ambiguous directions)
- **PAG** → reject (latent confounders possible)

### Domain Knowledge Injection
Your domain knowledge directly influences:
1. Which algorithm is selected (Filter + Reranker LLM prompts)
2. How hyperparameters are tuned (HP Selector prompt)
3. Which edges the Judge confirms or rejects (LLM evaluation prompt)

### Bootstrap Edge Confidence
Every edge in the result has a confidence score (0-1) from bootstrap resampling.
`edge_confidence: {"X->Y": 0.95, "Z->W": 0.62}` — higher = more stable across resamples.

### LLM Pruning Transparency
The Judge's LLM decisions are visible:
`llm_pruning: {"confirmed": ["X->Y"], "rejected": ["Z->W"]}` — you can see exactly what the LLM added/removed.

### Algorithm Selection Transparency
When using LLM-powered selection, the reasoning is visible in `provenance.selection_reasoning`:
- `candidates` — which algorithms were considered
- `scores` — how they were scored
- `ranking_reason` — why the winner was chosen
- `hp_reasoning` — why each hyperparameter was set to its value

## Environment Variables

For LLM-powered algorithm selection:

```bash
export LLM_PROVIDER=openai        # or openrouter, ollama
export LLM_MODEL=gpt-4o           # or any supported model
export OPENAI_API_KEY=sk-...      # your API key
```

Without these, the system falls back to rule-based selection (still works, just fewer algorithms).

## Verify Installation

```bash
causal-copilot doctor       # Check all dependencies
causal-copilot quickstart   # Run demo analysis
causal-copilot mcp          # Start MCP server
```
