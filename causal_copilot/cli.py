"""Causal-Copilot CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_doctor(args):
    """Check environment and report installed capabilities."""
    print("Causal-Copilot Doctor")
    print("=" * 40)

    from causal_copilot import __version__

    print(f"Version: {__version__}")

    # Check core deps
    core_deps = ["numpy", "pandas", "scipy", "sklearn", "networkx", "statsmodels"]
    print("\nCore dependencies:")
    for dep in core_deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "ok")
            print(f"  + {dep} {ver}")
        except ImportError:
            print(f"  x {dep} NOT INSTALLED")

    # Check optional extras
    extras = {
        "agent": ["openai", "pydantic"],
        "inference": ["dowhy", "econml"],
        "viz": ["matplotlib", "seaborn"],
        "gpu": ["torch"],
        "web": ["gradio"],
        "algorithms": ["lingam", "tigramite"],
    }
    print("\nOptional extras:")
    for group, deps in extras.items():
        installed = []
        missing = []
        for dep in deps:
            try:
                __import__(dep)
                installed.append(dep)
            except ImportError:
                missing.append(dep)
        status = "+" if not missing else "~" if installed else "x"
        detail = f"missing: {', '.join(missing)}" if missing else "all installed"
        print(f"  {status} [{group}] {detail}")

    # Agent
    print("\nAgent:")
    try:
        from causal_copilot.agent.llm import PROVIDERS

        print("  + agent module available")
        for name in PROVIDERS:
            print(f"    {name}: {PROVIDERS[name]['base_url']}")
    except ImportError:
        print("  x agent not installed (pip install causal-copilot[agent])")

    if getattr(args, "llm", False):
        print("\nLLM Connectivity:")
        try:
            from causal_copilot.agent.llm import AgentLLM
            import os

            provider = os.getenv("LLM_PROVIDER", "openai")
            llm = AgentLLM(provider=provider)
            llm._client = llm._client.with_options(timeout=10.0, max_retries=0)
            response = llm.complete("Say 'ok' and nothing else.")
            print(f"  + {provider} ({llm.model}): connected")
        except Exception as e:
            print(f"  x connection failed: {e}")

    print("\nPlatform:")
    import platform

    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Arch: {platform.machine()}")


def cmd_version(args):
    from causal_copilot import __version__

    print(f"causal-copilot {__version__}")


def cmd_analyze(args):
    """Run causal discovery on a CSV file."""
    from causal_copilot import CausalCopilot

    try:
        copilot = CausalCopilot(planner=args.planner)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = copilot.analyze(
        args.data,
        algorithm=args.algorithm,
        timeout=args.timeout,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Result written to {out_path}")
    else:
        print(result.summary)
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")
        if result.provenance:
            p = result.provenance
            print(f"\nProvenance: {p.algorithm} | seed={p.seed} | {p.runtime_seconds:.1f}s")

    sys.exit(0 if result.status == "ok" else 1)


def cmd_benchmark(args):
    """Run benchmark evaluation."""
    try:
        from benchmarks.runner import run_benchmark
        from benchmarks.scenarios import ALL_SCENARIOS
    except ImportError:
        print(
            "Error: benchmarks package is not installed.\n"
            "The 'benchmark' command is only available when running from source.\n"
            "Clone the repo and run:  python -m causal_copilot benchmark ...",
            file=sys.stderr,
        )
        sys.exit(1)
    from causal_copilot.algorithms.registry import REGISTRY

    algorithms = [args.algorithm] if args.algorithm else list(REGISTRY.keys())
    scenarios = [args.scenario] if args.scenario else list(ALL_SCENARIOS.keys())

    results = []
    for algo in algorithms:
        for scenario in scenarios:
            print(f"Running {algo} on {scenario}...", end=" ", flush=True)
            r = run_benchmark(algo, scenario, seed=args.seed, timeout=args.timeout)
            results.append(r.to_dict())
            if r.status == "skipped":
                print(f"SKIPPED ({r.error})")
            elif r.status == "failed":
                print(f"FAILED ({r.error})")
            else:
                m = r.metrics
                label = f"[{r.output_type}] " if r.output_type != "dag" else ""
                print(f"{label}SHD={m.shd} Skel-F1={m.skeleton_f1:.3f} F1={m.f1:.3f} ({r.runtime_seconds:.1f}s)")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output}")

    # Summary table — primary metric depends on output type
    print(f"\n{'Algorithm':<20} {'Type':<6} {'Scenario':<15} {'SHD':>5} {'Skel-F1':>8} {'F1*':>6} {'Orient':>7}")
    print("-" * 72)
    print("  * F1 = directed; only meaningful for DAG outputs, not CPDAG/PAG")
    print()
    for r in results:
        if r.get("status") != "ok":
            status = r.get("status", "?").upper()
            print(f"{r['algorithm']:<20} {r.get('output_type','?'):<6} {r['scenario']:<15} {status:>5}")
        else:
            m = r["metrics"]
            print(f"{r['algorithm']:<20} {r['output_type']:<6} {r['scenario']:<15} {m['shd']:>5} {m['skeleton_f1']:>8.3f} {m['f1']:>6.3f} {m['orientation_accuracy']:>7.3f}")


def cmd_agent_analyze(args):
    """Run LLM-driven autonomous causal analysis."""
    try:
        from causal_copilot.agent import AgentCopilot
    except ImportError:
        print("Error: Agent mode requires: pip install causal-copilot[agent]", file=sys.stderr)
        sys.exit(1)

    try:
        agent = AgentCopilot(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = agent.analyze(
            args.data,
            query=args.query or "",
            timeout=args.timeout,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Result written to {out_path}")
    else:
        print(result.summary)
        interp = result.discovery_metadata.get("agent_interpretation")
        if interp:
            print(f"\nInterpretation: {interp}")
        if result.algorithm_selection_reason:
            print(f"\nAlgorithm selection: {result.algorithm_selection_reason}")
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")
        if result.provenance:
            p = result.provenance
            print(
                f"\nProvenance: {p.algorithm} | planner={p.planner}"
                f" | model={p.planner_model} | seed={p.seed}"
                f" | {p.runtime_seconds:.1f}s"
            )

    sys.exit(0 if result.status == "ok" else 1)


def cmd_quickstart(args):
    """Run a demo analysis on bundled synthetic data."""
    import numpy as np
    import pandas as pd

    from causal_copilot import CausalCopilot

    print("Causal-Copilot Quickstart")
    print("=" * 40)
    print("Generating synthetic data: X → Y → Z (linear, Gaussian noise)\n")

    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n) * 0.3
    z = 0.6 * y + rng.normal(size=n) * 0.4
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    copilot = CausalCopilot()
    result = copilot.analyze(df, seed=42)

    print(f"Status: {result.status}")
    print(f"Summary: {result.summary}")

    if result.adjacency_matrix is not None:
        print("\nAdjacency matrix (columns → rows):")
        cols = ["X", "Y", "Z"]
        header = "     " + "  ".join(f"{c:>4}" for c in cols)
        print(header)
        for i, row_label in enumerate(cols):
            vals = "  ".join(f"{int(result.adjacency_matrix[i, j]):>4}" for j in range(len(cols)))
            print(f"  {row_label:>2} {vals}")

    if result.warnings:
        print(f"\nWarnings: {result.warnings}")

    if result.provenance:
        p = result.provenance
        print(f"\nProvenance: {p.algorithm} | planner={p.planner} | seed={p.seed} | {p.runtime_seconds:.1f}s")

    print("\nAssumptions:")
    for a in result.assumptions:
        print(f"  - {a}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull result written to {out_path}")

    sys.exit(0 if result.status == "ok" else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="causal-copilot",
        description="Autonomous causal analysis from tabular data.",
    )
    sub = parser.add_subparsers(dest="command")

    p_doctor = sub.add_parser("doctor", help="Check environment and dependencies")
    p_doctor.add_argument("--llm", action="store_true", help="Test LLM connectivity")
    sub.add_parser("version", help="Show version")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run causal analysis on a CSV file")
    p_analyze.add_argument("data", help="Path to CSV file")
    p_analyze.add_argument("--output", "-o", help="Output JSON file path")
    p_analyze.add_argument("--algorithm", "-a", help="Force a specific algorithm")
    p_analyze.add_argument("--planner", "-p", default="rule", help="Planner: rule (default)")
    p_analyze.add_argument("--timeout", "-t", type=int, default=300, help="Timeout in seconds (default: 300)")
    p_analyze.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run benchmark evaluation")
    p_bench.add_argument("--algorithm", "-a", help="Run specific algorithm (default: all)")
    p_bench.add_argument("--scenario", help="Run specific scenario (default: all synthetic)")
    p_bench.add_argument("--output", "-o", help="Output JSON file")
    p_bench.add_argument("--timeout", "-t", type=int, default=120, help="Timeout per run (default: 120)")
    p_bench.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    # agent
    p_agent = sub.add_parser("agent", help="LLM-driven autonomous analysis")
    agent_sub = p_agent.add_subparsers(dest="agent_command")
    p_agent_analyze = agent_sub.add_parser("analyze", help="Run agent analysis on CSV")
    p_agent_analyze.add_argument("data", help="Path to CSV file")
    p_agent_analyze.add_argument("--query", "-q", default="", help="Causal question")
    p_agent_analyze.add_argument("--output", "-o", help="Output JSON file path")
    p_agent_analyze.add_argument(
        "--provider", default="openai", help="LLM provider: openai, openrouter, ollama, lmstudio"
    )
    p_agent_analyze.add_argument("--model", "-m", default=None, help="Override model name")
    p_agent_analyze.add_argument("--api-key", default=None, help="API key (or set env var)")
    p_agent_analyze.add_argument("--timeout", "-t", type=int, default=300, help="Timeout (s)")
    p_agent_analyze.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    # quickstart
    p_quick = sub.add_parser("quickstart", help="Run demo analysis on synthetic data")
    p_quick.add_argument("--output", "-o", help="Output JSON file path")

    # mcp
    sub.add_parser("mcp", help="Start MCP server (causal discovery as an agent skill)")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "version":
        cmd_version(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "agent":
        if getattr(args, "agent_command", None) == "analyze":
            cmd_agent_analyze(args)
        else:
            p_agent.print_help()
            sys.exit(1)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "quickstart":
        cmd_quickstart(args)
    elif args.command == "mcp":
        try:
            from causal_copilot.mcp.server import mcp as mcp_server
        except ImportError:
            print("Error: MCP server requires: pip install causal-copilot[mcp]", file=sys.stderr)
            sys.exit(1)
        mcp_server.run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
