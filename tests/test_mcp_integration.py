"""Integration test: verify MCP server starts and tools are registered."""
import asyncio

import pytest


class TestServerRegistration:
    def test_all_tools_registered(self):
        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "analyze", "diagnose_data", "run_algorithm", "discover",
            "refine_graph", "estimate_effects", "explain_graph",
            "explain_result", "list_algorithms",
        }
        for t in expected:
            assert t in tool_names, f"Tool '{t}' not registered"

    def test_prompts_registered(self):
        from causal_copilot.mcp.server import mcp

        prompts = asyncio.run(mcp.list_prompts())
        prompt_names = {p.name for p in prompts}
        assert "causal_expert" in prompt_names
        assert "analyze_dataset" in prompt_names

    def test_tool_count(self):
        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        # At least 9 tools expected
        assert len(tools) >= 9

    def test_mcp_cli_entry(self):
        from causal_copilot.cli import main

        with pytest.raises(SystemExit):
            main(["mcp", "--help"])
