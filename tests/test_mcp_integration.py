"""Integration test: verify MCP server starts and 4 tools are registered."""

import asyncio

import pytest


class TestServerRegistration:
    def test_all_tools_registered(self):
        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "discover",
            "inspect_graph",
            "diagnose_data",
            "run_algorithm",
        }
        for t in expected:
            assert t in tool_names, f"Tool '{t}' not registered"

    def test_no_deleted_tools(self):
        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        deleted = {
            "analyze",
            "explain_graph",
            "explain_result",
            "estimate_effects",
            "refine_graph",
            "list_algorithms",
        }
        for t in deleted:
            assert t not in tool_names, f"Deleted tool '{t}' still registered"

    def test_prompts_registered(self):
        from causal_copilot.mcp.server import mcp

        prompts = asyncio.run(mcp.list_prompts())
        prompt_names = {p.name for p in prompts}
        assert "causal_expert" in prompt_names
        assert "analyze_dataset" in prompt_names

    def test_tool_count(self):
        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        assert len(tools) == 12

    def test_mcp_cli_entry(self):
        from causal_copilot.cli import main

        with pytest.raises(SystemExit):
            main(["mcp", "--help"])
