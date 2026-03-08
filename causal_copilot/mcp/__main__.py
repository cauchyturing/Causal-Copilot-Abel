"""Entry point: python -m causal_copilot.mcp"""

import sys

from causal_copilot.mcp.server import mcp

if "--http" in sys.argv:
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
else:
    mcp.run()
