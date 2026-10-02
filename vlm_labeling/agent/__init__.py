"""
vlm_labeling.agent — labelling by an AI agent (e.g. Claude Code over MCP) with SAM2 and a human review gate.

The agent looks at the image (with a coordinate grid, zooming where heads are small), places one point per
glandular head, SAM2 cuts the mask, a shape check trims stalks and merged neighbours, and the result is saved as
a *pending* label and/or pushed to Label Studio as a prediction. Only a person turns it into training data:
CTIP's export uses human-confirmed annotations unless predictions are explicitly requested.
"""

from vlm_labeling.agent.session import AgentLabelSession, Head, LabelRecord

__all__ = ["AgentLabelSession", "Head", "LabelRecord"]
