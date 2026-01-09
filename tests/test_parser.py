import json
from src.llm.local_agent import parse_tool_call


def test_parse_no_tool():
    """
    Test case 1: No tool call, simple response
    """
    response = "This is a simple response without any tool call."
    parsed_tool = parse_tool_call(response)
    assert not parsed_tool


def test_parse_markdown_json():
    """
    Test case 2: Tool call in markdown JSON format
    """
    tool_call = {
        "tool": "test_tool",
        "args": {"param1": "value1"}
    }

    response = f'```json\n{json.dumps(tool_call)}\n```'
    parsed_tool = parse_tool_call(response)
    assert parsed_tool == tool_call


def test_parse_clean_json():
    """
    Test case 3: Tool call in clean JSON format
    """
    tool_call = {
        "tool": "test_tool",
        "args": {"param1": "value1"}
    }

    response = json.dumps(tool_call)
    parsed_tool = parse_tool_call(response)
    assert parsed_tool == tool_call


def test_parse_json_with_text():
    """
    Test case 4: Tool call with surrounding text
    """
    tool_call = {
        "tool": "test_tool",
        "args": {"param1": "value1"}
    }

    response = f"Here is the tool call:\n```json\n{json.dumps(tool_call)}\n```"
    parsed_tool = parse_tool_call(response)
    assert parsed_tool == tool_call
