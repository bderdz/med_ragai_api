import pytest, json
from src.llm.dispatcher import tool_dispatcher, ToolNotFoundError, ToolTimeoutError


async def mock_tool(name: str) -> dict[str, str]:
    return {"hello": name}


async def slow_tool(name: str) -> dict[str, str]:
    import asyncio
    await asyncio.sleep(2)
    return {"hello": name}


allowed_tools = {
    "greet": mock_tool,
    "slow_greet": slow_tool
}


@pytest.mark.asyncio
async def test_allowed_tool():
    """
    Test case: Allowed tool is called with valid arguments
    """
    tool_name = "greet"
    tool_args = {"name": "Alice"}

    result = await tool_dispatcher(tool_name, tool_args, allowed_tools)
    assert result == json.dumps({"hello": 'Alice'})


@pytest.mark.asyncio
async def test_not_allowed_tool():
    """
    Test case: Not allowed tool is requested
    """
    tool_name = "unknown_tool"
    tool_args = {}
    with pytest.raises(ToolNotFoundError):
        await tool_dispatcher(tool_name, tool_args, allowed_tools)


@pytest.mark.asyncio
async def test_tool_timeout():
    """
    Test case: Tool execution exceeds timeout
    """
    tool_name = "slow_greet"
    tool_args = {"name": "Bob"}

    with pytest.raises(ToolTimeoutError):
        await tool_dispatcher(tool_name, tool_args, allowed_tools, timeout=1.0)
