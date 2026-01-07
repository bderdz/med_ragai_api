import logging, asyncio, json, time
from typing import Any, Callable

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics")


class ToolError(Exception):
    """Tool related errors base class"""
    pass


class ToolTimeoutError(ToolError):
    """Raised when a tool execution times out"""
    pass


class ToolValidationError(ToolError):
    """Raised when a tool's arguments fail validation"""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found"""
    pass


def sanitize_tool_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    sanitized_args = {}

    for key, value in tool_args.items():
        if isinstance(value, str):
            value = value.strip()

            if len(value) > 1000:
                raise ToolValidationError(f"Argument '{key}' exceeds maximum length.")

            if key == "gender":
                value = value.lower()
        elif isinstance(value, list):
            if key == "symptoms":
                if not all(isinstance(item, str) for item in value):
                    raise ToolValidationError(f"All items in '{key}' must be strings.")

                value = [item.strip() for item in value]

        sanitized_args[key] = value

    return sanitized_args


async def tool_dispatcher(
        tool_name: str,
        tool_args: dict[str, Any],
        allowed_tools: dict[str, Callable],
        timeout: float = 60.0) -> str:
    """
    Dispatches safe tool calls based on tool name and arguments.
    Checks if the tool is allowed, sanitizes arguments and enforces a timeout.
    Returns the tool result as a JSON string.
    """
    start_time = time.time()
    tool_status = "success"
    logger.info(f"DISPATCHER: Processing tool '{tool_name}' with args: {tool_args}")

    if tool_name not in allowed_tools:
        logger.error(f"SECURITY: Not allowed tool '{tool_name}' requested.")
        raise ToolNotFoundError(f"Tool '{tool_name}' is not available.")

    tool_fn = allowed_tools[tool_name]
    try:
        sanitized_args = sanitize_tool_args(tool_args)
        result = await asyncio.wait_for(tool_fn(**sanitized_args), timeout=timeout)
        return json.dumps(result)
    # Error handling
    except asyncio.TimeoutError:
        logger.error(f"TIMEOUT: Tool '{tool_name}' timed out after {timeout}s")
        tool_status = "timeout"
        raise ToolTimeoutError(f"Tool execution timeout after {timeout}s")
    except ToolValidationError as e:
        logger.error(f"TOOL VALIDATION ERROR: {e}")
        tool_status = "error"
        raise
    except ToolError as e:
        logger.error(f"TOOL ERROR: {e}")
        tool_status = "error"
        raise
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR in tool '{tool_name}': {e}")
        tool_status = "error"
        raise ToolError(f"An unexpected error occurred: {e}")
    finally:
        # Metrics logging
        total_time = time.time() - start_time
        latency = {
            "total_s": round(total_time, 4),
        }
        log_data = {
            "tool": tool_name,
            "status": tool_status,
            "latency": latency,
        }

        metrics_logger.info(f"TOOL EXECUTION: {json.dumps(log_data)}")
