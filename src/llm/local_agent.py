import json, re, logging, time
from json import JSONDecodeError
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from src.llm.guardrails import run_guardrails, SecurityError
from src.llm.tools import get_diagnosis_tool
from src.llm.dispatcher import tool_dispatcher, ToolError, ToolValidationError, ToolNotFoundError

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics")

# Local model config ( Pipeline kwargs )
config = {
    "temperature": 0.1,
    "max_new_tokens": 256,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "return_full_text": False,
}

# Allowed tools map
ALLOWED_TOOLS = {
    "get_diagnosis_tool": get_diagnosis_tool
}

SYSTEM = """You are a strict Data Collection Agent for a medical screening system.
You are NOT a doctor. You have NO personality. You do NOT give advice. You do NOT answer any other question.
DO NOT generate tool output by yourself. You MUST use the tool call to get the data.

### OBJECTIVE
Your ONLY goal is to collect exactly 3 variables from the user:
1. 'age' (integer)
2. 'gender' (must be 'male' or 'female')
3. 'symptoms' (list of strings)

### PROTOCOL (Follow strictly in order)
1. **GREETING:** If the conversation starts, output the STANDARD_GREETING (see below).
2. **ANALYSIS:** specific check if you have 'age', 'gender', and 'symptoms' from previous messages.
3. **MISSING INFO:** If any variable is missing, ask for it concisely.
   - ASK ONLY ONE QUESTION AT A TIME.
4. **COMPLETION:** ONLY IF you have all 3 variables, output the RAW JSON TOOL CALL.

### STANDARD_GREETING
"Hello. I am an AI medical assistant. To help you, I need to collect some basic information. First, how old are you?"

### JSON TOOL CALL FORMAT
When you have all info, output ONLY this JSON structure (no markdown, no text before/after):
{"tool": "get_diagnosis_tool", "args": {"age": 25, "gender": "male", "symptoms": ["fever"]}}

### EXAMPLES

User: Hi
Assistant: Hello. I am an AI medical assistant. To help you, I need to collect some basic information. First, how old are you?

User: I am 45.
Assistant: Are you male or female?

User: Male.
Assistant: Please list your symptoms.

User: I have a headache and high temperature.
Assistant: {"tool": "get_diagnosis_tool", "args": {"age": 45, "gender": "male", "symptoms": ["headache", "high temperature"]}}

User: Help, I feel sick.
Assistant: Hello. I am an AI medical assistant. To help you, I need to collect some basic information. First, how old are you?

User: I am 21.
Assistant: Are you male or female?

User: I'm a woman.
Assistant: Please list your symptoms.

User: I have a temperature, cough and headache.
Assistant: {"tool": "get_diagnosis_tool", "args": {"age": 21, "gender": "female", "symptoms": ["cough", "headache", "temperature"]}}
"""

output_formatting = """
# IMPORTANT
The tool has returned the medical data above. 
Interpret the JSON and strictly output the result in provided **styled text** format:
## Possible Diseases:
1. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
2. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
...
If the list is empty, just say: "No relevant data found."
Do NOT add any other text. DO NOT return JSON.
"""


def parse_tool_call(response: str) -> dict[str, Any]:
    """
    Parse tool call json from model response text.
    Returns JSON dictionary
    """
    logger.debug(f"Parsing tool call from response: {response}")
    response = response.replace("```", "").replace("json", "")
    parsed_call = re.search(r"\s*({.*}?)\s*", response, re.DOTALL)
    if parsed_call:
        try:
            tool_call = json.loads(parsed_call.group())
            if "tool" not in tool_call:
                raise JSONDecodeError("Bad tool call format.")
            if "args" not in tool_call:
                raise JSONDecodeError("No args provided in tool call.")
            return tool_call
        except JSONDecodeError as e:
            logger.error(f"TOOL PARSER JSON ERROR: {e}")
            raise
    return {}


class LocalChatAgent:
    """
    A local chat agent with tools.
    Main task is collect information about patient.
    """

    def __init__(self, model_name: str):
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs=config
        )
        self.agent = ChatHuggingFace(
            llm=self.model,
            verbose=True
        )
        self.history: list[BaseMessage] = [SystemMessage(content=SYSTEM)]
        logger.info(f"LocalChatAgent initialized with model: {model_name}")

    def reset_history(self):
        logger.info("Chat Agent history reset.")
        self.history = [SystemMessage(content=SYSTEM)]

    async def chat(self, prompt: str) -> str:
        start_time = time.time()
        # Guardrails check
        try:
            run_guardrails(prompt)
        except SecurityError as e:
            return f"SECURITY ERROR: {e}"
        # Prompt processing
        user_msg = HumanMessage(content=prompt)
        self.history.append(user_msg)

        response = self.agent.invoke(self.history)
        logger.debug(f"Local model RAW response: {response.content}")
        self.history.append(response)

        # Metrics template
        log_data = {
            "model": self.model.model_id,
            "tool": False,
            "latency": {
                "total_s": 0.0,
            },
        }

        # Tool call processing
        try:
            tool_call = parse_tool_call(response.content)

            if tool_call:
                tool_output_msg = ""
                # Tool dispatching
                try:
                    tool_output = await tool_dispatcher(
                        tool_call["tool"],
                        tool_call["args"],
                        ALLOWED_TOOLS,
                        timeout=60.0)
                    tool_output_msg = f"{tool_output}\n\n{output_formatting}"

                except ToolValidationError as e:
                    tool_output_msg = f"ERROR: The data provided is invalid: {e}. Please ask the user for correct information."
                except ToolNotFoundError as e:
                    tool_output_msg = f"ERROR: The requested tool was not found: {e}. DO NOT HALLUCINATE tools."
                except ToolError as e:
                    tool_output_msg = f"ERROR: There was an error executing the tool: {e}. Please try again."
                except Exception as e:
                    tool_output_msg = f"ERROR: Unexpected error during tool execution: {e}. Please try again."

                # Add tool output to history
                print(tool_output_msg)
                self.history.append(HumanMessage(tool_output_msg))

                # Agent response with tool output
                final_response = self.agent.invoke(self.history)
                self.history.append(final_response)

                # Metrics logging
                total_s = round(time.time() - start_time, 4)
                log_data["tool"] = True
                log_data["latency"]["total_s"] = total_s

                metrics_logger.info(f"LOCAL MODEL METRICS: {json.dumps(log_data)}")
                return final_response.content

        except JSONDecodeError as e:
            self.history.append(HumanMessage(f"ERROR: Failed to parse tool call JSON: {e}. Please try again."))

            # Metrics logging
            total_s = round(time.time() - start_time, 4)
            log_data["latency"]["total_s"] = total_s
            metrics_logger.info(f"LOCAL MODEL METRICS: {json.dumps(log_data)}")

            return response.content

        # Metrics logging
        total_s = round(time.time() - start_time, 4)
        log_data["latency"]["total_s"] = total_s
        metrics_logger.info(f"LOCAL MODEL METRICS: {json.dumps(log_data)}")

        # No tool call detected, return original response
        return response.content
