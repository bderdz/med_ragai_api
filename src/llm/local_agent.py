import json, os, re
from json import JSONDecodeError
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.llm.tools import get_diagnosis_tool

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
{"tool": "get_diagnosis_tool", "arguments": {"age": 25, "gender": "male", "symptoms": ["fever"]}}

### RESPOND FORMAT
The tool returns the medical data,
Interpret the JSON and strictly output the result in this format:
## Possible Diseases:
1. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
2. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
...
If the list is empty, just say: "No relevant data found."
Do NOT add any other text.

### EXAMPLES

User: Hi
Assistant: Hello. I am an AI medical assistant. To help you, I need to collect some basic information. First, how old are you?

User: I am 45.
Assistant: Are you male or female?

User: Male.
Assistant: Please list your symptoms.

User: I have a headache and high temperature.
Assistant: {"tool": "get_diagnosis_tool", "arguments": {"age": 45, "gender": "male", "symptoms": ["headache", "high temperature"]}}

User: Help, I feel sick.
Assistant: Hello. I am an AI medical assistant. To help you, I need to collect some basic information. First, how old are you?

User: I am 21.
Assistant: Are you male or female?

User: I'm a woman.
Assistant: Please list your symptoms.

User: I have a temperature, cough and headache.
Assistant: {"tool": "get_diagnosis_tool", "arguments": {"age": 21, "gender": "female", "symptoms": ["cough", "headache", "temperature"]}}
"""

output_formatting = """
The tool has returned the medical data above. 
Interpret the JSON and strictly output the result in this **styled text** format:
## Possible Diseases:
1. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
2. **Disease Name:** {name} 
**ICD Code:** {icd_code}
**Reasoning:** {reasoning}
...
If the list is empty, just say: "No relevant data found."
Do not add any other text.
"""


def parse_tool_calls(response: str) -> dict[str, Any]:
    """
    Parse tool call json from model response text.
    Returns JSON dictionary
    """
    response = response.replace("```", "").replace("json", "")
    print("DEBUG: ", response)
    parsed_call = re.search(r"\s*({.*}?)\s*", response, re.DOTALL)
    if parsed_call:
        try:
            tool_call = json.loads(parsed_call.group())
            if "tool" not in tool_call:
                raise JSONDecodeError("Bad tool call format.")
            return tool_call
        except JSONDecodeError as e:
            print(f"ERROR: JSON parser ERROR: {e}")
            return {"error": f"Bad tool call format, JSON parsing failed. Please try again."}
    return {}


async def call_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """
    Tool call by tool name with provided arguments as a parametr.
    Tool must be in the 'allowed tools'
    """
    tool_fn = ALLOWED_TOOLS[tool_name]
    try:
        result = await tool_fn(**tool_args)
    except Exception as e:
        result = {"error": f"Tool execution error: {e}."}

    return f"TOOL '{tool_name}' OUTPUT: {json.dumps(result)}"


class LocalChatAgent:
    """
    A local chat agent with tools.
    Main task is collect information about patient.
    """

    def __init__(self, model: str):
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model,
            task="text-generation",
            pipeline_kwargs=config
        )
        self.agent = ChatHuggingFace(
            llm=self.model,
            verbose=True
        )
        self.history: list[BaseMessage] = [SystemMessage(content=SYSTEM)]
        print("INFO: LocalChatAgent initialized with model:", model)

    def reset_history(self):
        self.history = [SystemMessage(content=SYSTEM)]

    async def chat(self, prompt: str) -> str:
        user_msg = HumanMessage(content=prompt)
        self.history.append(user_msg)

        response = self.agent.invoke(self.history)
        print("RAW RESPONSE:", response.content)
        self.history.append(response)
        # Tool call processing
        tool_call = parse_tool_calls(response.content)
        if tool_call:
            if "error" in tool_call:
                self.history.append(HumanMessage(tool_call["error"]))
            else:
                tool_result = await call_tool(tool_call["tool"], tool_call["arguments"])
                self.history.append(HumanMessage(tool_result + "\n\n" + output_formatting))

            final_response = self.agent.invoke(self.history)
            self.history.append(final_response)
            return final_response.content

        return response.content
