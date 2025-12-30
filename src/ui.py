from typing import Any

import gradio as gr
from src.llm import LocalChatAgent

EXAMPLES = [
    "Hello",
    "Hi, i'm a 23 years old man. I have headache, cough and temperature.",
    "Hi, I'm not feeling well. I have a persistent cough and sore throat."]


class ChatAgentUI:
    def __init__(self, local_model: str):
        self.chat_agent = LocalChatAgent(local_model)

        with gr.Blocks(title="Medical Diagnosis Assistant API") as self.ui:
            gr.HTML("<h1 style='text-align: center;'>👨‍⚕️ Medical Diagnosis Assistant API</h1>")
            with gr.Row():
                with gr.Column(scale=2, min_width=300):
                    chatbot = gr.Chatbot(min_height=550)
                    self.interface = gr.ChatInterface(
                        fn=self.respond,
                        examples=EXAMPLES,
                        chatbot=chatbot
                    )
                    self.interface.chatbot.clear(fn=self.clear_history)
                with gr.Column(scale=1):
                    gr.Markdown("**Welcome to the Medical Diagnosis Assistant**\n\n"
                                "**This AI-powered medical assistant helps you identify possible diseases based on your symptoms.**")
                    gr.Markdown("## ❔ How to use:")
                    gr.Markdown("### 💬 Chat:")
                    gr.Markdown(
                        "You can start a conversation with the AI medical assistant by typing your messages in the chat interface on the left."
                        "**The assistant will guide you through a series of questions to gather necessary information for diagnosis:**\n\n"
                        "1. Age\n\n2. Gender\n\n3. List of symptoms\n\n")
                    gr.Markdown("### 📡 API:")
                    gr.Markdown(
                        "The chat agent is built on top of the Diagnosis Assistant API. You can access the API documentation at **/docs endpoint** "
                        "[http://localhost:8000/docs](http://localhost:8000/docs) once the server is running.")
                    gr.Markdown("### ⚠️ Info:")
                    gr.Markdown(
                        "**Bigger LLM like gemini-2.5-flash may take longer time to respond than lite version..️**\n\n"
                        "**HTTP Request timeout is set to 120 seconds.**")

    def clear_history(self) -> None:
        """Clear chat agent history"""
        self.chat_agent.reset_history()
        print("INFO: Chat history cleared.")

    async def respond(self, message: str, history: Any) -> str:
        if not message:
            return ""

        answer = await self.chat_agent.chat(message)
        return answer
