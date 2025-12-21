import gradio as gr
import httpx
from src.schemas import SymptomsInput

info = """
<div style="text-align: center;font-size:18px;font-weight:bold;">
    <p>⚠️ Bigger models like gemini-2.5-flash may take longer time to respond than lite version. ⚠️</p>
    <p>HTTP Request timeout is set to 120 seconds.</p>
</div>
"""


async def diagnose_request(gender: str, age: int, symptoms: str):
    # Input validation
    if age <= 0:
        return "Please enter a valid age"
    if not gender:
        return "Please select a gender"
    if not symptoms:
        return "Please enter at least one symptom"

    symptoms_list = [symptom.strip() for symptom in symptoms.split(",")]
    body = SymptomsInput(age=age, gender=gender, symptoms=symptoms_list)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post("http://localhost:8000/diagnose", json=body.model_dump())
            response.raise_for_status()
            data = response.json()
            output = "Possible Diseases:\n\n"
            for disease in data["possible_diseases"]:
                output += f"- {disease['name']} (ICD Code: {disease['icd_code']})\nREASONING:\n{disease['reasoning']}\n\n"

            return output
        except httpx.HTTPStatusError as e:
            return f"API ERROR {e.response.status_code}: {e.response.text}"
        except httpx.RequestError as e:
            return f"REQUEST ERROR: {e}"


def load_ui() -> gr.Interface:
    api_ui = gr.Interface(
        fn=diagnose_request,
        inputs=[
            gr.Radio(["male", "female"], label="Gender", value="male"),
            gr.Number(label="Age", value=20),
            gr.Textbox(label="Symptoms", placeholder="symptom1, symptom2, ...", lines=3)
        ],
        outputs=[gr.Textbox(label="Diagnosis Result", lines=20)],
        article=info,
        title="Medical Diagnosis Assistant",
        description="Enter patient information to get possible disease diagnoses based on symptoms.",
        flagging_mode="never"
    )

    return api_ui
