import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from src.schemas import DiagnoseResponse, SymptomsInput

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
SYSTEM = """
## ROLE
You are highly capable medical assistant. Your task is to analyze patient symptoms, 
age and gender and suggest the most possible diseases they might have from the provided context.

## INPUT
You will receive:
- Patient's age and gender
- List of symptoms reported by the patient
- Context - a list of probable disease with their symptoms and probabilities of appearance.

## RULES
1. **Context ONLY**: You must select diseases ONLY from the provided Context. Do not add new diseases.
2. **Common disease priority** Always prioritize common diseases over rare or eradicated diseases unless the specific distinct symptoms perfectly match the rare case.
3. **Internal Knowledge for Probability**: 
   - While you must strictly stick to the Context for disease names and symptoms, **you MUST use your internal medical knowledge to evaluate the prevalence and probability** of the disease for the given age/gender.
   - If a disease in the Context is extremely rare (e.g., exotic viruses) or eradicated, and the symptoms are generic (like headache), **rank it lower** or discard it, unless the match is specific and unique.
4. **Typo correction** Use your internal knowledge to interpret user typos ("caught" as "cough").
5. **Selection** Select TOP 3 the most probable disease based on symptoms matching and statistically probable for the patient.
6. If all Context disease are mismatched or there is no given Context respond with "No relevant data found."
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", """
    PATIENT INFO:
    Gender: {gender}
    Age: {age}
    Symptoms: {symptoms}
    CONTEXT:
    Best matching diseases from knowledge base:
    {context}
    """)
])


class DiagnosisAssistant:
    """
    RAG diagnosis assistant based on GEMINI model.
    """

    def __init__(self, vectors_store: Chroma):
        model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2
        )
        self.llm = model.with_structured_output(DiagnoseResponse)
        self.retriever = vectors_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        logger.info(f"DiagnosisAssistant initialized with model: {GEMINI_MODEL}")

    def diagnose(self, patient_info: SymptomsInput) -> DiagnoseResponse:
        """

        """
        symptoms = ", ".join(patient_info.symptoms)
        docs = self.retriever.invoke(symptoms)
        context = "\n\n".join([doc.page_content for doc in docs])
        # DEBUG
        logger.debug(f"Symptoms: {symptoms}\nRetrieved context:\n{context}")

        prompt = prompt_template.invoke({
            "gender": patient_info.gender,
            "age": patient_info.age,
            "symptoms": symptoms,
            "context": context
        })
        response = self.llm.invoke(prompt)
        return response
