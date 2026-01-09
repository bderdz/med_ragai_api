import logging, os, time, json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from src.schemas import DiagnoseResponse, SymptomsInput
from src.llm.guardrails import run_guardrails, SecurityError

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("metrics")

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RERANK_MODEL = os.getenv("RERANK_MODEL", 'cross-encoder/ms-marco-MiniLM-L-6-v2')
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
        self.llm = model.with_structured_output(DiagnoseResponse, include_raw=True)
        self.retriever = vectors_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})
        self.cross_encoder = CrossEncoder(RERANK_MODEL)
        logger.info(f"DiagnosisAssistant initialized with model: {GEMINI_MODEL}")

    def diagnose(self, patient_info: SymptomsInput) -> DiagnoseResponse:
        """
        Diagnose patient based on symptoms using RAG based gemini model.
        """
        start_time = time.time()

        # Guardrails check
        run_guardrails(patient_info.__str__())

        symptoms = ", ".join(patient_info.symptoms)

        # Retrieval
        start_retrieval = time.time()
        docs = self.retriever.invoke(symptoms)
        retrieval_time = time.time() - start_retrieval

        # Reranking
        start_rerank = time.time()
        pairs = [[symptoms, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        # Choose only Top 6 docs
        top_k_docs = [doc for _, doc in scored_docs[:6]]

        rerank_time = time.time() - start_rerank
        total_retrieval_time = retrieval_time + rerank_time

        context = "\n\n".join([doc.page_content for doc in top_k_docs])
        logger.debug(f"Symptoms: {symptoms}\nRetrieved context:\n{context}")

        # LLM
        prompt = prompt_template.invoke({
            "gender": patient_info.gender,
            "age": patient_info.age,
            "symptoms": symptoms,
            "context": context
        })
        start_llm = time.time()
        response = self.llm.invoke(prompt)
        llm_time = time.time() - start_llm

        # Metrics
        total_time = time.time() - start_time
        token_usage = response["raw"].usage_metadata
        latency = {
            "retrieval_s": round(retrieval_time, 4),
            "rerank_s": round(rerank_time, 4),
            "total_retrieval_s": round(total_retrieval_time, 4),
            "llm_s": round(llm_time, 4),
            "total_s": round(total_time, 4)
        }
        log_data = {
            "model": GEMINI_MODEL,
            "context_docs_count": len(docs),
            "latency": latency,
            "token_usage": token_usage
        }

        metrics_logger.info(f"DIAGNOSE METRICS: {json.dumps(log_data)}")
        return response["parsed"]
