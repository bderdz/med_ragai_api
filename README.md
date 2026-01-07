# Medical Diagnosis RAG Assistant

**A Retrieval-Augmented Generation (RAG) based medical diagnosis assistant
designed to assist in medical diagnosis. The assistant
helps to provide medical diagnosis based on patient symptoms, age and gender**
using a combination of a local vector database (ChromaDB), a local reranking model,
and reasoning LLM Gemini model via API.

Project is built with Fast API as main backend part and a simple chat wrapper around it
for project showcase, only for the needs of the subject IBM LLM-Basic.

## Overview

**This project created for 2 university subjects:**

1. **Medical information systems:** Main API implementation for medical diagnosis.
2. **IBM LLM-Basic:** A chat wrapper around the API with local LLM model for collecting patient information.

### Architecture

### Data source

**Disease dataset cleaned and preprocessed using Pandas, NumPy and thefuzz:**

- Disease symptoms formatted as percentage of total cases for each disease to improve matching accuracy.
- To match ICD10 codes to disease names, was used fuzzy matching with thefuzz library is used (after disease names was normalized).

### Vector Store

### Diagnosis Assistant model

### API

### Local Chat Agent

### Chat Wrapper

## Tech stack

- **Package Manager:** [uv](https://docs.astral.sh/uv/)
- **Backend:** FastAPI
- **Chat frontend:** Gradio
- **LLM Framework:** LangChain
- **Vector Store:** ChromaDB
- **LLM Models:**
    - **Main reasoning model:** Gemini via API
    - **Local chat agent:** Qwen2.5-0.5B-Instruct
    - **Reranking model:** MiniLM-L6-v2
- **Testing**: Pytest
- **Logging:** Python's built-in logging module
- **Data Handling:** Pandas, NumPy, Pydantic
- **Dataset processing**: thefuzz, Pandas, NumPy

## Running the project

1. **Clone the repository:**

```bash
git clone https://github.com/bderdz/med_ragai_api.git
cd med_ragai_api
```

2. **Setup project environment:**

use UV project manager (**Recommended**):

```bash
# Install UV if you don't have it yet
pip install uv
```

```bash
# In project root directory run:
uv sync
```

activate the environment:

```bash
source .venv/bin/activate
```

or run files with uv directly:

```bash
uv run <file_name>.py
```

## Tests

To run tests, use the following command at the project root directory:

```bash
# Default test run
pytest
# Run tests and save results to a txt file
pytest > logs/test_results.txt
```

*Tests logs will be saved in the `logs/pytest.log` file.*

1. API: `tests/test_api.py`
    - `test_api_works` - Tests if the API is reachable and returns a 200 status code.
    - `test_diagnose_endpoint` - Tests the `/diagnose` endpoint with sample symptoms and checks if the response contains a diagnosis.
    - `test_diagnose_validation` - Tests the input schema validation for the `/diagnose` endpoint.
    - `test_diagnose_prompt_injection` - Tests if the API is protected against prompt injection attacks.

## Sources

**Disease dataset:** [A Structured Bangla Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy](https://data.mendeley.com/datasets/rjgjh8hgrt/5)