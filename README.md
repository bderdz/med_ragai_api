# Medical Diagnosis RAG Assistant

This project created for 2 university subjects:

- Medical information systems
- IBM LLM-Basic

## Overview

## Project dived into 2 main parts:

1. API
2. Chat wrapper around the API with local llm model

## Tests

To run tests, use the following command at the project root directory:

```bash
pytest
```

*Tests logs will be saved in the `logs/pytest.log` file.*

1. API: `tests/test_api.py`
    - `test_api_works` - Tests if the API is reachable and returns a 200 status code.
    - `test_diagnose_endpoint` - Tests the `/diagnose` endpoint with sample symptoms and checks if the response contains a diagnosis.
    - `test_diagnose_validation` - Test the input schema validation for the `/diagnose` endpoint.
    - `test_diagnose_prompt_injection` - Test if the API is protected against prompt injection attacks.

## Sources

**Disease dataset:** [A Structured Bangla Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy](https://data.mendeley.com/datasets/rjgjh8hgrt/5)