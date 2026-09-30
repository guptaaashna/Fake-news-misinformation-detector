# TruthShield AI

TruthShield AI is a URL-only Streamlit MVP for explainable article verification. It extracts the readable article body, identifies candidate factual claims, gathers external context, verifies claims from explicit evidence relations, checks basic URL security indicators, calculates transparent prototype scores, stores completed analyses locally, and exports reports.

## Features

- URL-only article input
- General HTML article extraction with boilerplate filtering
- NLTK text processing and article statistics
- spaCy rule-based candidate claim extraction
- GDELT contextual news search
- Local six-class DistilBERT LoRA claim predictions
- Explainable Supported, Contradicted, and Insufficient Evidence results
- Basic HTTPS, DNS, SSL, hostname, and suspicious-pattern indicators
- Four bounded prototype indicators with component explanations
- SQLite analysis history, newest first
- HTML, CSV, and PDF report downloads
- Plotly, Matplotlib, and Seaborn visualizations
- Graceful handling of invalid URLs, blocked pages and missing evidence

## Tech Stack

- Python 3.11+, Streamlit, Requests, BeautifulSoup
- spaCy with `en_core_web_sm`, NLTK
- Matplotlib, Seaborn, Plotly, python-dotenv

## Architecture

```text
URL -> Article extraction -> Text processing -> Candidate claims
	-> GDELT contextual evidence -> Evidence-only claim verification
	-> Independent local LoRA claim prediction
	-> URL security indicators -> Prototype scores and explanations
	-> Streamlit dashboard, SQLite history, HTML/CSV/PDF exports
```

Modules communicate through ordinary Python functions. There are no separate services or REST servers.

## Project Structure

```text
truthshield-ai/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── services/       # extraction, NLP, evidence, verification, security, scoring, storage, exports
├── ui/             # Streamlit dashboard and components
├── data/           # local SQLite database
└── tests/
```

## Requirements

- Python 3.11 or newer
- Git
- Internet access for article and evidence requests

## Windows Setup in VS Code

Open the project folder in VS Code, then open **Terminal > New Terminal**. The commands below use PowerShell.

### 1. Create and activate a virtual environment

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in an Administrator PowerShell or use the Command Prompt alternative:

```bat
.venv\Scripts\activate.bat
```

In VS Code, choose the interpreter at `.venv\Scripts\python.exe` with **Python: Select Interpreter**.

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the spaCy English model:

```powershell
python -m spacy download en_core_web_sm
```

Download the NLTK resources used by text processing:

```powershell
python -c "import nltk; [nltk.download(resource) for resource in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4')]"
```

### 3. Create the local environment file

```powershell
Copy-Item .env.example .env
```

Keep `.env` private. It is ignored by Git.

### 4. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Stop the server with `Ctrl+C`.

## Testing

Run the complete suite:

```powershell
python -m pytest
```

Tests use mocked HTTP/API, DNS, SSL, and temporary SQLite databases. They cover extraction, text processing, claims, evidence, verification, security, scoring, persistence, pipeline failures, and report exports.

## Local Data and Exports

Completed analyses are stored in `data/truthshield.db`. The database is local and ignored by Git. Reports are generated in memory and downloaded through the dashboard as HTML, CSV, or PDF files.

## Limitations

- Simple HTTP extraction may not fully capture JavaScript-rendered pages.
- Claim extraction uses explainable heuristics and does not identify every factual claim.
- GDELT results are contextual and neutral; they do not prove a claim.
- Verification depends on available evidence and does not treat missing evidence as falsity.
- Security checks are basic indicators, not a malware or reputation verdict.
- Scores are heuristic prototype indicators, not scientifically validated probabilities.
- Network access and publisher page structure affect results.

## Local LoRA Classifier

The trained adapter in `models/distilbert-liar-lora/` uses the six LIAR labels in `training/setup_model.py`. On first prediction, the app downloads the repository's configured base checkpoint, [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), from Hugging Face and attaches the local adapter with PEFT. The adapter and tokenizer stay local; Hugging Face caches the base checkpoint for later runs.

Predictions are displayed separately from evidence verification. GDELT articles are contextual and neutral; neither GDELT results nor the model prediction alone determine the evidence verification status. No Google Fact Check API is used.

References:

- [DistilBERT documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
- [Hugging Face text classification guide](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)
