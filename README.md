# Financial Research Assistant

A simple app that answers questions about company annual reports (PDF files). You ask a question, the app finds the matching parts of the reports, and an AI writes the answer from them.

## What it does

- Reads PDF reports and saves them in a searchable database.
- Answers your questions using only what is in those reports.
- Shows the source file and page for each answer.
- Shows live forex rates (USD/INR, EUR/INR and more).
- Has a small chat for basic company info and currency conversion.

## How it works

1. **Load PDF:** the report is read page by page.
2. **Split:** pages are cut into small pieces (chunks).
3. **Embed:** each chunk is turned into numbers so it can be searched by meaning.
4. **Store:** the numbers are saved in ChromaDB.
5. **Answer:** the best matching chunks are sent to an AI model on Groq, which writes the report.

## Project structure

```
financial_research_project/
├── app.py                  # Starts the server (API + web page)
├── requirements.txt        # Python packages
├── .env                    # Your Groq key (GROQ_API_KEY=...)
├── Data/                   # PDF reports and the saved vector database      
├── pipeline/
│   ├── pipeline.py         # Runs the full flow: ingest, search, answer
│   ├── vector_store.py     # Embeddings and ChromaDB
│   └── agent_workflow.py   # Talks to the Groq AI model
└── static/
    └── index.html          # The web page (HTML, CSS, JavaScript)
```

## Tech used

- Python, FastAPI
- LangChain, ChromaDB
- sentence-transformers (`all-MiniLM-L6-v2`)
- Groq API
- HTML, CSS, JavaScript

##local run 
##http://127.0.0.1:8000/
<img width="958" height="428" alt="Screenshot 2026-11-04 185112" src="https://github.com/user-attachments/assets/fa479b1d-07b0-40de-82f4-928607b829ac" />

