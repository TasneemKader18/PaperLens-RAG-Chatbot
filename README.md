# PaperLens

AI-powered research paper assistant. Upload PDFs, ask questions in natural language, get answers from your papers.

## What It Does

PaperLens lets you have a conversation with your research papers. Upload PDFs, and the system:

1. Extracts text and splits it into chunks
2. Generates semantic embeddings for each chunk
3. Indexes them in a FAISS vector store
4. When you ask a question, retrieves the most relevant passages
5. Sends them as context to a local LLM that generates an answer citing the source papers

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask (Python) |
| PDF Parsing | PyMuPDF |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Search | FAISS (IndexFlatIP) |
| LLM | TinyLlama 1.1B via Ollama |
| Frontend | HTML, CSS, JavaScript |
| Data Persistence | JSON file storage |

## Project Structure


```
PaperLens/
├── app.py                 # Flask server and API routes
├── rag_engine.py          # RAG pipeline (extract, chunk, embed, retrieve, query)
├── data_store.py          # JSON-based persistence for logs and history
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── styles.css         # Stylesheet
└── data/                  # Auto-created at runtime
    ├── chat_history.json
    ├── upload_history.json
    └── events.json
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/PaperLens.git
cd PaperLens
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Pull the LLM model**

```bash
ollama pull tinyllama:1.1b
```

**4. Start Ollama** (if not already running)

```bash
ollama serve
```

**5. Run the app**

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

1. **Upload** - Drag and drop PDF research papers (or click to browse) in the sidebar
2. **Ask** - Type a question about your papers in the chat input
3. **Read** - Get an AI-generated answer with source citations

### Example Questions

- "Summarize the key findings"
- "What methodology was used?"
- "Compare the results across papers"

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serve the frontend |
| POST | `/upload` | Upload and index PDF files |
| POST | `/chat` | Ask a question (SSE streaming response) |
| GET | `/papers` | List indexed papers |
| POST | `/clear` | Clear the knowledge base |

## Configuration

Key parameters in `rag_engine.py`:

```python
OLLAMA_MODEL = "tinyllama:1.1b"   # LLM model
CHUNK_SIZE = 200                   # Words per chunk
CHUNK_OVERLAP = 50                 # Overlap between chunks
TOP_K = 2                          # Number of chunks retrieved
MAX_CONTEXT_CHARS = 1500           # Context size limit for the prompt
```

## Limitations

- **CPU-only** - Designed for laptops without GPU. TinyLlama 1.1B trades answer quality for accessibility.
- **No OCR** - Scanned/image-only PDFs won't extract any text.
- **Single user** - Shared state, no authentication or session isolation.
- **English-focused** - Both the embedding model and LLM perform best on English text.

