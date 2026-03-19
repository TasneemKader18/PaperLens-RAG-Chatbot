import logging
import time
import json
import fitz
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama:1.1b" 

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 2

embedder = SentenceTransformer("all-MiniLM-L6-v2")


class RAGEngine:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.paper_names = []

    def extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()  # 🔥 important for Windows
        return text

    def chunk_text(self, text: str, source: str):
        words = text.split()
        result = []

        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if chunk.strip():
                result.append({
                    "text": chunk,
                    "source": source
                })

        return result

    def add_paper(self, pdf_path: str, filename: str) -> bool:
        if filename in self.paper_names:
            return False

        text = self.extract_text(pdf_path)
        if not text.strip():
            return False

        new_chunks = self.chunk_text(text, filename)
        texts = [c["text"] for c in new_chunks]

        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True
        ).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.chunks.extend(new_chunks)
        self.index.add(embeddings)
        self.paper_names.append(filename)

        return True

    def clear(self):
        self.index = None
        self.chunks = []
        self.paper_names = []

    def retrieve(self, query: str):
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            q_emb,
            min(TOP_K, self.index.ntotal)
        )

        results = []
        for j, i in enumerate(indices[0]):
            if i < len(self.chunks):
                results.append({
                    **self.chunks[i],
                    "score": float(scores[0][j])
                })

        return results

    def query(self, question: str):
        log.info("Query called: %s", question[:200])

        if not self.paper_names:
            return {
                "answer": "Please upload at least one research paper first.",
                "sources": []
            }

        context_chunks = self.retrieve(question)
        log.info("Retrieved %d context chunks", len(context_chunks))

        # 🔥 limit context size (important for small models)
        MAX_CONTEXT_CHARS = 1500
        context = ""

        for c in context_chunks:
            chunk_text = f"\n\n[From: {c['source']}]\n{c['text']}"
            if len(context) + len(chunk_text) > MAX_CONTEXT_CHARS:
                break
            context += chunk_text

        prompt = f"""You are a research paper assistant. Answer the question using ONLY the provided context from research papers. If the context doesn't contain enough information, say so. Be precise and cite which paper the information comes from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        log.info("Prompt size: %d chars", len(prompt))
        log.info("Sending request to Ollama at %s (model=%s)", OLLAMA_URL, OLLAMA_MODEL)

        t0 = time.time()

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.0},
                },
                stream=True,
                timeout=60,
            )

            elapsed = time.time() - t0
            log.info("Ollama response status: %d (took %.1fs)", resp.status_code, elapsed)

            resp.raise_for_status()

            answer = ""

            for line in resp.iter_lines():
                if not line:
                    continue

                try:
                    decoded = line.decode("utf-8")
                    data = json.loads(decoded)

                    token = data.get("response")
                    if token:
                        answer += token

                    if data.get("done"):
                        break

                except Exception as e:
                    log.error("Stream parse error: %s | line=%s", e, line)

            # 🔥 fallback if streaming failed
            if not answer.strip():
                log.warning("Empty response, falling back to non-streaming")

                fallback = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=60,
                )

                fallback.raise_for_status()
                answer = fallback.json().get("response", "")

            log.info("Ollama answered (%d chars)", len(answer))

        except requests.exceptions.ConnectionError as e:
            log.error("Cannot connect to Ollama: %s", e)
            answer = "Error: Cannot connect to Ollama."

        except requests.exceptions.Timeout:
            log.error("Ollama request timed out")
            answer = "Error: Ollama timeout."

        except Exception as e:
            log.exception("Ollama request failed: %s", e)
            answer = f"Error: {e}"

        sources = sorted(set(c["source"] for c in context_chunks))

        return {
            "answer": answer,
            "sources": sources
        }