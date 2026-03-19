import os
import json
import logging
import time
import tempfile
import threading
from flask import Flask, render_template, request, jsonify, Response
from rag_engine import RAGEngine
import data_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.INFO)

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max

engine = RAGEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    log.info("Upload request received: %d file(s)", len(files))
    added, skipped = [], []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            log.warning("Skipped non-PDF file: %s", f.filename)
            skipped.append(f.filename)
            continue

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        f.save(tmp.name)
        try:
            if engine.add_paper(tmp.name, f.filename):
                log.info("Indexed paper: %s", f.filename)
                added.append(f.filename)
            else:
                log.warning("Skipped (duplicate or empty): %s", f.filename)
                skipped.append(f.filename)
        except Exception as e:
            log.exception("Error indexing paper %s: %s", f.filename, e)
            skipped.append(f.filename)
        finally:
            os.unlink(tmp.name)

    log.info("Upload complete: added=%s, skipped=%s, total_chunks=%d",
             added, skipped, len(engine.chunks))
    data_store.log_upload(added, skipped, len(engine.chunks))
    return jsonify(
        added=added,
        skipped=skipped,
        papers=engine.paper_names,
        total_chunks=len(engine.chunks),
    )


@app.route("/chat", methods=["POST"])
def chat():
    log.info("Chat request received")

    try:
        body = request.get_json(force=True)
    except Exception as e:
        log.error("Failed to parse JSON body: %s", e)
        return jsonify(answer="Error: Invalid request body.", sources=[]), 400

    if not body:
        log.error("Empty or null JSON body")
        return jsonify(answer="Error: Empty request body.", sources=[]), 400

    question = body.get("question", "")
    log.info("Question: %s", question[:200])

    def generate():
        result = {"done": False, "data": None}
        steps = []
        t_start = time.time()

        def send_step(step_text):
            steps.append(step_text)

        def do_work():
            try:
                send_step("Searching knowledge base...")
                query_result = engine.query(question)
                log.info("Answer generated (%d chars), sources: %s",
                         len(query_result.get("answer", "")), query_result.get("sources", []))
                result["data"] = json.dumps(query_result)
                result["parsed"] = query_result
                result["done"] = True
            except Exception as e:
                log.exception("Error during query: %s", e)
                err_result = {"answer": f"Backend error: {e}", "sources": []}
                result["data"] = json.dumps(err_result)
                result["parsed"] = err_result
                result["done"] = True

        worker = threading.Thread(target=do_work)
        worker.start()

        last_step_index = 0
        while not result["done"]:
            while last_step_index < len(steps):
                yield f"data: {json.dumps({'step': steps[last_step_index]})}\n\n"
                last_step_index += 1
            yield f"data: {json.dumps({'ping': True})}\n\n"
            time.sleep(0.1)

        # Send any remaining steps
        while last_step_index < len(steps):
            yield f"data: {json.dumps({'step': steps[last_step_index]})}\n\n"
            last_step_index += 1

        # Log chat interaction
        duration = round(time.time() - t_start, 2)
        parsed = result.get("parsed", {})
        data_store.log_chat(question, parsed.get("answer", ""), parsed.get("sources", []), duration)

        # Send final result
        yield f"data: {result['data']}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route("/papers")
def papers():
    log.debug("Papers list requested: %d papers", len(engine.paper_names))
    return jsonify(papers=engine.paper_names, total_chunks=len(engine.chunks))


@app.route("/clear", methods=["POST"])
def clear():
    log.info("Clearing knowledge base (%d papers, %d chunks)",
             len(engine.paper_names), len(engine.chunks))
    engine.clear()
    data_store.log_event("kb_cleared")
    return jsonify(message="Knowledge base cleared.")


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
