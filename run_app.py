import os
import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.5)
    print("Opening Paper Explorer in your browser: http://127.0.0.1:8000")
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("==================================================")
    print("       Paper Explorer - AI Research Assistant     ")
    print("==================================================")
    print("Backend: FastAPI + ChromaDB + SentenceTransformers")
    print("LLM: OpenRouter (Grounded RAG)")
    print("Serving on: http://127.0.0.1:8000")
    print("==================================================")

    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
