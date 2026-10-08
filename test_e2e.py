import urllib.request
import json
import sys

print("--- 1. Testing GET /sample-papers ---")
req = urllib.request.Request("http://127.0.0.1:8000/sample-papers")
with urllib.request.urlopen(req) as resp:
    samples = json.loads(resp.read().decode())
    print("Available samples:", [s["id"] for s in samples])

print("\n--- 2. Testing POST /load-sample/transformer ---")
req = urllib.request.Request("http://127.0.0.1:8000/load-sample/transformer", method="POST")
with urllib.request.urlopen(req) as resp:
    load_res = json.loads(resp.read().decode())
    print("Paper title:", load_res["title"])
    print("Total pages:", load_res["total_pages"])
    print("Total chunks:", load_res["total_chunks"])
    print("Sample questions:", load_res["sample_questions"][:2])

print("\n--- 3. Testing POST /ask-question ---")
q_data = json.dumps({
    "paper_id": "transformer",
    "question": "What is the mathematical formula for Scaled Dot-Product Attention?"
}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/ask-question",
    data=q_data,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    ans_res = json.loads(resp.read().decode())
    print("Model used:", ans_res.get("model_used"))
    print("Citations received:", len(ans_res.get("citations", [])))
    for i, c in enumerate(ans_res.get("citations", [])[:2]):
        print(f"  Citation {i+1}: Page {c.get('page')} ({c.get('section')})")
    print("\nAnswer preview:")
    clean_ans = ans_res.get("answer", "")[:400].encode("ascii", "ignore").decode()
    print(clean_ans)

print("\n--- 4. Testing GET / (Frontend) ---")
req = urllib.request.Request("http://127.0.0.1:8000/")
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode()
    print(f"Frontend HTTP status: {resp.status}, HTML bytes: {len(html)}")
    assert "Paper Explorer" in html

print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY!")
