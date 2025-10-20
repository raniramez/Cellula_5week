# 🧠 LangGraph Python Code Assistant (Task 0)

This is a **LangGraph-powered RAG assistant** that can either **generate** or **explain** Python code.  
It follows the Task 0 spec from your internship PDF: a clear state machine with nodes for intent → retrieval → (generate/explain) → final.

---

## 🚀 What it does
- **Intent detection**: figures out whether you want to *generate* code or *explain* code.
- **Retrieval (RAG)**: uses FAISS + the **HumanEval** dataset to fetch similar code tasks.
- **LLM reasoning**: calls OpenAI **gpt-4o-mini** to write or explain clean Python.
- **Deterministic flow**: implemented with **LangGraph** nodes and conditional edges.

---

## 🧩 How it works (in one picture)
```
User → [intent] → [retrieve] → [generate | explain] → [final]
```

- `intent`: keyword-based classifier (no extra LLM call)  
- `retrieve`: embeds the query, searches HumanEval with FAISS (top‑5)  
- `generate`: returns runnable Python grounded on retrieved examples  
- `explain`: structured explanation: What/How/Concepts/Pitfalls/Improved example  

---

## 🔧 Requirements
Install once:
```bash
pip install -U langgraph langchain langchain-openai faiss-cpu datasets openai ipython
```

> If you’re on Apple Silicon and see FAISS issues, you can skip FAISS and use a simple Python list search temporarily.

---

## 🔑 OpenAI key
In Colab or terminal:
```python
import os, getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("🔑 Enter your OpenAI API key: ")
```

---

## ▶️ Run it
```bash
python task0_langgraph_rag_agent.py
```
Then try messages like:
- `write a python function that checks if a string is a palindrome`
- `explain what zip() does in python`
- paste a short function and ask: `explain this`

Type `exit` to quit.

---

## 📚 Dataset
- **OpenAI HumanEval** (Hugging Face): used to ground the model with real Python tasks.

---

## 🧪 For Task 1 (deployment)
This module exposes a small helper:
```python
from task0_langgraph_rag_agent import run_week5
print(run_week5("write a function to reverse a list"))
```
You can import this into a **Streamlit** app or **FastAPI** service easily.

---

## 📝 Notes
- Keep prompts short; if you paste very long code, consider trimming.
- Retrieval is semantic, so phrasing matters. Try rewording if results feel off.

