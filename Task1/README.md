# 🧠 Cellula Internship — Task 1: Unified System Deployment (Weeks 3–5)

This repository merges **Weeks 3, 4, and 5** of the Cellula internship into **one deployable system**, as required by *Task 1: All System Deployment*.

---

## 🚀 Overview

This unified app integrates everything developed across the previous weeks:

| Week | Focus | Frameworks & Tools |
|------|--------|--------------------|
| **Week 3** | Retrieval-Augmented Code Generator (RAG) | FAISS, Sentence-Transformers, Transformers (TinyLlama) |
| **Week 4** | LangChain Utilities (Memory, RAG, Metrics, Evaluation) | LangChain, ChromaDB, DeepEval, Phi-3-mini |
| **Week 5** | LangGraph Python Assistant | LangGraph, OpenAI GPT-4o-mini |

All three systems are merged into one cohesive Streamlit app that can be deployed on **Streamlit Cloud**, **Colab (via Cloudflare)**, or locally.

---

## 🧩 Features

- **Single Interface** – One Streamlit app for all modules.
- **Modular Tabs** – Each week’s logic runs independently inside the app.
- **OpenAI Integration** – Add your API key securely via sidebar.
- **RAG + Memory + Evaluation** – Combines vector search, chain memory, and DeepEval metrics.
- **LangGraph Assistant** – A state-machine-driven agent that can generate or explain Python code.

---

## 🖥️ Run Locally

### 1️⃣ Setup Environment
```bash
git clone https://github.com/<your-username>/cellula_task1_unified.git
cd cellula_task1_unified

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2️⃣ Launch App
```bash
streamlit run run_this.py
```
Then open the URL shown in your terminal (default: http://localhost:8501).

---

## 🧠 Run on Google Colab

Colab doesn’t display Streamlit inline, but you can expose it via **Cloudflare**:

```python
!apt-get install -y wget
!unzip -q cellula_task1_unified.zip -d /content/cellula_task1_unified
%cd /content/cellula_task1_unified
!pip install -q -r requirements.txt faiss-cpu==1.8.0.post1 cloudflared

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

import re, subprocess, time
proc = subprocess.Popen(["/usr/local/bin/cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    if "trycloudflare.com" in line:
        print(line.strip())
        break

!streamlit run run_this.py --server.headless true --server.port 8501 &
```

Then open the printed `trycloudflare.com` link.

---

## 🧱 Repository Structure

```
cellula_task1_unified/
├── run_this.py                 # ✅ Main entry point to launch the Streamlit app
├── requirements.txt            # Dependencies
├── README.md                   # Project description
├── weeks/
│   ├── week3/                  # RAG code generator (FAISS + TinyLlama)
│   ├── week4/                  # LangChain utilities (memory, RAG, metrics, evaluation)
│   └── week5/                  # LangGraph code assistant
└── utils/                      # Future utilities (optional)
```

---

## 🔐 API Key Setup

Some modules (Week 4 Task 4 and Week 5) require an **OpenAI API key**.

- Add your key in the Streamlit sidebar: `OPENAI_API_KEY`
- Or set it manually before running:
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "sk-..."
  ```
> The key is **never saved** or uploaded — it only lives in your runtime memory.

---

## 📦 Deployment on Streamlit Cloud

1. Push this project to GitHub.  
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud).  
3. Connect your GitHub repo and select `run_this.py` as the entry point.  
4. Add your `OPENAI_API_KEY` in Streamlit **Secrets**.  
5. Deploy and share your public URL.

---

## 🧑‍💻 Author
**Rani Ramez**  
Mechatronics & AI Engineering | Ain Shams University  
📧 [LinkedIn Profile](https://www.linkedin.com/in/raniramez)

---

## 🏁 Notes
- You can safely run this on CPU (TinyLlama + Sentence-Transformers work fine).  
- FAISS or Chroma will create small local folders for embeddings.  
- Cloudflare tunnel automatically closes when Colab stops.

---
