

import os, numpy as np, faiss, getpass
from typing import TypedDict, Literal, Dict, Any, List
from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


os.environ["OPENAI_API_KEY"] = getpass.getpass("🔑 Enter your OpenAI API key: ")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


print("Loading HumanEval dataset...")
dataset = load_dataset("openai/openai_humaneval")["test"]

corpus = [
    {
        "id": row["task_id"],
        "text": f"{row['prompt']}\n\n{row['canonical_solution']}",
    }
    for row in dataset
]

texts = [c["text"] for c in corpus]
print(f"Embedding {len(texts)} code snippets...")
embeddings = embeddings_model.embed_documents(texts)

dim = len(embeddings[0])
index = faiss.IndexFlatIP(dim)
embeddings = np.array(embeddings, dtype="float32")
faiss.normalize_L2(embeddings)
index.add(embeddings)

print(f"✅ Index ready with {index.ntotal} examples.\n")



class AgentState(TypedDict, total=False):
    """Defines all the data the agent keeps track of between nodes."""
    message: str
    intent: Literal["generate", "explain", "unknown"]
    retrieved: List[Dict[str, Any]]
    answer: str
    history: List[Dict[str, str]]



def classify_intent(state: AgentState) -> AgentState:
    """Decides if the user wants to generate or explain code."""
    msg = (state.get("message") or "").lower()
    intent: Literal["generate", "explain", "unknown"] = "unknown"

    gen_hints = ["write", "generate", "build", "create", "implement", "code for", "make a function", "solution"]
    exp_hints = ["explain", "what does", "how it works", "walk me through", "why"]

    if any(h in msg for h in gen_hints):
        intent = "generate"
    elif any(h in msg for h in exp_hints) or ("\n" in msg and ("def " in msg or "class " in msg)):
        intent = "explain"

    out = dict(state)
    out["intent"] = intent
    print(f"[Intent detected: {intent}]")
    return out


def retrieve_examples(state: AgentState) -> AgentState:
    """Finds the top 5 similar code snippets from HumanEval."""
    query = state.get("message", "")
    q = embeddings_model.embed_query(query)
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, 5)

    results = [
        {"id": corpus[i]["id"], "score": float(s), "text": corpus[i]["text"]}
        for s, i in zip(scores[0], idxs[0])
    ]
    print(f"🔍 Retrieved {len(results)} examples from HumanEval.")
    out = dict(state)
    out["retrieved"] = results
    return out


def generate_code(state: AgentState) -> AgentState:
    """Uses GPT-4o-mini to generate Python code based on retrieved examples."""
    user = state.get("message", "")
    grounding = "\n\n".join(
        f"[{r['id']} | score={r['score']:.3f}]\n{r['text']}" for r in state.get("retrieved", [])
    )
    system = (
        "You are a precise Python code generator. "
        "Return clean, runnable code with short comments. Prefer readability and correctness."
    )
    prompt = f"User request:\n{user}\n\nRelevant examples:\n{grounding}\n\nRespond with code only if possible."
    ai = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}])

    out = dict(state)
    out["answer"] = ai.content
    return out


def explain_code(state: AgentState) -> AgentState:
    """Uses GPT-4o-mini to explain Python code clearly and concisely."""
    user = state.get("message", "")
    grounding = "\n\n".join(
        f"[{r['id']} | score={r['score']:.3f}]\n{r['text']}" for r in state.get("retrieved", [])
    )
    system = (
        "You are a helpful Python explainer. Use short sections:\n"
        "- What it does\n- How it works\n- Key concepts\n- Pitfalls\n- Tiny improved example"
    )
    prompt = f"User input:\n{user}\n\nRelated examples:\n{grounding}"
    ai = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": prompt}])

    out = dict(state)
    out["answer"] = ai.content
    return out


def finalize(state: AgentState) -> AgentState:
    """Updates chat history and returns the final response."""
    hist = state.get("history", [])
    hist = hist + [{"role": "assistant", "content": state.get("answer", "[no answer]")}]
    out = dict(state)
    out["history"] = hist
    return out



graph = StateGraph(AgentState)
graph.add_node("intent", classify_intent)
graph.add_node("retrieve", retrieve_examples)
graph.add_node("generate", generate_code)
graph.add_node("explain", explain_code)
graph.add_node("final", finalize)

graph.set_entry_point("intent")
graph.add_edge("intent", "retrieve")

def router(state: AgentState):
    """Directs the graph to 'generate' or 'explain' depending on intent."""
    i = state.get("intent", "unknown")
    return i if i in {"generate", "explain"} else "explain"

graph.add_conditional_edges("retrieve", router, {"generate": "generate", "explain": "explain"})
graph.add_edge("generate", "final")
graph.add_edge("explain", "final")
graph.add_edge("final", END)

app = graph.compile(checkpointer=MemorySaver())


display(Image(app.get_graph().draw_mermaid_png()))


print("\n=== LangGraph Code Assistant Ready ===")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ").strip()
    if user.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    state = app.invoke({"message": user}, config={"configurable": {"thread_id": "colab"}})
    print("\nAssistant:\n", state.get("answer", "[no answer]"))
