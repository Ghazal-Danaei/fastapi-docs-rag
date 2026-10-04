import glob, pathlib, sys
import anthropic, chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
DOCS = "data/raw/**/*.md"
MODEL = "claude-opus-5"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(path="data/chroma").get_or_create_collection("naive", embedding_function=ef)
client = anthropic.Anthropic()


def chunk(text, size=1000, overlap=100):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def ingest():
    for path in glob.glob(DOCS, recursive=True):
        chunks = chunk(pathlib.Path(path).read_text(encoding="utf-8"))
        col.upsert(
            ids=[f"{path}::{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path}] * len(chunks),
        )
    print(f"{col.count()} chunks indexed")


def answer(question):
    res = col.query(query_texts=[question], n_results=5)
    context = "\n\n".join(
        f"[{i + 1}] (source: {m['source']})\n{d}"
        for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0]))
    )
    msg = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system="Answer only from the provided context. Cite sources like [1]. "
               "If the context does not contain the answer, say you don't know.",
        messages=[{"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if the model declines, the API retries on a fallback model
    )
    if msg.stop_reason == "refusal":
        return "Request declined."
    return "".join(b.text for b in msg.content if b.type == "text")


if __name__ == "__main__":
    ingest() if sys.argv[1] == "ingest" else print(answer(" ".join(sys.argv[1:])))
