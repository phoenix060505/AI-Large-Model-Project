import os
import json
import textwrap
from typing import List, Dict, Any

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from rag_config import (
    INDEX_DIR,
    EMBED_MODEL_PATH,
    TOP_K,
    MAX_EMBED_LEN,
    EMBED_DEVICE,
)

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/autodl-fs/data/models/Qwen3-32B"
)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths
    ]


def get_query_instruction(query: str) -> str:
    task = "Given a Chinese user question, retrieve relevant passages that answer the question"
    return f"Instruct: {task}\nQuery:{query}"


class Qwen3Embedder:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 8192):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        print(f"[INFO] Loading embedding tokenizer: {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[INFO] Loading embedding model on {self.device}", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode_query(self, query: str):
        query_text = get_query_instruction(query)

        batch = self.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)
        emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)

        return emb.float().cpu().numpy().astype("float32")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


class FaissRetriever:
    def __init__(self, index_dir: str, embedder: Qwen3Embedder):
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.items_path = os.path.join(index_dir, "items.jsonl")

        self.index = faiss.read_index(self.index_path)
        self.items = load_jsonl(self.items_path)
        self.embedder = embedder

        print(f"[INFO] Loaded FAISS index: {self.index_path}", flush=True)
        print(f"[INFO] Loaded items: {len(self.items)}", flush=True)

    def search(self, query: str, top_k: int = 5):
        q_emb = self.embedder.encode_query(query)
        scores, ids = self.index.search(q_emb, top_k)

        results = []

        for rank, idx in enumerate(ids[0], start=1):
            if idx < 0:
                continue

            item = self.items[int(idx)]
            results.append({
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "id": item.get("id", int(idx)),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "index_text": item.get("index_text", ""),
            })

        return results


def build_context(results: List[Dict[str, Any]], max_chars: int = 6000):
    parts = []

    for r in results:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()

        if q or a:
            text = f"问题：{q}\n答案：{a}"
        else:
            text = str(r.get("index_text", "")).strip()

        parts.append(f"[资料{r['rank']} | 相似度 {r['score']:.4f}]\n{text}")

    context = "\n\n".join(parts)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[提示：后续参考资料因长度限制已截断]"

    return context


def build_messages(question: str, context: str):
    system_prompt = (
        "你是一个严谨的中文问答助手。"
        "你必须优先根据给定参考资料回答问题。"
        "如果参考资料中没有足够信息，请回答“根据现有资料无法确定”。"
        "不要编造参考资料中不存在的内容。"
    )

    user_prompt = f"""请根据下面参考资料回答问题。

参考资料：
{context}

用户问题：
{question}

回答要求：
1. 优先依据参考资料。
2. 如果资料不足，回答“根据现有资料无法确定”。
3. 用中文回答，简洁准确。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class QwenGenerator:
    def __init__(self, model_path: str):
        print(f"[INFO] Loading LLM tokenizer: {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"[INFO] Loading LLM model: {model_path}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        self.model.eval()

    @torch.no_grad()
    def generate(self, question: str, context: str):
        messages = build_messages(question, context)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.8,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return answer.strip()


def print_sources(results):
    print("\n检索到的参考资料：")
    for r in results:
        q = str(r.get("question", "")).replace("\n", " ")[:120]
        a = str(r.get("answer", "")).replace("\n", " ")[:180]
        print("-" * 80)
        print(f"[{r['rank']}] score={r['score']:.4f}, id={r.get('id')}")
        print("Q:", q)
        print("A:", a)


def main():
    print("=" * 80)
    print("[RAG Chat - Transformers Backend]")
    print("INDEX_DIR        :", INDEX_DIR)
    print("EMBED_MODEL_PATH :", EMBED_MODEL_PATH)
    print("LLM_MODEL_PATH   :", LLM_MODEL_PATH)
    print("EMBED_DEVICE     :", EMBED_DEVICE)
    print("TOP_K            :", TOP_K)
    print("=" * 80)

    embedder = Qwen3Embedder(
        EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )

    retriever = FaissRetriever(INDEX_DIR, embedder)

    generator = QwenGenerator(LLM_MODEL_PATH)

    print("\n输入问题开始问答。输入 exit / quit / q 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        results = retriever.search(question, top_k=TOP_K)
        print_sources(results)

        context = build_context(results)

        print("\n正在生成回答...\n")
        answer = generator.generate(question, context)

        print("=" * 80)
        print("最终回答：")
        print(textwrap.fill(answer, width=100))
        print("=" * 80)


if __name__ == "__main__":
    main()
