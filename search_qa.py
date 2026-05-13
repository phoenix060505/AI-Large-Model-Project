import os
import json

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from rag_config import INDEX_DIR, EMBED_MODEL_PATH, TOP_K, MAX_EMBED_LEN, EMBED_DEVICE


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


def get_query_instruction(query):
    task = "Given a Chinese user question, retrieve relevant passages that answer the question"
    return f"Instruct: {task}\nQuery:{query}"


class Qwen3Embedder:
    def __init__(self, model_path, device="cuda", max_length=8192):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode_query(self, query):
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


def load_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    items_path = os.path.join(INDEX_DIR, "items.jsonl")

    index = faiss.read_index(index_path)
    items = load_items(items_path)

    embedder = Qwen3Embedder(
        EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )

    while True:
        query = input("\n请输入问题，输入 exit 退出：").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break

        q_emb = embedder.encode_query(query)
        scores, ids = index.search(q_emb, TOP_K)

        print("\n检索结果：")
        for rank, idx in enumerate(ids[0], start=1):
            item = items[int(idx)]
            score = float(scores[0][rank - 1])

            print("=" * 80)
            print(f"[{rank}] score={score:.4f} id={item['id']}")
            print("问题：", item.get("question", "")[:300])
            print("答案：", item.get("answer", "")[:500])


if __name__ == "__main__":
    main()
