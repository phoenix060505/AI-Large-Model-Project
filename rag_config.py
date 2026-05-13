import os

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/AI-Large-Model-Project/sustech_qa_pairs.jsonl"
)

INDEX_DIR = os.environ.get(
    "INDEX_DIR",
    "/root/AI-Large-Model-Project/rag_index_sustech_qwen3_embedding"
)

EMBED_MODEL_PATH = os.environ.get(
    "EMBED_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen/Qwen3-Embedding-0___6B"
)

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/autodl-fs/data/models/Qwen3-32B"
)

VLLM_URL = os.environ.get(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions"
)

VLLM_MODEL = os.environ.get(
    "VLLM_MODEL",
    "qwen3-32b"
)

TOP_K = int(os.environ.get("TOP_K", "5"))

# 如果 vLLM 占用显存太多，可以运行前 export EMBED_DEVICE=cpu
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

MAX_EMBED_LEN = int(os.environ.get("MAX_EMBED_LEN", "8192"))
