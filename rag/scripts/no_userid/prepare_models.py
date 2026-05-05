import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
MODEL_DIR = Path(os.getenv("EMBEDDING_MODEL_DIR", "./models/bge-small-zh-v1.5")).resolve()

MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
print(f"开始下载 embedding 模型: {MODEL_NAME}")
print(f"保存目录: {MODEL_DIR}")

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("下载完成，后续请通过目录加载模型。")
