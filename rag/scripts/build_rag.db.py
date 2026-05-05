import os
import shutil

# ⚠️ 屏蔽 ChromaDB 烦人的 telemetry 报错
# 🚀 核心修复：强制让 HuggingFace 组件使用国内高速镜像源下载模型！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# 使用本地免费开源的 HuggingFace Embedding 模型
from langchain_huggingface import HuggingFaceEmbeddings

def rebuild_vector_database(pdf_directory: str, db_directory: str = "./chroma_db"):
    print("===========================================")
    print(" 📚 正在启动知识库离线构建引擎 (纯本地版)...")
    print("===========================================")

    if os.path.exists(db_directory):
        print(f"⚠️ 发现旧数据库目录 '{db_directory}'，正在清理...")
        shutil.rmtree(db_directory)
        print("✅ 旧数据库已清空！")

    # 1. 加载文档
    print(f"\n📂 1. 正在扫描目录: {pdf_directory}")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"   -> 正在读取: {pdf_file}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    print(f"✅ 已成功读取 {len(documents)} 页 PDF 文献。")

    # 2. 切分文本
    print("\n✂️ 2. 正在进行语义切块 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 文档已被切分为 {len(splits)} 个知识碎片。")

    # 3. 向量化 (使用 BAAI 智源的开源中文模型，第一次运行会自动下载，约 100MB)
    print("\n🧠 3. 正在加载本地 Embedding 模型并构建向量库...")
    print("   (初次运行会从 HuggingFace 下载轻量级模型，请耐心等待)")
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_directory 
    )

    print("\n🎉 知识库构建大功告成！全流程 0 成本，且数据未离开本地！")

if __name__ == "__main__":
    TARGET_PDF_DIRECTORY = "/mnt/d/rag/rag文献库" # 确保路径正确
    rebuild_vector_database(TARGET_PDF_DIRECTORY)