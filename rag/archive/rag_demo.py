# import os
# # 设置 API Key (如果使用本地模型 Ollama 则不需要这一步)
# os.environ["OPENAI_API_KEY"] = "sk-您的OpenAI-Key"

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA

# def build_rag_system(pdf_path):
#     print("1. 正在加载文档...")
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     print("2. 正在切分文本...")
#     # 医疗文档通常比较严谨，Chunk Size 不宜过小，避免断章取义
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,  # 每个块的大小
#         chunk_overlap=200 # 重叠部分，防止上下文丢失
#     )
#     splits = text_splitter.split_documents(documents)

#     print("3. 正在向量化并存入数据库 (Chroma)...")
#     # 使用 OpenAI 的 Embedding 模型将文本转为向量
#     embeddings = OpenAIEmbeddings()
#     # 持久化存储到本地 'db' 文件夹，避免每次重启都要重新向量化
#     vectorstore = Chroma.from_documents(
#         documents=splits, 
#         embedding=embeddings,
#         persist_directory="./chroma_db" 
#     )

#     print("4. 构建检索问答链...")
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # 温度设为0，让回答更严谨
    
#     # 核心：将检索器 (Retriever) 和 LLM 串联
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff", # "stuff" 意为将检索到的内容全部塞入 Prompt
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}) # 只找最相关的3段
#     )
    
#     return qa_chain

# # --- 运行测试 ---
# if __name__ == "__main__":
#     # 假设目录下有一个医疗指南 pdf
#     try:
#         rag = build_rag_system("sample_medical_guide.pdf")
        
#         while True:
#             query = input("\n请输入问题 (输入 q 退出): ")
#             if query == 'q': break
            
#             # 获取答案
#             response = rag.invoke({"query": query})
#             print(f"\n[AI 回答]: {response['result']}")
            
#             # (可选) 打印出AI参考了哪些原文片段，用于核查
#             # print(f"参考来源: {response['source_documents']}")
            
#     except Exception as e:
#         print(f"发生错误: {e}")
#         print("请确保目录下存在 pdf 文件且 API Key 正确。")
import os
from typing import List, Optional, Any

# LangChain 组件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# 适配器基类
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

import glm  # 导入上面的 glm.py

# --- 1. Embedding 适配器 ---
class GLMEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [glm.get_embeddings(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return glm.get_embeddings(text)

# --- 2. LLM 适配器 ---
class GLMLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "glm-4-custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return glm.query_model(prompt)

# --- 3. 构建 RAG 系统 ---
def build_rag_system(pdf_directory):
    print("1. 正在加载文档...")
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"目录不存在: {pdf_directory}")
        
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        raise ValueError("目录下没有找到 PDF 文件")
        
    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    print(f"已加载 {len(documents)} 页文档")

    print("2. 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)

    print("3. 正在向量化并存入数据库 (Chroma)...")
    embeddings = GLMEmbeddings() # 使用适配器
    
    # 这里的 persist_directory 只要指定一个文件夹路径即可
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("4. 构建检索问答链...")
    llm = GLMLLM() # 使用适配器

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    
    return qa_chain

if __name__ == "__main__":
    # 修改您的路径
    target_directory = "/mnt/d/rag文献库" 
    
    try:
        rag = build_rag_system(target_directory)
        print("\n=== 系统初始化完成 ===")
        
        while True:
            query = input("\n请输入问题 (输入 q 退出): ")
            if query.lower() == 'q': break
            
            print("AI 正在思考...")
            response = rag.invoke({"query": query})
            print(f"\n[AI 回答]: {response['result']}")
            
    except Exception as e:
        print(f"\n发生错误: {e}")
