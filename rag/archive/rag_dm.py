# # import os
# # import shutil  # 用于删除文件夹（如果需要强制重建）
# # from typing import List, Optional, Any

# # # LangChain 组件
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_chroma import Chroma
# # from langchain.chains import RetrievalQA

# # # 适配器基类
# # from langchain_core.embeddings import Embeddings
# # from langchain_core.language_models.llms import LLM
# # from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# # import glm  # 导入上面的 glm.py

# # # --- 1. Embedding 适配器 ---
# # class GLMEmbeddings(Embeddings):
# #     def embed_documents(self, texts: List[str]) -> List[List[float]]:
# #         return [glm.get_embeddings(text) for text in texts]

# #     def embed_query(self, text: str) -> List[float]:
# #         return glm.get_embeddings(text)

# # # --- 2. LLM 适配器 ---
# # class GLMLLM(LLM):
# #     @property
# #     def _llm_type(self) -> str:
# #         return "glm-4-custom"

# #     def _call(
# #         self,
# #         prompt: str,
# #         stop: Optional[List[str]] = None,
# #         run_manager: Optional[CallbackManagerForLLMRun] = None,
# #         **kwargs: Any,
# #     ) -> str:
# #         return glm.query_model(prompt)

# # # --- 3. 构建 RAG 系统 (优化版) ---
# # def build_rag_system(pdf_directory):
# #     # 定义向量库的持久化路径
# #     persist_dir = "./chroma_db"
    
# #     # 实例化 Embedding 模型
# #     embeddings = GLMEmbeddings()

# #     # --- 核心逻辑修改：检查数据库是否存在 ---
# #     if os.path.exists(persist_dir) and os.listdir(persist_dir):
# #         print(f"1. 检测到现有向量库 ({persist_dir})，正在直接加载...")
# #         # 直接加载现有的数据库，不再读取 PDF
# #         vectorstore = Chroma(
# #             persist_directory=persist_dir, 
# #             embedding_function=embeddings
# #         )
# #         print("   数据库加载成功！")
        
# #     else:
# #         print("1. 未检测到向量库，正在从 PDF 构建...")
# #         if not os.path.exists(pdf_directory):
# #             raise FileNotFoundError(f"目录不存在: {pdf_directory}")
            
# #         pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
# #         if not pdf_files:
# #             raise ValueError("目录下没有找到 PDF 文件")
            
# #         documents = []
# #         for pdf_file in pdf_files:
# #             pdf_path = os.path.join(pdf_directory, pdf_file)
# #             loader = PyPDFLoader(pdf_path)
# #             documents.extend(loader.load())
        
# #         print(f"   已加载 {len(documents)} 页文档")

# #         print("2. 正在切分文本...")
# #         text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=800,
# #             chunk_overlap=150
# #         )
# #         splits = text_splitter.split_documents(documents)

# #         print("3. 正在向量化并存入数据库 (这可能需要一点时间)...")
# #         # from_documents 会自动创建并持久化
# #         vectorstore = Chroma.from_documents(
# #             documents=splits,
# #             embedding=embeddings,
# #             persist_directory=persist_dir
# #         )
# #         print("   向量库构建完成并已保存。")

# #     print("4. 构建检索问答链...")
# #     llm = GLMLLM()

# #     # 这里的 return_source_documents=True 是为了方便查看参考来源
# #     qa_chain = RetrievalQA.from_chain_type(
# #         llm=llm,
# #         chain_type="stuff",
# #         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
# #         return_source_documents=True 
# #     )
    
# #     return qa_chain

# # if __name__ == "__main__":
# #     # 修改您的路径
# #     target_directory = "/mnt/d/rag/rag文献库"
    
# #     # 可选：如果你添加了新 PDF 想强制重新生成，取消下面这行的注释
# #     # if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
    
# #     try:
# #         rag = build_rag_system(target_directory)
# #         print("\n=== 系统初始化完成 ===")
        
# #         while True:
# #             query = input("\n请输入问题 (输入 q 退出): ")
# #             if query.lower() == 'q': break
            
# #             print("AI 正在思考...")
# #             response = rag.invoke({"query": query})
            
# #             print(f"\n[AI 回答]: {response['result']}")
            
# #             # 打印来源（可选）
# #             print("\n--- 参考来源 ---")
# #             for doc in response.get('source_documents', []):
# #                 source = os.path.basename(doc.metadata.get('source', '未知'))
# #                 page = doc.metadata.get('page', 0) + 1
# #                 print(f"- {source} (第 {page} 页)")
# #             print("----------------")
            
# #     except Exception as e:
# #         import traceback
# #         traceback.print_exc()
# #         print(f"\n发生错误: {e}")
# import os
# import shutil
# from typing import List, Optional, Any

# # LangChain 组件
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# # --- 修改点 1: 引入对话检索链 ---
# from langchain.chains import ConversationalRetrievalChain

# # 适配器基类
# from langchain_core.embeddings import Embeddings
# from langchain_core.language_models.llms import LLM
# from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# import glm  # 导入 glm.py

# # --- 1. Embedding 适配器 (保持不变) ---
# class GLMEmbeddings(Embeddings):
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [glm.get_embeddings(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return glm.get_embeddings(text)

# # --- 2. LLM 适配器 (保持不变) ---
# class GLMLLM(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "glm-4-custom"

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         return glm.query_model(prompt)

# # --- 3. 构建 RAG 系统 (升级版) ---
# def build_rag_chain(pdf_directory):
#     """
#     构建并返回一个支持历史记录的 ConversationalRetrievalChain
#     """
#     # 屏蔽 ChromaDB 的遥测报错
#     os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
#     persist_dir = "./chroma_db"
#     embeddings = GLMEmbeddings()

#     # 1. 加载或构建向量库
#     if os.path.exists(persist_dir) and os.listdir(persist_dir):
#         print(f"检测到向量库 {persist_dir}，直接加载...")
#         vectorstore = Chroma(
#             persist_directory=persist_dir, 
#             embedding_function=embeddings
#         )
#     else:
#         print("未检测到向量库，正在重新构建...")
#         if not os.path.exists(pdf_directory):
#             os.makedirs(pdf_directory, exist_ok=True)
#             # 如果目录是空的或者不存在，这里会报错，需确保有文件
        
#         pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#         if not pdf_files:
#             # 如果没有文件，返回 None，让 UI 提示用户上传
#             return None
            
#         documents = []
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(pdf_directory, pdf_file)
#             loader = PyPDFLoader(pdf_path)
#             documents.extend(loader.load())
        
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
#         splits = text_splitter.split_documents(documents)

#         vectorstore = Chroma.from_documents(
#             documents=splits,
#             embedding=embeddings,
#             persist_directory=persist_dir
#         )

#     # 2. 初始化 LLM
#     llm = GLMLLM()

#     # 3. --- 修改点 2: 使用 ConversationalRetrievalChain ---
#     # 这个链条会自动处理：输入 + 历史 -> 独立问题 -> 检索 -> 生成回答
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#     )
    
#     return qa_chain
import os
from typing import List, Optional, Any

# LangChain 组件
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate  # 新增: 用于自定义提示词

# 适配器基类
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

import glm  # 导入您的 glm.py

# --- 1. Embedding 适配器 ---
class GLMEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 增加批处理时的错误处理或日志
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
def build_rag_chain(pdf_directory, use_history=True):
    # 屏蔽 ChromaDB 的遥测报错
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
    persist_dir = "./chroma_db"
    embeddings = GLMEmbeddings()

    # --- A. 加载或构建向量库 ---
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"检测到向量库 {persist_dir}，直接加载...")
        vectorstore = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
    else:
        print("未检测到向量库，正在重新构建...")
        if not os.path.exists(pdf_directory):
            # 防止目录不存在报错
            os.makedirs(pdf_directory, exist_ok=True)
            return None
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        if not pdf_files:
            return None
            
        documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    # --- B. 定义关键提示词 (修复不连贯的核心) ---
    # 这是一个专门用于“将后续问题改写为独立问题”的中文 Prompt
    chinese_condense_prompt = PromptTemplate.from_template(
        """设定：你是一个智能助手。
任务：结合下面的"对话历史"和"用户的新问题"，将"用户的新问题"改写为一个独立的、完整的搜索查询语句。
要求：
1. 如果新问题包含代词（如"它"、"这个"），请根据历史将其替换为具体的名词。
2. 如果新问题已经很完整，不需要改写，请直接原样输出。
3. 不要回答问题，只要输出改写后的句子。

对话历史：
{chat_history}

用户的新问题：{question}

改写后的独立问题："""
    )

    # --- C. 构建链 ---
    llm = GLMLLM()

    if use_history:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            condense_question_prompt=chinese_condense_prompt, # 注入中文提示词
            verbose=True # 开启日志，方便你在终端看到改写过程
        )
    else:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    return qa_chain
