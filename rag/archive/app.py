import streamlit as st
import os
from rag_dm import build_rag_chain

MAX_HISTORY_TURNS = 4


def should_use_history(question):
    # Heuristic: only use history when the user references prior context.
    cues = [
        "它", "这个", "上述", "上文", "前面", "后面", "这项", "该", "其",
        "这些", "那些", "刚才", "之前", "刚刚", "同上", "前文", "如下",
    ]
    return any(cue in question for cue in cues)

# --- 页面配置 ---
st.set_page_config(page_title="RAG 智能助手", layout="wide")
st.title("🤖 基于 GLM-4 的文档问答助手")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📚 知识库设置")
    target_directory = st.text_input("PDF 文件夹路径", value="/mnt/d/rag/rag文献库")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置知识库"):
            st.cache_resource.clear()
            st.success("已清除缓存")
    with col2:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

# --- 核心逻辑：加载 RAG 链 ---
@st.cache_resource(show_spinner="正在加载向量数据库...")
def load_chain(directory):
    return build_rag_chain(directory)

# --- 初始化状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# LangChain 需要的历史格式是 [(q, a), (q, a)]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 显示历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 处理输入 ---
if prompt := st.chat_input("请输入您的问题..."):
    # 1. UI 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 调用模型
    chain = load_chain(target_directory)
    
    if chain is None:
        st.error("无法加载知识库，请检查路径及文件。")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("AI 正在思考 (并在结合上下文)...")
            
            try:
                # 3. 关键调用：传入 query 和 chat_history
                # chat_history 必须是列表的元组：[('Q1', 'A1'), ('Q2', 'A2')]
                history_for_query = st.session_state.chat_history if should_use_history(prompt) else []
                history_for_query = history_for_query[-MAX_HISTORY_TURNS:]
                result = chain.invoke({
                    "question": f"{prompt}。请用中文回答。",
                    "chat_history": history_for_query
                })
                
                answer = result["answer"]
                source_docs = result["source_documents"]

                # 4. 格式化来源
                source_text = "\n\n--- **参考来源** ---\n"
                seen_sources = set()
                for doc in source_docs:
                    src_name = os.path.basename(doc.metadata.get('source', '未知文件'))
                    if src_name not in seen_sources:
                        source_text += f"- 📄 `{src_name}`\n"
                        seen_sources.add(src_name)
                
                final_response = answer + source_text
                
                # 5. UI 显示回答
                message_placeholder.markdown(final_response)
                
                # 6. 更新状态 (UI 历史 + 逻辑历史)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                # 这一步至关重要：保存“纯净”的问答对，不要包含 Prompt 里的指令
                st.session_state.chat_history.append((prompt, answer))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                message_placeholder.error(f"发生错误: {str(e)}")
