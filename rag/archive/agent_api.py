import os
import json
import shutil
import asyncio
import httpx
import redis  # 引入 Redis
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from zhipuai import ZhipuAI

# ================= RAG 知识库与环境变量 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("⏳ 正在挂载本地医学知识库...")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=hf_embeddings)
print("✅ 本地知识库挂载完毕！")

# ================= 🚀 Redis 客户端初始化 =================
# 连接本地默认的 Redis 服务 (6379端口)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

try:
    redis_client.ping()
    print("✅ Redis 状态缓存服务连接成功！系统已具备高并发多用户隔离能力。")
except Exception as e:
    print(f"❌ Redis 连接失败，请检查服务是否启动: {e}")
# =============================================================

app = FastAPI(title="Agent Router API (Redis + Async + RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M" 
client = ZhipuAI(api_key=API_KEY)
TOOL_BACKEND_URL = "http://127.0.0.1:8000"

# 提取固定的系统提示词
SYSTEM_PROMPT = (
    "你是一个专业的眼底影像科AI多智能体中枢。根据用户的意图灵活调度工具：\n"
    "1. 【仅分割】：调用 vision_segmentation_tool，拿到结果后总结，不要诊断。\n"
    "2. 【诊断分析】：调用 vision 获取特征 -> 调用 disease_diagnosis 获取疾病标签。\n"
    "3. 【医学咨询/出具方案】：当得知疾病结果后并主动要求咨询报告，或者用户主动询问医学知识、治疗方案时，你【必须】调用 medical_knowledge_retrieval_tool 查阅指南。\n"
    "4. 【异常拦截】：若工具返回“系统拦截”，终止流程并解释。\n"
    "5. ⚠️【严格控制】：只有用户明确要咨询方案，或者明确要诊断报告之类的话，才调用 medical_knowledge_retrieval_tool，否则诊断完直接给出结果即可。\n"
    "回复时使用专业、温和的自然语言，严禁输出代码块。"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "vision_segmentation_tool",
            "description": "第一步：视觉感知工具。提取RV和FAZ掩码及8项血流指标。分析图片必须先调这个。",
            "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "disease_diagnosis_tool",
            "description": "第二步：临床诊断工具。基于视觉特征进行疾病分类打分。诊断前必须调用。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "medical_knowledge_retrieval_tool",
            "description": "第三步：RAG 知识库检索工具。只有用户要求给出具体治疗方案、用药建议，或回答专业医学问题时才调用此工具。",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "需要查阅的具体医学问题"}}, "required": ["query"]}
        }
    }
]

# ================= 🚀 Redis 状态管理助手函数 =================
def get_session_state(session_id: str):
    """从 Redis 获取用户的专属记忆和上下文"""
    data = redis_client.get(f"agent_session:{session_id}")
    if data:
        return json.loads(data)
    else:
        # 如果是新用户，初始化专属的干净状态
        return {
            "history": [{"role": "system", "content": SYSTEM_PROMPT}],
            "context": {"rv_mask_base64": None, "faz_mask_base64": None, "metrics": None, "scan_type": None}
        }

def save_session_state(session_id: str, state: dict):
    """保存状态到 Redis，并设置 24 小时过期时间（自动清理内存）"""
    redis_client.setex(f"agent_session:{session_id}", 86400, json.dumps(state))
# ===================================================================

@app.post("/api/chat")
# 💡 强行把默认的 session_id 写死成 "test_doctor_001"，避免跟之前的记忆撞车
async def chat_with_agent(text: str = Form(""), file: UploadFile = File(None), session_id: str = Form("test_doctor_001")):
    user_prompt = text.strip()
    image_path = None
    file_info_msg = ""

    # 每次请求进来，先从 Redis 取出属于该用户的记忆
    state = get_session_state(session_id)

    if file:
        os.makedirs("temp_uploads", exist_ok=True)
        image_path = os.path.abspath(f"temp_uploads/{file.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_info_msg = f"\n[系统：用户上传了图片 {image_path}。请严格根据意图调用工具。]"
        
        # 💡 如果不说话，默认只要求诊断，不索要方案
        if not user_prompt:
            user_prompt = "请帮我分析并诊断这张图。" 

    final_user_content = user_prompt + file_info_msg
    state["history"].append({"role": "user", "content": final_user_content})

    # 滑动窗口记忆：防止单个用户聊得太多导致 Token 爆炸
    if len(state["history"]) > 15:
        state["history"] = [state["history"][0]] + state["history"][-14:]

    # 先保存一下用户的输入
    save_session_state(session_id, state)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'thinking', 'msg': 'Node 1: 正在识别意图与规划工作流...'})}\n\n"
        await asyncio.sleep(0.1) 

        max_loops = 4 
        current_loop = 0
        final_text = "分析完毕。"

        while current_loop < max_loops:
            current_loop += 1
            
            response = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=state["history"], 
                tools=tools,
                temperature=0.1
            )
            msg = response.choices[0].message
            state["history"].append(msg.model_dump())
            save_session_state(session_id, state) 

            if not msg.tool_calls:
                final_text = msg.content
                yield f"data: {json.dumps({'type': 'thinking', 'msg': '正在生成最终报告...'})}\n\n"
                await asyncio.sleep(0.1)
                break
                
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                if func_name == "vision_segmentation_tool":
                    yield f"data: {json.dumps({'type': 'thinking', 'msg': 'Node: 唤醒 Vision Agent，正在提取特征 (异步)...'})}\n\n"
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as http_client:
                            with open(args.get("image_path", image_path), "rb") as f:
                                files = {"file": f}
                                res = await http_client.post(f"{TOOL_BACKEND_URL}/api/v1/agent/vision/analyze", files=files)
                        
                        if res.status_code != 200:
                            tool_result = f"系统拦截：{res.json().get('detail', '尺寸不支持')}。"
                        else:
                            data = res.json()
                            if not data["image_metadata"]["is_valid_octa"]:
                                tool_result = "系统拦截：血管密度极度异常，判定为非标准 OCTA！警告用户，停止诊断。"
                                yield f"data: {json.dumps({'type': 'vision_data', 'data': data})}\n\n"
                            else:
                                state["context"]["rv_mask_base64"] = data["visualizations"]["rv_mask_base64"]
                                state["context"]["faz_mask_base64"] = data["visualizations"]["faz_mask_base64"]
                                state["context"]["metrics"] = data["metrics"]
                                state["context"]["scan_type"] = data["image_metadata"]["scan_type"]
                                save_session_state(session_id, state) 
                                
                                tool_result = f"视觉分割成功。指标：{data['metrics']}。如果需要诊断，请继续调用 disease_diagnosis_tool。"
                                yield f"data: {json.dumps({'type': 'vision_data', 'data': data})}\n\n"
                    except Exception as e:
                        tool_result = f"视觉异常: {str(e)}"

                elif func_name == "disease_diagnosis_tool":
                    yield f"data: {json.dumps({'type': 'thinking', 'msg': 'Node: 唤醒 Clinical Agent，进行病变研判 (异步)...'})}\n\n"
                    payload = {
                        "scan_type": state["context"]["scan_type"], 
                        "rv_mask_base64": state["context"]["rv_mask_base64"],
                        "faz_mask_base64": state["context"]["faz_mask_base64"],
                        "metrics": state["context"]["metrics"]
                    }
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as http_client:
                            res = await http_client.post(f"{TOOL_BACKEND_URL}/api/v1/agent/classify", json=payload)
                        data = res.json()
                        pred_disease = data['prediction']['label_cn']
                        
                        # 💡 铁腕防爆改：强行按住大模型，禁止它主动去搜 RAG！
                        tool_result = (
                            f"诊断成功。初步判定疾病为：{pred_disease}。\n"
                            f"🛑【系统级最高指令】：现在请立即停止思考！【绝对禁止】调用 medical_knowledge_retrieval_tool，除非用户在刚才的对话中明确打出了“治疗”、“方案”、“怎么治”、“报告”等字眼！请直接用一句话输出诊断结果，结束工作流！"
                        )
                        
                        yield f"data: {json.dumps({'type': 'classify_data', 'data': data})}\n\n"
                    except Exception as e:
                        tool_result = f"诊断异常: {str(e)}"

                elif func_name == "medical_knowledge_retrieval_tool":
                    search_query = args.get("query", "眼底疾病")
                    yield f"data: {json.dumps({'type': 'thinking', 'msg': f'Node: 唤醒 RAG Agent，异步查阅知识库 [{search_query}]...'})}\n\n"
                    
                    try:
                        docs = await asyncio.to_thread(vector_db.similarity_search, search_query, k=3)
                        retrieved_text = "\n\n".join([f"【文献参考 {i+1}】: {doc.page_content}" for i, doc in enumerate(docs)])
                        tool_result = f"查阅成功。以下是本地文献片段，请结合诊断结果给出建议：\n{retrieved_text}"
                    except Exception as e:
                        tool_result = f"知识检索失败: {str(e)}"

                else:
                    tool_result = "未知工具。"

                state["history"].append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})
                save_session_state(session_id, state) 

        yield f"data: {json.dumps({'type': 'final_text', 'text': final_text})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")