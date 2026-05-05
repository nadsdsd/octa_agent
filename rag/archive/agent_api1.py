import os
import json
import shutil
import asyncio
import httpx  # 引入异步 HTTP 请求库
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from zhipuai import ZhipuAI

# ================= 新增：RAG 知识库依赖与全局挂载 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("⏳ 正在挂载本地医学知识库与词向量模型...")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=hf_embeddings)
print("✅ 本地知识库挂载完毕！")
# ===================================================================

app = FastAPI(title="Agent Router API (Async + SSE + RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M" # 你的 Key
client = ZhipuAI(api_key=API_KEY)
TOOL_BACKEND_URL = "http://127.0.0.1:8000"

# 统一状态管理
SESSION_STATE = {
    "history": [
        {
            "role": "system", 
            "content": (
                "你是一个专业的眼底影像科AI多智能体中枢。根据用户的意图灵活调度工具：\n"
                "1. 【仅分割】：调用 vision_segmentation_tool，拿到结果后总结，不要诊断。\n"
                "2. 【诊断分析】：调用 vision 获取特征 -> 调用 disease_diagnosis 获取疾病标签。\n"
                "3. 【医学咨询/出具方案】：当得知疾病结果后并主动要求咨询报告，或者用户主动询问医学知识、治疗方案时，你【必须】调用 medical_knowledge_retrieval_tool 查阅指南。\n"
                "4. 【异常拦截】：若工具返回“系统拦截”，终止流程并解释。\n"
                "5. ⚠️【严格控制】：只有用户明确要咨询方案，或者明确要诊断报告之类的话，才调用 medical_knowledge_retrieval_tool，否则诊断完直接给出结果即可。\n"
                "回复时使用专业、温和的自然语言，严禁输出代码块。"
            )
        }
    ],
    "context": {"rv_mask_base64": None, "faz_mask_base64": None, "metrics": None, "scan_type": None}
}

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
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "需要查阅的具体医学问题或疾病名称"}
                },
                "required": ["query"]
            }
        }
    }
]

@app.post("/api/chat")
async def chat_with_agent(text: str = Form(""), file: UploadFile = File(None)):
    user_prompt = text.strip()
    image_path = None
    file_info_msg = ""

    if file:
        os.makedirs("temp_uploads", exist_ok=True)
        image_path = os.path.abspath(f"temp_uploads/{file.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_info_msg = f"\n[系统：用户上传了图片 {image_path}。请严格根据意图调用工具。]"
        
        # 💡 优化 1：去掉强制出建议的潜台词，把选择权还给大模型
        if not user_prompt:
            user_prompt = "请帮我分析并诊断这张图。" 

    final_user_content = user_prompt + file_info_msg
    SESSION_STATE["history"].append({"role": "user", "content": final_user_content})

    # 💡 优化 2：滑动窗口记忆，防止上下文爆炸导致越来越慢
    if len(SESSION_STATE["history"]) > 15:
        # 永远保留第一条 System Prompt，再加上最近的 14 条对话
        SESSION_STATE["history"] = [SESSION_STATE["history"][0]] + SESSION_STATE["history"][-14:]

    async def event_generator():
        yield f"data: {json.dumps({'type': 'thinking', 'msg': 'Node 1: 正在识别意图与规划工作流...'})}\n\n"
        await asyncio.sleep(0.1) 

        max_loops = 4 
        current_loop = 0
        final_text = "分析完毕。"

        while current_loop < max_loops:
            current_loop += 1
            
            # 使用大模型推荐的 glm-4.7-flash
            response = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=SESSION_STATE["history"],
                tools=tools,
                temperature=0.1
            )
            msg = response.choices[0].message
            SESSION_STATE["history"].append(msg.model_dump())

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
                        # 💡 优化 3：使用 httpx 异步发送图片请求，不阻塞主线程
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
                                SESSION_STATE["context"]["rv_mask_base64"] = data["visualizations"]["rv_mask_base64"]
                                SESSION_STATE["context"]["faz_mask_base64"] = data["visualizations"]["faz_mask_base64"]
                                SESSION_STATE["context"]["metrics"] = data["metrics"]
                                SESSION_STATE["context"]["scan_type"] = data["image_metadata"]["scan_type"]
                                
                                tool_result = f"视觉分割成功。指标：{data['metrics']}。如果需要诊断，请继续调用 disease_diagnosis_tool。"
                                yield f"data: {json.dumps({'type': 'vision_data', 'data': data})}\n\n"
                    except Exception as e:
                        tool_result = f"视觉异常: {str(e)}"

                elif func_name == "disease_diagnosis_tool":
                    yield f"data: {json.dumps({'type': 'thinking', 'msg': 'Node: 唤醒 Clinical Agent，进行病变研判 (异步)...'})}\n\n"
                    payload = {
                        "scan_type": SESSION_STATE["context"]["scan_type"],
                        "rv_mask_base64": SESSION_STATE["context"]["rv_mask_base64"],
                        "faz_mask_base64": SESSION_STATE["context"]["faz_mask_base64"],
                        "metrics": SESSION_STATE["context"]["metrics"]
                    }
                    try:
                        # 💡 优化 4：使用 httpx 异步发送分类请求
                        async with httpx.AsyncClient(timeout=60.0) as http_client:
                            res = await http_client.post(f"{TOOL_BACKEND_URL}/api/v1/agent/classify", json=payload)
                        data = res.json()
                        pred_disease = data['prediction']['label_cn']
                        
                        # 💡 优化 5：软化潜台词，让大模型自主决定要不要查资料
                        tool_result = f"诊断成功。初步判定疾病为：{pred_disease}。请回顾用户的提问意图：如果用户明确要求治疗建议或详细报告，请继续调用 medical_knowledge_retrieval_tool；如果用户只想知道有无患病，请直接输出诊断结果，结束工作流。"
                        
                        yield f"data: {json.dumps({'type': 'classify_data', 'data': data})}\n\n"
                    except Exception as e:
                        tool_result = f"诊断异常: {str(e)}"

                elif func_name == "medical_knowledge_retrieval_tool":
                    search_query = args.get("query", "眼底疾病")
                    yield f"data: {json.dumps({'type': 'thinking', 'msg': f'Node: 唤醒 RAG Agent，异步查阅知识库 [{search_query}]...'})}\n\n"
                    
                    try:
                        # 💡 优化 6：将耗时的本地向量检索放入异步线程池，防止前端卡死
                        docs = await asyncio.to_thread(vector_db.similarity_search, search_query, k=3)
                        retrieved_text = "\n\n".join([f"【文献参考 {i+1}】: {doc.page_content}" for i, doc in enumerate(docs)])
                        
                        tool_result = f"查阅成功。以下是本地医学文献中的权威片段，请结合上述诊断结果给出建议：\n{retrieved_text}"
                    except Exception as e:
                        tool_result = f"知识检索失败: {str(e)}"

                else:
                    tool_result = "未知工具。"

                SESSION_STATE["history"].append({"role": "tool", "content": tool_result, "tool_call_id": tool_call.id})

        yield f"data: {json.dumps({'type': 'final_text', 'text': final_text})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")