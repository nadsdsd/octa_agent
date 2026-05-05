import os
import json
import requests
from zhipuai import ZhipuAI

# --- 初始化大模型客户端 (请替换为你的真实 API Key) ---
API_KEY = "daac45d2564c035c55a4967469d6abae.nJlpVaLeA9WtFuM8"
client = ZhipuAI(api_key=API_KEY)

# 后端 FastAPI 的地址
VISION_API_URL = "http://127.0.0.1:8000/api/v1/agent/vision/analyze"
CLASSIFY_API_URL = "http://127.0.0.1:8000/api/v1/agent/classify"

# ==========================================
# 1. 核心创新：本地上下文状态机 (避免 Base64 撑爆 LLM)
# ==========================================
# 临时存储上一张处理过的图片的重型特征，LLM 只需要知道 "状态已准备好" 即可。
SESSION_CONTEXT = {
    "image_path": None,
    "scan_type": None,
    "rv_mask_base64": None,
    "faz_mask_base64": None,
    "metrics": None
}

# ==========================================
# 2. 定义 Skills (工具描述：教大模型怎么用工具)
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "vision_segmentation_tool",
            "description": "视觉感知工具：用于对OCTA眼底图像进行Mamba模型分割，提取RV（视网膜血管）和FAZ（无血管区）掩码，并计算8项血流量化指标。当用户提到'分割'、'提取特征'、'计算指标'，或上传了一张图片需要分析时，必须首先调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "本地图像文件的绝对路径，例如 /mnt/d/octa_agent/data/10301.bmp"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "disease_diagnosis_tool",
            "description": "临床诊断工具：基于视觉智能体提取的量化指标和掩码，调用多模态集成模型（Ensemble）预测眼底疾病类型及置信度。当用户要求'诊断'、'判断疾病'或问'这是什么病'时调用。注意：调用此工具前，必须确保已经对该图片调用过 vision_segmentation_tool。",
            "parameters": {
                "type": "object",
                "properties": {}, # 故意留空：因为沉重的数据我们通过本地 SESSION_CONTEXT 传递
            }
        }
    }
]

# ==========================================
# 3. 工具的具体执行逻辑 (Python 函数)
# ==========================================
def execute_vision_segmentation(image_path: str) -> str:
    """执行视觉分割，保存重特征到本地，返回轻量级文本给大模型"""
    print(f"\n[🛠️ Tool Execution] 正在调用 Mamba 视觉分割模型分析: {image_path} ...")
    if not os.path.exists(image_path):
        return f"错误：找不到图像文件 {image_path}"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/bmp")}
            response = requests.post(VISION_API_URL, files=files)
            
        if response.status_code != 200:
            return f"视觉引擎处理失败：{response.text}"
            
        data = response.json()
        
        # 将重型 Base64 数据拦截在本地字典中！
        SESSION_CONTEXT["image_path"] = image_path
        SESSION_CONTEXT["scan_type"] = data["image_metadata"]["scan_type"]
        SESSION_CONTEXT["rv_mask_base64"] = data["visualizations"]["rv_mask_base64"]
        SESSION_CONTEXT["faz_mask_base64"] = data["visualizations"]["faz_mask_base64"]
        SESSION_CONTEXT["metrics"] = data["metrics"]
        
        # 只把清爽的 JSON 字符串返回给大模型
        summary = {
            "msg": "图像分割已完成，掩码已缓存在内存中。",
            "scan_type": data["image_metadata"]["scan_type"],
            "metrics": data["metrics"]
        }
        return json.dumps(summary, ensure_ascii=False)
        
    except Exception as e:
        return f"视觉引擎请求异常: {str(e)}"


def execute_disease_diagnosis() -> str:
    """执行疾病诊断，直接从内存中读取分割好的掩码和指标"""
    print(f"\n[🛠️ Tool Execution] 正在调用 RVMamba 多模态集成模型进行诊断 ...")
    
    if not SESSION_CONTEXT["rv_mask_base64"] or not SESSION_CONTEXT["metrics"]:
        return "错误：内存中缺少视觉特征数据。请先告诉用户需要先执行图像分割工具。"

    payload = {
        "scan_type": SESSION_CONTEXT["scan_type"],
        "rv_mask_base64": SESSION_CONTEXT["rv_mask_base64"],
        "faz_mask_base64": SESSION_CONTEXT["faz_mask_base64"],
        "metrics": SESSION_CONTEXT["metrics"]
    }

    try:
        response = requests.post(CLASSIFY_API_URL, json=payload)
        if response.status_code != 200:
            return f"诊断引擎处理失败：{response.text}"
            
        data = response.json()
        prediction = data["prediction"]
        
        # 整理结果返回给大模型，让它润色
        result_str = (
            f"诊断完成！预测疾病: {prediction['label_cn']} ({prediction['label_en']}), "
            f"综合置信度: {prediction['confidence']:.2%}\n"
            f"概率分布详情: {json.dumps(prediction['distribution'], ensure_ascii=False)}"
        )
        return result_str
        
    except Exception as e:
        return f"诊断引擎请求异常: {str(e)}"

# ==========================================
# 4. 主控循环 (The Agent Loop)
# ==========================================
def chat_with_agent():
    print("==================================================")
    print(" 🤖 多智能体中枢已启动 (Type 'quit' to exit)")
    print("==================================================")
    
    messages = [
        {"role": "system", "content": "你是一个专业的眼底影像科AI助手。你可以使用工具来分割OCTA图片并给出疾病诊断。请用专业、温和的医学口吻回复用户。如果用户提供的路径不完整，可以追问。"}
    ]
    
    while True:
        user_input = input("\n🧑‍⚕️ 用户: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # 第一轮：询问大模型，看看它是否想用工具
        response = client.chat.completions.create(
            model="glm-4.7-flash", # 推荐使用标准的 glm-4 支持函数调用
            messages=messages,
            tools=tools,
            temperature=0.1 # 工具调用时温度要低一点，保证准确性
        )
        
        response_msg = response.choices[0].message
        messages.append(response_msg.model_dump()) # 把大模型的思考加入历史

        # 检查大模型是否决定调用工具
        if response_msg.tool_calls:
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                # 执行对应的本地函数
                if func_name == "vision_segmentation_tool":
                    tool_result = execute_vision_segmentation(func_args.get("image_path"))
                elif func_name == "disease_diagnosis_tool":
                    tool_result = execute_disease_diagnosis()
                else:
                    tool_result = f"Unknown tool: {func_name}"
                    
                # 将工具的执行结果追加到对话中
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
                
            # 第二轮：带着工具返回的结果，再次询问大模型，生成最终回复
            final_response = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                temperature=0.7
            )
            final_content = final_response.choices[0].message.content
            print(f"\n🤖 Agent: {final_content}")
            messages.append({"role": "assistant", "content": final_content})
            
        else:
            # 如果不需要调用工具（纯聊天），直接打印结果
            print(f"\n🤖 Agent: {response_msg.content}")

if __name__ == "__main__":
    chat_with_agent()