# import requests

# url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# payload = {
#     "model": "charglm-4",
#     "messages": [
#         {
#             "role": "system",
#             "content": "你是一个有用的AI助手。"
#         },
#         {
#             "role": "user",
#             "content": "请介绍一下人工智能的发展历程。"
#         }
#     ],
#     "temperature": 0.8,
#     "max_tokens": 96,
#     "stream": False,
#     "thinking": { "type": "enabled" },
#     "do_sample": True,
#     "top_p": 0.6,
#     "tool_stream": False,
#     "response_format": { "type": "text" }
# }
# headers = {
#     "Authorization": "Bearer 9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M",
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.text)
from zhipuai import ZhipuAI

# --- 请务必替换为您的真实 API Key ---
API_KEY = "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M" 

client = ZhipuAI(api_key=API_KEY)

def get_embeddings(text):
    """调用智谱 embedding-2 模型"""
    response = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return response.data[0].embedding

def query_model(prompt):
    """调用智谱 glm-4 模型"""
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content