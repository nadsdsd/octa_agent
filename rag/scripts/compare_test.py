import glm
import time

query = "octa是什么"

print(f"问题: {query}")
print("-" * 30)

# 1. 纯 GLM-4 回答 (无 RAG)
print("[1] 纯 GLM-4 模型 (无 RAG):")
start = time.time()
response_raw = glm.query_model(query) # 直接调用 API，不查数据库
print(response_raw)
print(f"(耗时: {time.time()-start:.2f}s)")

print("-" * 30)

# 2. RAG 系统回答
# 注意：这里需要运行 rag_demo.py 才能看到效果，
# 或者您直接对比您刚才终端里的那个英文回答。
print("[2] RAG 系统 (刚才的回答):")
print("(请查看 rag_demo.py 的输出，通常它是基于特定论文的定义的，且可能是英文)")