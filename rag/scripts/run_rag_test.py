# import os
# import sys
# import re
# import csv
# import collections
# import time
# import numpy as np

# # 尝试导入您的 RAG 系统
# # 确保 rag_dm.py 和 glm.py 在当前目录下
# try:
#     from rag_dm import build_rag_chain
# except ImportError:
#     print("错误: 未找到 rag_dm.py 或其依赖 (glm.py)。请确保脚本在正确的目录下运行。")
#     sys.exit(1)

# def parse_test_file(filepath):
#     """
#     解析 test.txt 文件，提取问题和标准答案 (Ground Truth)
#     """
#     questions = []
#     current_q = {}
    
#     with open(filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     # 使用正则匹配 Q1: ... 和 * **答案:** ...
#     # 假设格式为: "**Q\d+: (.*?)**" 和 "* **答案:** (.*?)"
#     q_pattern = re.compile(r'\*\*Q\d+:\s*(.*?)\*\*')
#     a_pattern = re.compile(r'\*\s*\*\*答案:\*\*\s*(.*)')
    
#     lines = content.split('\n')
#     for line in lines:
#         line = line.strip()
#         q_match = q_match = q_pattern.search(line)
#         a_match = a_pattern.search(line)
        
#         if q_match:
#             if 'question' in current_q: # 保存上一个问题
#                 questions.append(current_q)
#                 current_q = {}
#             current_q['question'] = q_match.group(1).strip()
        
#         elif a_match:
#             if 'question' in current_q:
#                 current_q['ground_truth'] = a_match.group(1).strip()
#                 questions.append(current_q)
#                 current_q = {}
    
#     # 添加最后一个（如果存在）
#     if 'question' in current_q and 'ground_truth' in current_q:
#         questions.append(current_q)
        
#     print(f"成功加载 {len(questions)} 个测试问题。")
#     return questions

# def normalize_text(text):
#     """简单的文本标准化: 转小写，去除标点"""
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text) # 去除标点
#     return text.split()

# def compute_f1(prediction, ground_truth):
#     """
#     计算基于词符 (Token) 的 F1 分数
#     用于衡量生成答案的准确性 (Accuracy Proxy)
#     """
#     pred_tokens = normalize_text(prediction)
#     truth_tokens = normalize_text(ground_truth)
    
#     common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
#     num_same = sum(common.values())
    
#     if len(pred_tokens) == 0 or len(truth_tokens) == 0:
#         return int(pred_tokens == truth_tokens)
    
#     if num_same == 0:
#         return 0
    
#     precision = 1.0 * num_same / len(pred_tokens)
#     recall = 1.0 * num_same / len(truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

# def compute_retrieval_hit(source_docs, ground_truth):
#     """
#     计算检索命中率 (Retrieval Recall / Hit Rate)
#     检查标准答案中的关键词是否出现在检索到的文档内容中
#     """
#     if not source_docs:
#         return 0
    
#     truth_tokens = set(normalize_text(ground_truth))
#     # 移除常见停用词以减少误报（简化版）
#     stopwords = {'的', '了', '是', '在', '和', '与', 'the', 'a', 'in', 'of', 'and'}
#     truth_tokens = truth_tokens - stopwords
    
#     if not truth_tokens:
#         return 0

#     full_context = " ".join([doc.page_content.lower() for doc in source_docs])
    
#     # 计算有多少个标准答案的关键词出现在了上下文中
#     hits = 0
#     for token in truth_tokens:
#         if token in full_context:
#             hits += 1
            
#     # 如果超过 50% 的关键词被检索到，视为“命中”
#     overlap_ratio = hits / len(truth_tokens)
#     return 1 if overlap_ratio > 0.5 else 0

# def run_evaluation():
#     # 1. 准备路径
#     pdf_dir = "rag文献库"  # 您的 PDF 文件夹路径
#     test_file = "test.txt"
#     output_file = "rag_test_results.csv"
    
#     if not os.path.exists(test_file):
#         print(f"未找到测试文件: {test_file}")
#         return

#     # 2. 初始化 RAG 系统
#     print("正在初始化 RAG 系统...")
#     # 注意：这需要 glm.py 和 API Key 配置正确
#     qa_chain = build_rag_chain(pdf_dir)
#     if qa_chain is None:
#         print("RAG 系统初始化失败。请检查目录和文件。")
#         return

#     # 3. 加载问题
#     test_data = parse_test_file(test_file)
    
#     results = []
#     total_f1 = 0
#     total_hit_rate = 0
    
#     print("\n=== 开始测试 (共 {} 题) ===".format(len(test_data)))
    
#     for i, item in enumerate(test_data):
#         q = item['question']
#         gt = item['ground_truth']
        
#         print(f"\n正在测试 Q{i+1}: {q[:30]}...")
        
#         try:
#             # 调用 RAG
#             # 注意：chat_history 设为空，因为我们测试单轮问答性能
#             response = qa_chain.invoke({"question": q, "chat_history": []})
            
#             answer = response.get('answer', "无回答")
#             source_docs = response.get('source_documents', [])
            
#             # 计算指标
#             f1_score = compute_f1(answer, gt)
#             hit = compute_retrieval_hit(source_docs, gt)
            
#             total_f1 += f1_score
#             total_hit_rate += hit
            
#             results.append({
#                 "id": i+1,
#                 "question": q,
#                 "ground_truth": gt,
#                 "prediction": answer,
#                 "f1_score": round(f1_score, 4),
#                 "retrieval_hit": hit,
#                 "sources": [os.path.basename(d.metadata.get('source', '')) for d in source_docs]
#             })
            
#             print(f"  -> F1 Score: {f1_score:.4f} | Retrieval Hit: {hit}")
            
#         except Exception as e:
#             print(f"  -> 发生错误: {e}")
#             results.append({
#                 "id": i+1,
#                 "question": q,
#                 "ground_truth": gt,
#                 "prediction": f"ERROR: {e}",
#                 "f1_score": 0,
#                 "retrieval_hit": 0,
#                 "sources": []
#             })

#     # 4. 计算平均指标
#     avg_f1 = total_f1 / len(test_data) if test_data else 0
#     avg_hit = total_hit_rate / len(test_data) if test_data else 0
    
#     print("\n" + "="*40)
#     print("测试总结报告")
#     print("="*40)
#     print(f"测试问题总数: {len(test_data)}")
#     print(f"平均 Accuracy (F1-Score): {avg_f1:.4f}")
#     print(f"平均 Retrieval Recall (Hit Rate): {avg_hit:.4f}")
#     print(f"注: AUC 无法在无排序评分的情况下计算，已用 Hit Rate 代替检索效能评估。")
#     print("="*40)
    
#     # 5. 保存结果
#     with open(output_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=["id", "question", "ground_truth", "prediction", "f1_score", "retrieval_hit", "sources"])
#         writer.writeheader()
#         writer.writerows(results)
    
#     print(f"\n详细结果已保存至: {output_file}")

# if __name__ == "__main__":
#     run_evaluation()
import os
import sys
import re
import csv
import collections
import time
import numpy as np

# 尝试导入您的 RAG 系统
# 确保 rag_dm.py 和 glm.py 在当前目录下
try:
    from rag_dm import build_rag_chain
except ImportError:
    print("错误: 未找到 rag_dm.py 或其依赖 (glm.py)。请确保脚本在正确的目录下运行。")
    sys.exit(1)

def parse_test_file(filepath):
    """
    解析 test.txt 文件，提取问题和标准答案 (Ground Truth)
    """
    questions = []
    current_q = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则匹配 Q1: ... 和 * **答案:** ...
    # 假设格式为: "**Q\d+: (.*?)**" 和 "* **答案:** (.*?)"
    q_pattern = re.compile(r'\*\*Q\d+:\s*(.*?)\*\*')
    a_pattern = re.compile(r'\*\s*\*\*答案:\*\*\s*(.*)')
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        q_match = q_pattern.search(line)
        a_match = a_pattern.search(line)
        
        if q_match:
            if 'question' in current_q: # 保存上一个问题
                questions.append(current_q)
                current_q = {}
            current_q['question'] = q_match.group(1).strip()
        
        elif a_match:
            if 'question' in current_q:
                current_q['ground_truth'] = a_match.group(1).strip()
                questions.append(current_q)
                current_q = {}
    
    # 添加最后一个（如果存在）
    if 'question' in current_q and 'ground_truth' in current_q:
        questions.append(current_q)
        
    print(f"成功加载 {len(questions)} 个测试问题。")
    return questions

def normalize_text(text):
    """简单的文本标准化: 英文按词、中文按字切分"""
    text = text.lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)

def compute_f1(prediction, ground_truth):
    """
    计算基于词符 (Token) 的 F1 分数
    用于衡量生成答案的准确性 (Accuracy Proxy)
    """
    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(ground_truth)
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_retrieval_hit(source_docs, ground_truth):
    """
    计算检索命中率 (Retrieval Recall / Hit Rate)
    检查标准答案中的关键词是否出现在检索到的文档内容中
    """
    if not source_docs:
        return 0
    
    truth_tokens = set(normalize_text(ground_truth))
    # 移除常见停用词以减少误报（简化版）
    stopwords = {'的', '了', '是', '在', '和', '与', 'the', 'a', 'in', 'of', 'and'}
    truth_tokens = truth_tokens - stopwords
    
    if not truth_tokens:
        return 0

    full_context = " ".join([doc.page_content.lower() for doc in source_docs])
    
    # 计算有多少个标准答案的关键词出现在了上下文中
    hits = 0
    for token in truth_tokens:
        if token in full_context:
            hits += 1
            
    # 如果超过 50% 的关键词被检索到，视为“命中”
    overlap_ratio = hits / len(truth_tokens)
    return 1 if overlap_ratio > 0.5 else 0

def run_evaluation():
    # 1. 准备路径
    pdf_dir = "rag文献库"  # 您的 PDF 文件夹路径
    test_file = "test.txt"
    output_file = "rag_test_results.csv"
    
    if not os.path.exists(test_file):
        print(f"未找到测试文件: {test_file}")
        return

    # 2. 初始化 RAG 系统
    print("正在初始化 RAG 系统...")
    # 注意：这需要 glm.py 和 API Key 配置正确
    use_history = False
    qa_chain = build_rag_chain(pdf_dir, use_history=use_history)
    if qa_chain is None:
        print("RAG 系统初始化失败。请检查目录和文件。")
        return

    # 3. 加载问题
    test_data = parse_test_file(test_file)
    
    results = []
    total_f1 = 0
    total_hit_rate = 0
    
    print("\n=== 开始测试 (共 {} 题) ===".format(len(test_data)))
    
    for i, item in enumerate(test_data):
        q = item['question']
        gt = item['ground_truth']
        
        print(f"\n正在测试 Q{i+1}: {q[:30]}...")
        
        try:
            # 调用 RAG
            # 注意：chat_history 设为空，因为我们测试单轮问答性能
            # --- 修改点：在这里强制添加中文指令 ---
            q_with_instruction = q + " 中文回答"
            
            input_keys = getattr(qa_chain, "input_keys", None)
            if input_keys:
                payload = {}
                if "question" in input_keys:
                    payload["question"] = q_with_instruction
                elif "query" in input_keys:
                    payload["query"] = q_with_instruction
                else:
                    payload[list(input_keys)[0]] = q_with_instruction
                if "chat_history" in input_keys:
                    payload["chat_history"] = []
                response = qa_chain.invoke(payload)
            else:
                if use_history:
                    response = qa_chain.invoke({"question": q_with_instruction, "chat_history": []})
                else:
                    response = qa_chain.invoke({"query": q_with_instruction})
            
            answer = (
                response.get('answer')
                or response.get('result')
                or response.get('output_text')
                or "无回答"
            )
            source_docs = response.get('source_documents', [])
            
            # 计算指标
            f1_score = compute_f1(answer, gt)
            hit = compute_retrieval_hit(source_docs, gt)
            
            total_f1 += f1_score
            total_hit_rate += hit
            
            results.append({
                "id": i+1,
                "question": q,
                "ground_truth": gt,
                "prediction": answer,
                "f1_score": round(f1_score, 4),
                "retrieval_hit": hit,
                "sources": [os.path.basename(d.metadata.get('source', '')) for d in source_docs]
            })
            
            print(f"  -> F1 Score: {f1_score:.4f} | Retrieval Hit: {hit}")
            
        except Exception as e:
            print(f"  -> 发生错误: {e}")
            results.append({
                "id": i+1,
                "question": q,
                "ground_truth": gt,
                "prediction": f"ERROR: {e}",
                "f1_score": 0,
                "retrieval_hit": 0,
                "sources": []
            })

    # 4. 计算平均指标
    avg_f1 = total_f1 / len(test_data) if test_data else 0
    avg_hit = total_hit_rate / len(test_data) if test_data else 0
    
    print("\n" + "="*40)
    print("测试总结报告")
    print("="*40)
    print(f"测试问题总数: {len(test_data)}")
    print(f"平均 Accuracy (F1-Score): {avg_f1:.4f}")
    print(f"平均 Retrieval Recall (Hit Rate): {avg_hit:.4f}")
    print(f"注: AUC 无法在无排序评分的情况下计算，已用 Hit Rate 代替检索效能评估。")
    print("="*40)
    
    # 5. 保存结果
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "ground_truth", "prediction", "f1_score", "retrieval_hit", "sources"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n详细结果已保存至: {output_file}")

if __name__ == "__main__":
    run_evaluation()
