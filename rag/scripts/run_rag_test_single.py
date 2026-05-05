import collections
import csv
import os
import re

from rag_dm import build_rag_chain


def parse_test_file(filepath):
    questions = []
    current_q = {}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    q_pattern = re.compile(r"\*\*Q\d+:\s*(.*?)\*\*")
    a_pattern = re.compile(r"\*\s*\*\*\u7b54\u6848:\*\*\s*(.*)")

    for line in content.split("\n"):
        line = line.strip()
        q_match = q_pattern.search(line)
        a_match = a_pattern.search(line)

        if q_match:
            if "question" in current_q:
                questions.append(current_q)
                current_q = {}
            current_q["question"] = q_match.group(1).strip()
        elif a_match:
            if "question" in current_q:
                current_q["ground_truth"] = a_match.group(1).strip()
                questions.append(current_q)
                current_q = {}

    if "question" in current_q and "ground_truth" in current_q:
        questions.append(current_q)

    print(f"\u6210\u529f\u52a0\u8f7d {len(questions)} \u4e2a\u6d4b\u8bd5\u95ee\u9898\u3002")
    return questions


def normalize_text(text):
    text = text.lower()
    # Tokenize English by words and Chinese by single characters for fairer F1.
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(ground_truth)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_retrieval_hit(source_docs, ground_truth):
    if not source_docs:
        return 0

    truth_tokens = set(normalize_text(ground_truth))
    stopwords = {
        "\u7684",
        "\u662f",
        "\u5728",
        "\u548c",
        "\u4e0e",
        "\u4e0d",
        "the",
        "a",
        "in",
        "of",
        "and",
    }
    truth_tokens = truth_tokens - stopwords

    if not truth_tokens:
        return 0

    full_context = " ".join([doc.page_content.lower() for doc in source_docs])
    hits = sum(1 for token in truth_tokens if token in full_context)
    return 1 if (hits / len(truth_tokens)) > 0.5 else 0


def run_evaluation():
    pdf_dir = "/mnt/d/rag/rag\u6587\u732e\u5e93"
    test_file = "test.txt"
    output_file = "rag_test_results.csv"

    if not os.path.exists(test_file):
        print(f"\u672a\u627e\u5230\u6d4b\u8bd5\u6587\u4ef6: {test_file}")
        return

    print("\u6b63\u5728\u521d\u59cb\u5316 RAG \u7cfb\u7edf...")
    # Use the same chain type as app.py, but with empty history (single-turn).
    qa_chain = build_rag_chain(pdf_dir, use_history=True)
    if qa_chain is None:
        print("\u65e0\u6cd5\u521d\u59cb\u5316 RAG \u7cfb\u7edf\uff0c\u8bf7\u68c0\u67e5\u8def\u5f84\u3002")
        return

    test_data = parse_test_file(test_file)
    results = []
    total_f1 = 0
    total_hit_rate = 0

    print(f"\n=== \u5f00\u59cb\u6d4b\u8bd5 (\u5171 {len(test_data)} \u9898) ===")
    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n\u6b63\u5728\u6d4b\u8bd5 Q{i+1}: {q[:30]}...")

        try:
            q_with_instruction = f"{q}。请用中文回答。"
            response = qa_chain.invoke({"question": q_with_instruction, "chat_history": []})
            answer = response.get("answer") or "\u65e0\u56de\u7b54"
            source_docs = response.get("source_documents", [])

            f1_score = compute_f1(answer, gt)
            hit = compute_retrieval_hit(source_docs, gt)

            total_f1 += f1_score
            total_hit_rate += hit

            seen_sources = set()
            sources = []
            for doc in source_docs:
                name = os.path.basename(doc.metadata.get("source", ""))
                if name and name not in seen_sources:
                    seen_sources.add(name)
                    sources.append(name)

            results.append(
                {
                    "id": i + 1,
                    "question": q,
                    "ground_truth": gt,
                    "prediction": answer,
                    "f1_score": round(f1_score, 4),
                    "retrieval_hit": hit,
                    "sources": sources,
                }
            )

            print(f"  -> F1 Score: {f1_score:.4f} | Retrieval Hit: {hit}")
        except Exception as e:
            print(f"  -> \u53d1\u751f\u9519\u8bef: {e}")
            results.append(
                {
                    "id": i + 1,
                    "question": q,
                    "ground_truth": gt,
                    "prediction": f"ERROR: {e}",
                    "f1_score": 0,
                    "retrieval_hit": 0,
                    "sources": [],
                }
            )

    avg_f1 = total_f1 / len(test_data) if test_data else 0
    avg_hit = total_hit_rate / len(test_data) if test_data else 0

    print("\n" + "=" * 40)
    print("\u6d4b\u8bd5\u603b\u7ed3\u62a5\u544a")
    print("=" * 40)
    print(f"\u6d4b\u8bd5\u95ee\u9898\u603b\u6570: {len(test_data)}")
    print(f"\u5e73\u5747 Accuracy (F1-Score): {avg_f1:.4f}")
    print(f"\u5e73\u5747 Retrieval Recall (Hit Rate): {avg_hit:.4f}")
    print(
        "\u6ce8: AUC \u65e0\u6cd5\u5728\u65e0\u6392\u5e8f\u8bc4\u5206\u7684\u60c5\u51b5\u4e0b\u8ba1\u7b97\uff0c\u5df2\u7528 Hit Rate \u4ee3\u66ff\u68c0\u7d22\u6548\u80fd\u8bc4\u4f30\u3002"
    )
    print("=" * 40)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "ground_truth",
                "prediction",
                "f1_score",
                "retrieval_hit",
                "sources",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\u8be6\u7ec6\u7ed3\u679c\u5df2\u4fdd\u5b58\u81f3: {output_file}")


if __name__ == "__main__":
    run_evaluation()
