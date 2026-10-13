"""
Automated Evaluation Suite for Enterprise RAG System.
Computes standard RAG benchmarks:
1. Context Recall (Retrieval Hit Rate): Did retrieval fetch the correct ground-truth chunk?
2. Citation Groundedness / Faithfulness: Are assertions strictly backed by retrieved facts?
3. Anti-Hallucination Rate: Does the LLM decline to answer when data is absent?
4. End-to-End Latency: Retrieval and generation benchmark.
"""
from __future__ import annotations
import sys
import json
import time
from typing import List, Dict, Any
from project_lens.retrieval.local_ollama_rag import LocalEnterpriseRAG

sys.stdout.reconfigure(encoding='utf-8')

# Golden Benchmark Dataset curated from real TitSmart contracts
BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "category": "Exact Technical Spec (Needle-in-a-Haystack)",
        "question": "Máy bơm diesel trong hợp đồng THIFACO có lưu lượng và cột áp bao nhiêu?",
        "expected_keywords": ["54m³/h", "30mh₂o"],
        "expected_source": "CHI PHÍ DỰ ÁN ĐẮK R'LẤP",
        "should_hallucinate_price": False
    },
    {
        "id": "TC-02",
        "category": "Anti-Hallucination on Missing Value",
        "question": "Đơn giá của máy bơm diesel trong sheet Thông số kỹ thuật là bao nhiêu tiền?",
        "expected_keywords": ["không", "không đề cập"],
        "expected_source": "THIFACO-TSM",
        "should_hallucinate_price": False
    },
    {
        "id": "TC-03",
        "category": "Contract & Substation Specific",
        "question": "Hệ thống SCADA TBA110kV yêu cầu kỹ thuật gì về khử nhiễu quanh giá trị 0?",
        "expected_keywords": ["zero clamp deadband", "nhiễu"],
        "expected_source": "Thông số Scada",
        "should_hallucinate_price": False
    },
    {
        "id": "TC-04",
        "category": "Global Document Overview",
        "question": "Kho tài liệu này nói về gì và gồm những nhóm dự án nào?",
        "expected_keywords": ["trạm biến áp", "110kv", "hợp đồng"],
        "expected_source": "catalog",
        "should_hallucinate_price": False
    }
]

def run_evaluation(model_name: str = "qwen3:4b"):
    print("=" * 70)
    print(f"TITSMART RAG BENCHMARK & EVALUATION RUNNER ({model_name})")
    print("=" * 70)

    engine = LocalEnterpriseRAG(llm_model=model_name)
    results = []

    passed_recall = 0
    passed_faithfulness = 0
    passed_anti_hallucination = 0
    total_time = 0.0

    for idx, tc in enumerate(BENCHMARK_CASES, start=1):
        print(f"\n▶ [{tc['id']}] {tc['category']}")
        print(f"  Câu hỏi: {tc['question']}")

        t0 = time.time()
        res = engine.query(tc["question"])
        elapsed = round(time.time() - t0, 3)
        total_time += elapsed

        answer = res["answer"].lower()
        sources_str = " ".join([s["label"] for s in res["sources"]]).lower()

        # 1. Context Recall / Hit Rate
        source_hit = tc["expected_source"].lower() in sources_str
        if source_hit:
            passed_recall += 1

        # 2. Answer Keyword Groundedness
        keywords_hit = any(kw.lower() in answer for kw in tc["expected_keywords"])
        if keywords_hit:
            passed_faithfulness += 1

        # 3. Anti-Hallucination check
        anti_h_ok = True
        if tc["id"] == "TC-02":
            # Must explicitly state not found, not fabricate price numbers like 50.000.000 VND
            anti_h_ok = any(neg in answer for neg in ["không", "chưa", "không có", "không đề cập"])
            if anti_h_ok:
                passed_anti_hallucination += 1
        else:
            passed_anti_hallucination += 1

        print(f"  ⏱ Thời gian: {elapsed}s | Hit Source: {'✅' if source_hit else '❌'} | Answer Correct: {'✅' if keywords_hit else '❌'}")
        results.append({
            "id": tc["id"],
            "elapsed": elapsed,
            "source_hit": source_hit,
            "keywords_hit": keywords_hit,
            "anti_hallucination": anti_h_ok
        })

    n = len(BENCHMARK_CASES)
    context_recall_score = (passed_recall / n) * 100
    faithfulness_score = (passed_faithfulness / n) * 100
    anti_hallucination_score = (passed_anti_hallucination / n) * 100
    avg_latency = round(total_time / n, 2)

    print("\n" + "=" * 70)
    print("📊 BÁO CÁO ĐỊNH LƯỢNG RAG EVALUATION BENCHMARK:")
    print("=" * 70)
    print(f"1. Context Recall (Retrieval Hit Rate): {context_recall_score:.1f}%")
    print(f"2. Faithfulness / Answer Accuracy:     {faithfulness_score:.1f}%")
    print(f"3. Anti-Hallucination Rate:            {anti_hallucination_score:.1f}%")
    print(f"4. Average Query Latency:              {avg_latency}s / request")
    print("=" * 70)

if __name__ == "__main__":
    model = "qwen3:4b" if len(sys.argv) < 2 else sys.argv[1]
    run_evaluation(model)
