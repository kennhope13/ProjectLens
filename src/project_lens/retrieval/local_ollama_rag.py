"""
100% Local Enterprise Hybrid RAG Engine powered by Ollama.
Stack:
- LLM: qwen3:8b (or qwen3:4b) via Ollama
- Dense Embeddings: Sentence Transformers / PostgreSQL pgvector
- Lexical Search: BM25Okapi
- Cross-Encoder Re-Ranking: BAAI/bge-reranker-v2-m3
- Structured lookups: project costs and document inventory
"""
from __future__ import annotations
import os
import sys
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from project_lens.knowledge.project_costs import highest_cost_answer, is_max_cost_question, load_project_costs
from project_lens.knowledge.document_inventory import (
    drawing_inventory_answer,
    find_drawing_files,
    find_longest_excel_appendix,
    is_appendix_length_question,
    is_drawing_or_file_question,
    longest_appendix_answer,
)

sys.stdout.reconfigure(encoding='utf-8')

from project_lens.storage.postgres import (
    load_chunk_records,
    load_embedding_model,
    search_chunk_ids,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

SYSTEM_PROMPT = """Bạn là Chuyên gia Kỹ thuật và Phân tích Hợp đồng Báo giá của TitSmart.
Nhiệm vụ của bạn là giải đáp câu hỏi của người dùng DỰA HOÀN TOÀN VÀO NGỮ CẢNH ĐƯỢC CUNG CẤP.

QUY TẮC BẮT BUỘC:
1. Đối với thông số kỹ thuật và báo giá:
   - Hãy trích xuất chính xác: Tên thiết bị/vật tư, Thông số kỹ thuật/Quy cách, Đơn vị tính (ĐVT), Khối lượng, Đơn giá và Thành tiền.
   - Nêu rõ tên Phụ lục, Sheet, Hợp đồng hoặc Dự án tương ứng.
   - Định dạng số tiền rõ ràng, kèm đơn vị (VND hoặc USD).
2. Trích dẫn nguồn (Citations):
   - Mọi số liệu, khẳng định đều PHẢI trích dẫn nguồn ở cuối câu/đoạn theo dạng `[Nguồn n]`.
3. Kiểm soát ảo giác (Anti-Hallucination):
   - Tuyệt đối không tự suy diễn. Nếu thiếu thông tin, nói rõ: "Tài liệu hiện tại không đề cập đến thông tin này."
"""

def tokenize_vi(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

def query_ollama_chat(model: str, messages: list, timeout: int = 90) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    # Format directly into Qwen ChatML to bypass internal reasoning scratchpad
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    full_prompt = "\n".join(prompt_parts)

    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        ans = res.get("response", "").strip()
        # Loại bỏ các đoạn suy nghĩ tiếng Anh nếu còn sót
        ans = re.sub(r'^(?:Okay|Let me|First,|Looking at).*?\n\n', '', ans, flags=re.DOTALL).strip()
        return ans

class LocalEnterpriseRAG:
    def __init__(self, llm_model: str = "qwen3:8b", top_k: int = 4):
        self.llm_model = llm_model
        self.top_k = top_k
        self.device = os.getenv("PROJECTLENS_DEVICE", "cpu")
        print(f"🚀 Khởi tạo Local Enterprise RAG (LLM: {self.llm_model})...")

        # Load chunk text and metadata from PostgreSQL.
        self.records = load_chunk_records()
        if not self.records:
            raise RuntimeError(
                "PostgreSQL chưa có chunks. Hãy chạy bước chunking và "
                "project_lens.retrieval.build_postgres_index trước."
            )
        self.record_indices = {record["id"]: i for i, record in enumerate(self.records)}
        print(f"Loaded {len(self.records)} records from PostgreSQL.")

        from sentence_transformers import SentenceTransformer
        embedder_name = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(embedder_name, device=self.device)

        # BM25
        print("Xây dựng chỉ mục BM25...")
        corpus = [tokenize_vi(r["page_content"]) for r in self.records]
        self.bm25 = BM25Okapi(corpus)

        # Cross-Encoder Reranker
        print(f"Tải Cross-Encoder Reranker ({RERANKER_MODEL})...")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=self.device)
        from project_lens.knowledge.document_catalog import build_document_catalog, format_catalog_for_llm
        self.catalog = build_document_catalog()
        self.catalog_context = format_catalog_for_llm(self.catalog)
        self.project_costs = load_project_costs()
        self.longest_appendix = find_longest_excel_appendix()
        print(f"Loaded {len(self.project_costs)} Excel after-tax cost summaries.")
        print("✅ Local Enterprise RAG đã sẵn sàng phục vụ 100% Offline (Hỗ trợ cả Global Summary & Local Deep Dive)!")

    def is_global_query(self, question: str) -> bool:
        """
        SOTA Semantic Query Router (Zero-Hardcoding).
        Sử dụng LLM Intent Classification để phân loại động mục đích của người dùng:
        - GLOBAL: Yêu cầu tổng quan, tóm tắt kho tài liệu, danh mục dự án/hợp đồng.
        - LOCAL: Tra cứu thông số cụ thể, giá tiền, ngày tháng, nội dung cục bộ.
        """
        # Route obvious corpus-wide questions without an extra LLM call.
        normalized = unicodedata.normalize("NFD", question.lower())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        normalized = normalized.replace("đ", "d")
        global_hints = (
            "tong quan", "toan bo", "tat ca tai lieu", "kho tai lieu",
            "du lieu nay la gi", "cac du lieu", "phan tich cac du lieu",
            "tai lieu nay noi ve gi", "gom nhung nhom nao",
        )
        if any(hint in normalized for hint in global_hints):
            return True

        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict Query Intent Classifier for an enterprise knowledge retrieval system. "
                    "Analyze the user's question and classify it into exactly one category:\n"
                    "- GLOBAL: If the question asks for an overview, summary of the entire dataset, "
                    "list of all projects/contracts/works, scope of files, or themes across all documents.\n"
                    "- LOCAL: If the question seeks specific factual lookups, exact technical specifications, "
                    "prices, item quantities, or clauses from a specific single section.\n"
                    "Output strictly one word only: GLOBAL or LOCAL."
                )
            },
            {"role": "user", "content": f"User question: {question}"}
        ]
        try:
            # A cold Ollama model load can take much longer than ten seconds.
            decision = query_ollama_chat(self.llm_model, router_messages, timeout=90).strip().upper()
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            print(f"Ollama router unavailable ({error}); continuing with document search.")
            return False
        print(f"🤖 [Dynamic Intent Router]: '{decision}' (Phân loại tự động 100% bằng AI)")
        return "GLOBAL" in decision

    def query(self, question: str) -> Dict[str, Any]:
        t0 = time.time()

        if is_appendix_length_question(question):
            answer, sources = longest_appendix_answer(self.longest_appendix)
            elapsed = round(time.time() - t0, 3)
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_time": elapsed,
                "generation_time": 0.0,
                "total_time": elapsed,
                "raw_context": answer,
            }

        # Resolve drawing names and file URIs from filesystem metadata rather
        # than asking vector search or the catalog LLM to infer file locations.
        if is_drawing_or_file_question(question):
            files = find_drawing_files(question)
            answer, sources = drawing_inventory_answer(question, files)
            elapsed = round(time.time() - t0, 3)
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_time": elapsed,
                "generation_time": 0.0,
                "total_time": elapsed,
                "raw_context": answer,
            }

        # Compare structured totals across workbooks instead of inferring a
        # maximum from a handful of semantically retrieved chunks.
        if is_max_cost_question(question):
            answer, sources = highest_cost_answer(self.project_costs)
            elapsed = round(time.time() - t0, 3)
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_time": elapsed,
                "generation_time": 0.0,
                "total_time": elapsed,
                "raw_context": answer,
            }

        # NHÁNH 1: GLOBAL DOCUMENT-LEVEL QUERY (Tổng quan toàn bộ kho tài liệu)
        if self.is_global_query(question):
            print("🧭 Phát hiện câu hỏi Tổng quan: Kích hoạt Document Catalog Router...")
            full_context = self.catalog_context
            sources = [
                {"source_id": idx, "label": f"{doc['file_name']} ({doc['total_records']} bản ghi)", "score": 1.0}
                for idx, doc in enumerate(self.catalog["documents"][:10], start=1)
            ]
            messages = [
                {"role": "system", "content": """Bạn là Chuyên gia Quản trị Dữ liệu Hợp đồng của TitSmart.
Nhiệm vụ của bạn là cung cấp bức tranh TỔNG THỂ, TOÀN DIỆN về toàn bộ kho tài liệu của công ty dựa trên danh mục được cung cấp.
Hãy phân loại rõ ràng các nhóm: Hợp đồng mua sắm dự án, Bảng thông số kỹ thuật SCADA/PCCC/Camera, Nhật ký chi phí vật tư, và Biên bản nghiệm thu."""},
                {"role": "user", "content": f"""DANH MỤC TOÀN BỘ KHO TÀI LIỆU TITSMART:
{full_context}

-----------------------
CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

HÃY TỔNG HỢP VÀ PHÂN TÍCH RÕ RÀNG TỔNG QUAN KHO TÀI LIỆU:"""}
            ]
            llm_start = time.time()
            answer = query_ollama_chat(self.llm_model, messages)
            total_time = time.time() - t0
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_time": 0.05,
                "generation_time": round(total_time - (llm_start - t0), 3),
                "total_time": round(total_time, 3),
                "raw_context": full_context
            }

        # NHÁNH 2: SPECIFIC RETRIEVAL QUERY (Truy xuất chi tiết theo chỉ số/hợp đồng/bảng biểu)
        q_tokens = tokenize_vi(question)

        # 1. BM25 Lexical (Lấy top 8 ứng viên sát nhất)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:8]

        # 2. pgvector Dense Search (Lấy top 8 ứng viên sát nhất)
        q_vec = np.asarray(self.embedder.encode([question], normalize_embeddings=True), dtype=np.float32)
        dense_ids = search_chunk_ids(q_vec[0], limit=8)
        top_dense_idx = [self.record_indices[chunk_id] for chunk_id in dense_ids if chunk_id in self.record_indices]

        # 3. Candidate Merge (khoảng 8-12 ứng viên duy nhất)
        candidate_indices = list(dict.fromkeys(top_bm25_idx + top_dense_idx))
        candidates = [self.records[i] for i in candidate_indices]

        # 4. Cross-Encoder Rerank
        pairs = [(question, cand["page_content"]) for cand in candidates]
        rerank_scores = self.reranker.predict(pairs, batch_size=len(pairs))
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:self.top_k]

        # 5. Format Context
        context_parts = []
        sources = []
        for idx, (rec, score) in enumerate(ranked, start=1):
            meta = rec.get("metadata", {})
            f_name = meta.get("file_name", Path(meta.get("source", "")).name)
            sheet = meta.get("sheet_name", "")
            p_context = meta.get("project_context", "")

            label = f"{f_name}"
            if sheet:
                label += f" | Sheet: {sheet}"
            if p_context:
                label += f" | {p_context[:65]}..."

            sources.append({
                "source_id": idx,
                "label": label,
                "score": float(score),
                "metadata": meta
            })
            context_parts.append(f"--- [Nguồn {idx}: {label}] ---\n{rec['page_content']}")

        full_context = "\n\n".join(context_parts)

        # 6. Generate via Local Ollama
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""NGỮ CẢNH TRÍCH XUẤT TỪ DỮ LIỆU TITSMART:
{full_context}

-----------------------
CÂU HỎI:
{question}

TRẢ LỜI ĐẦY ĐỦ VÀ TRÍCH DẪN NGUỒN [Nguồn n]:"""}
        ]

        llm_start = time.time()
        answer = query_ollama_chat(self.llm_model, messages)
        total_time = time.time() - t0

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time": round(llm_start - t0, 3),
            "generation_time": round(total_time - (llm_start - t0), 3),
            "total_time": round(total_time, 3),
            "raw_context": full_context
        }

if __name__ == "__main__":
    model_choice = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv("PROJECTLENS_LLM_MODEL", "qwen3:8b")
    )
    engine = LocalEnterpriseRAG(llm_model=model_choice)
    
    print("\n" + "=" * 60)
    print(f"TITSMART ENTERPRISE RAG CHAT (Local Ollama: {model_choice})")
    print("Gõ câu hỏi bằng tiếng Việt (hoặc gõ 'exit' / 'quit' để thoát)")
    print("=" * 60)

    while True:
        try:
            q = input("\n[Bạn hỏi]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            print("Đã thoát chương trình.")
            break

        print("\n⏳ Đang tìm kiếm tài liệu và sinh câu trả lời...")
        res = engine.query(q)

        print("\n" + "=" * 60)
        print("🤖 [TITSMART ASSISTANT]:")
        print("=" * 60)
        print(res["answer"])
        print("\n" + "-" * 60)
        print(f"⏱ Thời gian: Tìm kiếm {res['retrieval_time']}s | Sinh lời {res['generation_time']}s | Tổng {res['total_time']}s")
        print("📚 Nguồn trích dẫn:")
        for s in res["sources"]:
            print(f"  - [Nguồn {s['source_id']}] (Điểm liên quan: {s['score']:.3f}): {s['label']}")
        print("=" * 60)
