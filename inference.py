"""
RAG Inference Engine (Hybrid Search Version)
============================================
Engine này thực hiện tìm kiếm "Lai" (Hybrid):
1. Tìm bằng Dense Vector (Hiểu ngữ nghĩa).
2. Tìm bằng Sparse Vector (Bắt từ khóa chính xác - BM25).
3. Dùng thuật toán RRF (Reciprocal Rank Fusion) để trộn kết quả lại.
"""

import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch

# Import Config
from config import Config

# Import FastEmbed cho Sparse
try:
    from fastembed import SparseTextEmbedding
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    print("⚠️ Chưa cài 'fastembed'. Chế độ Hybrid sẽ bị tắt.")

class RAGPipeline:
    def __init__(self):
        print("⏳ Đang khởi tạo Search Engine...")
        
        # --- 1. LOAD DENSE MODEL (Ngữ nghĩa) ---
        # Model này BẮT BUỘC phải khớp với model lúc Indexing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Device: {device.upper()}")
        
        try:
            self.dense_model = SentenceTransformer(Config.MODEL_PATH, device=device)
            print(f"   - Dense Model: {Config.MODEL_PATH}")
        except:
            print(f"   - ⚠️ Fallback Dense: {Config.DUMMY_MODEL_NAME}")
            self.dense_model = SentenceTransformer(Config.DUMMY_MODEL_NAME, device=device)

        # --- 2. LOAD SPARSE MODEL (Từ khóa) ---
        self.sparse_model = None
        if SPARSE_AVAILABLE and Config.SPARSE_AVAILABLE:
            print("   - Loading Sparse Model (BM25)...")
            # Lưu ý: Phải dùng đúng tên model đã dùng lúc Indexing ("Qdrant/bm25")
            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        # --- 3. KẾT NỐI QDRANT ---
        try:
            if Config.USE_QDRANT_CLOUD:
                self.qdrant = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
            else:
                self.qdrant = QdrantClient(url=Config.QDRANT_URL)
            print("✅ Kết nối Database thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối Qdrant: {e}")

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Dict]:
        t0 = time.time()
        
        # A. Tạo Dense Vector
        dense_vector = self.dense_model.encode(query, normalize_embeddings=True).tolist()
        
        # B. Tạo Sparse Vector
        sparse_vector = None
        if self.sparse_model:
            sparse_output = list(self.sparse_model.embed(query))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_output.indices.tolist(),
                values=sparse_output.values.tolist()
            )

        # C. Prefetch
        prefetch_requests = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=top_k * 2
            )
        ]
        
        if sparse_vector:
            prefetch_requests.append(
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2
                )
            )

        # D. Execute Search (ĐÃ SỬA LỖI Ở ĐÂY)
        search_result = self.qdrant.query_points(
            collection_name=Config.COLLECTION_NAME,
            prefetch=prefetch_requests,
            # Sửa tham số 'method' thành 'fusion'
            query=models.FusionQuery(fusion=models.Fusion.RRF), 
            limit=top_k,
            with_payload=True
        )
        
        # E. Format
        results = []
        for point in search_result.points:
            results.append({
                "score": point.score,
                "title": point.payload.get("title", "No Title"),
                "text": point.payload.get("text", ""),
                "url": point.payload.get("url", "")
            })
            
        print(f"🔍 Tìm thấy {len(results)} kết quả trong {time.time()-t0:.3f}s")
        return results

    def run_test(self, query: str):
        """Hàm test nhanh để xem kết quả"""
        print(f"\n❓ Câu hỏi: {query}")
        docs = self.retrieve_hybrid(query, top_k=3)
        
        print("--- KẾT QUẢ TÌM KIẾM (Top 3) ---")
        for i, d in enumerate(docs):
            print(f"[{i+1}] (Score: {d['score']:.4f}) {d['title']}")
            print(f"    Nội dung: {d['text']}") # In 150 ký tự đầu
            print("-" * 30)

# ===========================
# CHẠY THỬ
# ===========================
if __name__ == "__main__":
    engine = RAGPipeline()
    
    # Test 1: Câu hỏi cần ngữ nghĩa (Dense giỏi)
    engine.run_test("Ý nghĩa của chiến thắng Điện Biên Phủ?")
    
    # Test 2: Câu hỏi cần từ khóa chính xác (Sparse giỏi)
    # Thử hỏi về một tên riêng hoặc số liệu cụ thể trong dữ liệu của bạn
    engine.run_test("Triều đại nhà Nguyễn từ năm nào?")
