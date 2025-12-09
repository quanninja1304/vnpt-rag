import time
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
import torch
from config import Config

class AdvancedRetriever:
    def __init__(self):
        print("🚀 Initializing Advanced Retriever...")
        
        # 1. Qdrant
        self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        
        # 2. Embedding Model (Dense) - FALLBACK LOGIC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.dense_model = SentenceTransformer(Config.MODEL_PATH, device=device)
        except:
            print(f"⚠️ Dense Fallback: {Config.DUMMY_MODEL_NAME}")
            self.dense_model = SentenceTransformer(Config.DUMMY_MODEL_NAME, device=device)
            
        # 3. Sparse Model (BM25)
        self.sparse_model = None
        if getattr(Config, 'SPARSE_AVAILABLE', True):
            try:
                self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            except Exception as e:
                print(f"⚠️ Sparse load failed: {e}")

        # 4. RE-RANKER (QUAN TRỌNG NHẤT)
        # Model này sẽ chấm điểm lại sự phù hợp giữa Query và Document
        print("⏳ Loading Re-ranker (BAAI/bge-reranker-v2-m3)...")
        try:
            # Model này hỗ trợ tiếng Việt rất tốt và Multilingual
            self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)
            print("✅ Re-ranker loaded.")
        except Exception as e:
            print(f"⚠️ Re-ranker load failed: {e}. Downloading fallback...")
            # Fallback model nhẹ hơn nếu model trên lỗi
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    def search(self, query: str, top_k: int = 5):
        """
        Quy trình: Hybrid Search (Lấy 30) -> Re-ranking (Lọc lấy top_k)
        """
        start_time = time.time()
        
        # A. Hybrid Search (Lấy rộng - Recall phase)
        # ==========================================
        dense_vec = self.dense_model.encode(query, normalize_embeddings=True).tolist()
        
        sparse_vec = None
        if self.sparse_model:
            sparse_res = list(self.sparse_model.embed([query]))[0]
            sparse_vec = models.SparseVector(
                indices=sparse_res.indices.tolist(),
                values=sparse_res.values.tolist()
            )

        prefetch = [models.Prefetch(query=dense_vec, using="dense", limit=30)] # Lấy 30 ứng viên
        if sparse_vec:
            prefetch.append(models.Prefetch(query=sparse_vec, using="sparse", limit=30))

        # Tìm kiếm sơ bộ từ DB
        raw_results = self.client.query_points(
            collection_name=Config.COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=30 # Lấy dư ra để lọc nhiễu
        )
        
        if not raw_results.points:
            return []

        # B. Re-ranking (Lọc tinh - Precision phase)
        # ==========================================
        # Chuẩn bị dữ liệu cho Re-ranker
        documents = []
        points_map = [] # Để map lại payload sau khi sort
        
        for point in raw_results.points:
            # Kết hợp Title + Text để model hiểu ngữ cảnh đầy đủ
            doc_content = f"{point.payload.get('title', '')}. {point.payload.get('text', '')}"
            documents.append(doc_content)
            points_map.append(point)
        
        if not documents: return []

        # Re-ranker chấm điểm từng cặp (Query, Document)
        # predict trả về array điểm số (score càng cao càng liên quan)
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.reranker.predict(pairs)

        # Gán điểm mới và sắp xếp lại
        final_results = []
        for idx, score in enumerate(rerank_scores):
            point = points_map[idx]
            final_results.append({
                "score": float(score), # Điểm Re-ranker (quan trọng hơn điểm cũ)
                "title": point.payload.get('title'),
                "text": point.payload.get('text'),
                "category": point.payload.get('category'),
                "url": point.payload.get('url'),
                "initial_score": point.score # Điểm cũ để tham khảo
            })
        
        # Sort giảm dần theo điểm Re-ranker
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Cắt lấy Top K tốt nhất
        return final_results[:top_k]

# --- TEST ---
if __name__ == "__main__":
    retriever = AdvancedRetriever()
    
    # Test câu hỏi gây nhiễu
    q = "Chiến dịch Điện Biên Phủ diễn ra vào năm nào?"
    print(f"\n❓ Câu hỏi: {q}")
    
    results = retriever.search(q, top_k=3)
    
    for i, r in enumerate(results):
        print(f"\n--- Rank {i+1} (Re-rank Score: {r['score']:.4f}) ---")
        print(f"📚 {r['title']}")
        print(f"📝 {r['text']}")