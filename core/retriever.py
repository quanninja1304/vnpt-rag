import asyncio
import pickle
import logging
import string
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
import uuid
import pickle
import asyncio
import logging
import string
import bm25s
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from underthesea import word_tokenize

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

# Import nội bộ
from config import Config
from utils.logger import logger
from utils.text_utils import generate_uuid5
from core.llm_client import get_embedding_async, call_llm_generic

from underthesea import word_tokenize

TRANSLATOR = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

class HybridRetriever:
    """
    Hybrid Retriever V2.1 - Network Stability Optimized
    - Added Qdrant Semaphore (Throttling)
    - Added Exponential Backoff
    - Reduced Batch Size
    """
    
    VECTOR_TIMEOUT = 20.0  # Tăng lên chút cho chắc
    RETRIEVE_TIMEOUT = 10.0
    BATCH_SIZE = 20        # [FIX] Giảm batch size xuống 20 theo đề xuất
    RRF_K = 60
    
    def __init__(self, collection_name: str, top_k: int = 5, alpha_vector: float = 0.5):
        # [FIX] Cấu hình Client HTTP ổn định (Tắt HTTP2, Tắt gRPC)
        self.client = AsyncQdrantClient(
            url=Config.QDRANT_URL, 
            api_key=Config.QDRANT_API_KEY,
            timeout=30.0,
            prefer_grpc=False, 
            http2=False
        )
        
        self.collection_name = collection_name
        self.top_k = top_k
        self.alpha_vector = alpha_vector
        
        # [FIX] SEMAPHORE RIÊNG CHO QDRANT
        # Dù bên ngoài chạy 8 luồng, nhưng chỉ cho phép 3 luồng chui vào Qdrant cùng lúc
        self.qdrant_sem = asyncio.Semaphore(2) 
        
        self.bm25_retriever = None
        self.chunk_ids = [] 
        self.bm25_loaded = False
        
        self._load_bm25s()
    
    def _load_bm25s(self) -> None:
        index_dir = Config.BASE_DIR / "resources" / "bm25s_index"
        id_map_file = Config.BASE_DIR / "resources" / "bm25s_ids.pkl"
        
        if not index_dir.exists():
            logger.warning(f"❌ BM25s not found at {index_dir}")
            return

        try:
            self.bm25_retriever = bm25s.BM25.load(str(index_dir), load_corpus=False, mmap=True)
            with open(id_map_file, "rb") as f:
                self.chunk_ids = pickle.load(f)
            self.bm25_loaded = True
            logger.info(f"✓ BM25s Loaded: {len(self.chunk_ids)} docs")
        except Exception as e:
            logger.error(f"Failed to load BM25s: {e}")

    async def search(self, session, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.top_k
        
        vec_task = self._vector_search(session, query, top_k)
        bm25_task = self._bm25_search(query, top_k)
        
        (vec_hits_map, vec_scores), bm25_scores = await asyncio.gather(
            vec_task, bm25_task, return_exceptions=True
        )
        
        if isinstance(vec_hits_map, Exception):
            logger.error(f"Vector crash: {vec_hits_map}")
            vec_hits_map, vec_scores = {}, {}
        
        if isinstance(bm25_scores, Exception):
            bm25_scores = {}
        
        vec_hits_map, vec_scores = await self._fetch_missing_text_optimized(
            vec_hits_map, vec_scores, bm25_scores
        )
        
        return self._fuse_scores_rrf(vec_hits_map, vec_scores, bm25_scores)[:top_k]
    
    async def _vector_search(self, session, query: str, top_k: int) -> Tuple[Dict, Dict]:
        vec_hits_map = {}
        vec_scores = {}
        
        try:
            # 1. Lấy embedding (Thường là gọi API khác hoặc local, giữ nguyên)
            query_vec = await get_embedding_async(session, query)
            if not query_vec: 
                return {}, {}
            
            # [QUAN TRỌNG] Sử dụng Semaphore để tránh spam kết nối khi mạng đang nghẽn
            async with self.qdrant_sem:
                for attempt in range(3):
                    try:
                        # [TỐI ƯU 1] Giảm timeout mỗi lần thử để fail-fast và retry sớm
                        # [TỐI ƯU 2] Chỉ lấy những field payload thực sự cần thiết để giảm dung lượng gói tin (rất quan trọng)
                        res = await asyncio.wait_for(
                            self.client.query_points(
                                collection_name=self.collection_name,
                                query=query_vec,
                                limit=top_k, # Không lấy dư thừa top_k * 2
                                with_payload=["text", "chunk_id", "title"], # [FIX] Không dùng True, chỉ lấy field cần
                                search_params={"hnsw_ef": 64, "exact": False} # [FIX] Tăng tốc search, chấp nhận độ chính xác giảm nhẹ
                            ),
                            timeout=self.VECTOR_TIMEOUT
                        )
                        
                        for point in res.points:
                            if not point.payload: continue
                            # Dùng .get để an toàn hơn
                            cid = str(point.payload.get('chunk_id', ''))
                            if cid:
                                vec_hits_map[cid] = point.payload
                                vec_scores[cid] = float(point.score)
                        
                        # Nếu có kết quả thì thoát vòng lặp retry ngay
                        if vec_hits_map:
                            return vec_hits_map, vec_scores
                        
                        # Nếu thành công nhưng rỗng (do filter hoặc dữ liệu), không cần retry
                        break

                    except (asyncio.TimeoutError, Exception) as e:
                        # [TỐI ƯU 3] Giảm Backoff để không bị tích tụ thời gian chờ quá lâu
                        wait_time = 1.5 ** attempt 
                        if attempt < 2:
                            logger.warning(f"⚠️ Vector search retry {attempt+1}/3 (Latency: {wait_time}s)")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"❌ Vector search GAVE UP sau 3 lần thử.")

        except Exception as e:
            logger.error(f"🔥 Vector search fatal error: {e}")
        
        # Trả về kết quả (có thể rỗng) để BM25 gánh phía sau, không làm crash toàn bộ pipeline
        return vec_hits_map, vec_scores
    
    async def _bm25_search(self, query: str, top_k: int) -> Dict[str, float]:
        bm25_scores = {}
        if not self.bm25_loaded: return bm25_scores
        
        try:
            text = str(query).lower().translate(TRANSLATOR)
            tokens = word_tokenize(text)
            query_tokens = [t for t in tokens if len(t.strip()) > 0]
            
            if not query_tokens: return bm25_scores
            
            doc_indices, doc_scores = self.bm25_retriever.retrieve([query_tokens], k=top_k * 2)
            
            for i, doc_idx in enumerate(doc_indices[0]):
                score = float(doc_scores[0][i])
                if score > 0:
                    bm25_scores[self.chunk_ids[doc_idx]] = score
                    
        except Exception:
            pass
        return bm25_scores
    
    async def _fetch_missing_text_optimized(self, vec_hits_map, vec_scores, bm25_scores):
        """
        [FIX] Optimized Batch Fetching:
        - Sequential Batches (Tuần tự)
        - Smaller Batch Size (20)
        - Retry with Backoff
        """
        missing_ids = [cid for cid in bm25_scores.keys() if cid not in vec_hits_map]
        if not missing_ids: return vec_hits_map, vec_scores
        
        # Chia batch nhỏ (20 items)
        batches = [missing_ids[i:i + self.BATCH_SIZE] for i in range(0, len(missing_ids), self.BATCH_SIZE)]
        
        for batch_ids in batches:
            point_ids = [generate_uuid5(cid) for cid in batch_ids]
            
            # Dùng Semaphore để giới hạn cả việc fetch
            async with self.qdrant_sem:
                for attempt in range(3):
                    try:
                        points = await asyncio.wait_for(
                            self.client.retrieve(
                                collection_name=self.collection_name,
                                ids=point_ids,
                                with_payload=True
                            ),
                            timeout=self.RETRIEVE_TIMEOUT
                        )
                        
                        for point in points:
                            if point.payload:
                                cid = str(point.payload['chunk_id'])
                                vec_hits_map[cid] = point.payload
                                if cid not in vec_scores: vec_scores[cid] = 0.0
                        
                        break # Thành công -> Thoát
                    
                    except Exception as e:
                        wait_time = 1 * (attempt + 1)
                        # logger.warning(f"Batch fetch retry {attempt+1} in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        
                        if attempt == 2:
                            logger.error(f"Batch fetch GAVE UP: {e}")

        return vec_hits_map, vec_scores

    def _fuse_scores_rrf(self, vec_hits_map, vec_scores, bm25_scores) -> List[Dict]:
        def get_ranks(scores_dict):
            sorted_keys = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)
            return {k: i for i, k in enumerate(sorted_keys)}
            
        vec_ranks = get_ranks(vec_scores)
        bm25_ranks = get_ranks(bm25_scores)
        
        final_results = []
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        
        for cid in all_ids:
            if cid not in vec_hits_map: continue
            
            r_vec = vec_ranks.get(cid, float('inf'))
            r_bm25 = bm25_ranks.get(cid, float('inf'))
            
            score_v = 1.0 / (self.RRF_K + r_vec) if r_vec != float('inf') else 0.0
            score_b = 1.0 / (self.RRF_K + r_bm25) if r_bm25 != float('inf') else 0.0
            
            final_score = score_v * self.alpha_vector + score_b * (1 - self.alpha_vector)
            
            final_results.append({
                "chunk_id": cid,
                "text": vec_hits_map[cid].get('text', ''),
                "title": vec_hits_map[cid].get('title', ''),
                "score": final_score
            })
            
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results
    
    
async def rerank_with_small(session, question, initial_docs, top_n=8, stats=None):
    if not initial_docs: return []
    if len(initial_docs) <= top_n: return initial_docs

    # [TỐI ƯU] Chỉ lấy top 15 để rerank, tránh quá tải Model Small
    candidates = initial_docs[:15]

    # [TỐI ƯU] Cắt ngắn text preview xuống 300 ký tự (đủ để AI biết nội dung)
    docs_text = ""
    for i, doc in enumerate(candidates):
        clean_body = str(doc.get('text', '')).strip().replace("\n", " ")
        preview_text = " ".join(clean_body.split())[:300] 
        docs_text += f"ID [{i}]: {preview_text}...\n\n"

    system_prompt = """Bạn là chuyên gia lọc tin.
NHIỆM VỤ: Chọn các ID tài liệu liên quan nhất đến câu hỏi.
OUTPUT JSON: {"ids": [0, 2, ...]}"""

    user_prompt = f"""CÂU HỎI: "{question}"

DANH SÁCH:
{docs_text}

CHỌN ID LIÊN QUAN NHẤT:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # [FIX 1] Thêm timeout=30s để fail fast nếu model treo
        response = await call_llm_generic(
            session, messages, 
            Config.LLM_MODEL_SMALL, 
            stats, 
            max_tokens=100, 
            timeout=30 # Thêm tham số timeout
        )
        
        # [FIX LOGIC] Check response
        if response:
            # Tìm tất cả số trong response (chấp nhận cả format [1, 2] hoặc 1, 2)
            found_indices = [int(s) for s in re.findall(r'\d+', response)]
            
            valid_docs = []
            seen = set()
            
            # Lấy docs theo thứ tự AI chọn
            for idx in found_indices:
                if 0 <= idx < len(candidates) and idx not in seen:
                    valid_docs.append(candidates[idx])
                    seen.add(idx)
            
            # Backfill (Nếu AI chọn ít hơn top_n, lấy thêm từ danh sách gốc bù vào)
            if len(valid_docs) < top_n:
                for i, doc in enumerate(candidates):
                    if i not in seen:
                        valid_docs.append(doc)
                        if len(valid_docs) >= top_n: break
            
            return valid_docs[:top_n]
        else:
            logger.warning("⚠️ Rerank Empty Response. Using original order.")

    except Exception as e:
        logger.warning(f"Rerank Error: {e}")
    
    # Fallback: Trả về top_n đầu tiên của danh sách gốc
    return initial_docs[:top_n]