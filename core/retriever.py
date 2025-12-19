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
    Production-Grade Hybrid Retriever (Vector + BM25s).
    Features: RRF Fusion, Parallel Execution, Robust Error Handling.
    """
    
    VECTOR_TIMEOUT = 10.0 
    RETRIEVE_TIMEOUT = 5.0
    BATCH_SIZE = 50 
    RRF_K = 60
    
    def __init__(self, qdrant_client, collection_name: str, 
                 top_k: int = 5, alpha_vector: float = 0.5):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.top_k = top_k
        
        # [FIX 2] Validate Alpha
        if not 0 <= alpha_vector <= 1:
            raise ValueError(f"alpha_vector must be in [0, 1], got {alpha_vector}")
        self.alpha_vector = alpha_vector
        
        self.bm25_retriever = None
        self.chunk_ids = [] 
        self.bm25_loaded = False
        
        self._load_bm25s()
    
    def _load_bm25s(self) -> None:
        index_dir = Config.BASE_DIR / "bm25s_index"
        id_map_file = Config.BASE_DIR / "bm25s_ids.pkl"
        meta_file = Config.BASE_DIR / "bm25_metadata.json"
        
        if not index_dir.exists() or not id_map_file.exists():
            logger.warning(f"❌ BM25s files not found at {index_dir}")
            return

        try:
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    logger.info(f"Checking BM25 Meta: Ver {meta.get('version')}")

            self.bm25_retriever = bm25s.BM25.load(str(index_dir), load_corpus=False, mmap=True)
            
            with open(id_map_file, "rb") as f:
                self.chunk_ids = pickle.load(f)
                
            self.bm25_loaded = True
            logger.info(f"✓ BM25s Loaded successfully: {len(self.chunk_ids)} docs")
            
        except Exception as e:
            logger.error(f"Failed to load BM25s: {e}", exc_info=True)
            self.bm25_retriever = None
            self.bm25_loaded = False

    def get_stats(self) -> Dict:
        return {
            "bm25_loaded": self.bm25_loaded,
            "bm25_chunks": len(self.chunk_ids) if self.bm25_loaded else 0,
            "collection": self.collection_name,
            "alpha": self.alpha_vector
        }

    async def search(self, session, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.top_k
        
        # 1. Parallel Search
        vec_task = self._vector_search(session, query, top_k)
        bm25_task = self._bm25_search(query, top_k)
        
        (vec_hits_map, vec_scores), bm25_scores = await asyncio.gather(
            vec_task, bm25_task, return_exceptions=True
        )
        
        # Error Handling
        if isinstance(vec_hits_map, Exception):
            logger.error(f"Vector search crashed: {vec_hits_map}")
            vec_hits_map, vec_scores = {}, {}
        
        if isinstance(bm25_scores, Exception):
            logger.error(f"BM25 search crashed: {bm25_scores}")
            bm25_scores = {}
        
        # 2. Fetch Missing Text (Parallel Batched)
        vec_hits_map, vec_scores = await self._fetch_missing_text_parallel(
            vec_hits_map, vec_scores, bm25_scores
        )
        
        # 3. Fusion
        final_results = self._fuse_scores_rrf(vec_hits_map, vec_scores, bm25_scores)
        
        return final_results[:top_k]
    
    async def _vector_search(self, session, query: str, top_k: int) -> Tuple[Dict, Dict]:
        vec_hits_map = {}
        vec_scores = {}
        
        try:
            query_vec = await get_embedding_async(session, query)
            if not query_vec: return {}, {}
            
            res = await asyncio.wait_for(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k * 2,
                    with_payload=True
                ),
                timeout=self.VECTOR_TIMEOUT
            )
            
            for point in res.points:
                if not point.payload or 'chunk_id' not in point.payload: continue
                cid = str(point.payload['chunk_id'])
                vec_hits_map[cid] = point.payload
                vec_scores[cid] = float(point.score)
                
        except Exception as e:
            # [FIX 4] Better Logging context
            logger.error(f"Vector search error for query '{query[:30]}...': {e}")
        
        return vec_hits_map, vec_scores
    
    async def _bm25_search(self, query: str, top_k: int) -> Dict[str, float]:
        bm25_scores = {}
        if not self.bm25_loaded: return bm25_scores
        
        try:
            text = str(query).lower().translate(TRANSLATOR)
            tokens = word_tokenize(text)
            query_tokens = [t for t in tokens if len(t.strip()) > 0]
            
            if not query_tokens: return bm25_scores
            
            # BM25s retrieve (Fast C++)
            doc_indices, doc_scores = self.bm25_retriever.retrieve([query_tokens], k=top_k * 2)
            
            indices_list = doc_indices[0]
            scores_list = doc_scores[0]
            
            for i, doc_idx in enumerate(indices_list):
                score = float(scores_list[i])
                if score > 0:
                    chunk_id = self.chunk_ids[doc_idx]
                    bm25_scores[chunk_id] = score
                    
        except Exception as e:
            logger.error(f"BM25s search error: {e}")
            
        return bm25_scores
    
    async def _fetch_missing_text_parallel(self, vec_hits_map, vec_scores, bm25_scores):
        """
        [FIX 3] Parallel Batch Fetching
        """
        missing_ids = [cid for cid in bm25_scores.keys() if cid not in vec_hits_map]
        if not missing_ids: return vec_hits_map, vec_scores
        
        # Chia batch
        batches = [missing_ids[i:i + self.BATCH_SIZE] for i in range(0, len(missing_ids), self.BATCH_SIZE)]
        
        async def fetch_batch(batch_ids):
            try:
                point_ids = [generate_uuid5(cid) for cid in batch_ids]
                points = await self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=point_ids,
                    with_payload=True
                )
                return points
            except Exception as e:
                logger.error(f"Fetch batch error: {e}")
                return []

        # Chạy song song các batch
        results = await asyncio.gather(*[fetch_batch(b) for b in batches], return_exceptions=True)
        
        # Merge results
        for res_list in results:
            if isinstance(res_list, list):
                for point in res_list:
                    if point.payload and 'chunk_id' in point.payload:
                        cid = str(point.payload['chunk_id'])
                        vec_hits_map[cid] = point.payload
                        if cid not in vec_scores:
                            vec_scores[cid] = 0.0 # Vector score = 0
                            
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
            
            payload = vec_hits_map[cid]
            final_results.append({
                "chunk_id": cid,
                "text": payload.get('text', ''),
                "title": payload.get('title', ''),
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