import asyncio
import aiohttp
import pandas as pd
import json
import re
import pickle
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from qdrant_client import AsyncQdrantClient, models
from underthesea import word_tokenize
from config import Config

# ==============================================================================
# 0. CONFIG & LOGGING
# ==============================================================================
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'inference.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VNPT_BOT")

# ==============================================================================
# 1. SMART RATE LIMITER (Daily Quota Management)
# ==============================================================================
# class SmartRateLimiter:
#     """
#     Rate limiter theo ngày với auto-recovery.
#     Tự động đợi khi hết quota và reset sau 24h.
#     """
#     def __init__(self, daily_limit, buffer=0.9):
#         self.daily_limit = int(daily_limit * buffer)  # Để buffer 10% an toàn
#         self.requests_log = deque()
#         self.lock = asyncio.Lock()
    
#     async def acquire(self):
#         async with self.lock:
#             now = datetime.now()
#             cutoff = now - timedelta(days=1)
            
#             # Xóa các request cũ hơn 24h
#             while self.requests_log and self.requests_log[0] < cutoff:
#                 self.requests_log.popleft()
            
#             # Kiểm tra quota
#             if len(self.requests_log) >= self.daily_limit:
#                 oldest = self.requests_log[0]
#                 sleep_time = (oldest - cutoff).total_seconds()
#                 logger.warning(f"⏳ Daily limit reached ({self.daily_limit}). Sleeping {sleep_time:.0f}s")
#                 await asyncio.sleep(sleep_time + 1)
#                 return await self.acquire()  # Recursive check sau khi sleep
            
#             # Ghi nhận request
#             self.requests_log.append(now)
#             logger.debug(f"✅ Token acquired. Used: {len(self.requests_log)}/{self.daily_limit}")

class SimpleRateLimiter:
    """
    Rate limiter đơn giản theo thời gian.
    Đảm bảo khoảng cách tối thiểu giữa các request.
    """
    def __init__(self, requests_per_minute=50):
        self.min_interval = 60.0 / requests_per_minute  # giây giữa mỗi request
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug(f"⏳ Rate limit: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()


class EmbeddingRateLimiter:
    """Rate limiter theo phút cho embedding (300 req/phút)"""
    def __init__(self, per_minute=300):
        self.per_minute = per_minute
        self.requests_log = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            cutoff = now - 60
            
            # Xóa request cũ hơn 1 phút
            while self.requests_log and self.requests_log[0] < cutoff:
                self.requests_log.popleft()
            
            # Nếu đã đầy -> đợi
            if len(self.requests_log) >= self.per_minute:
                sleep_time = self.requests_log[0] - cutoff + 0.1
                await asyncio.sleep(sleep_time)
                return await self.acquire()
            
            self.requests_log.append(now)

# Khởi tạo limiters
# LIMITER_LARGE = SmartRateLimiter(daily_limit=500)
# LIMITER_SMALL = SmartRateLimiter(daily_limit=1000)
# LIMITER_EMBED = EmbeddingRateLimiter(per_minute=300)

LIMITER_LARGE = SimpleRateLimiter(requests_per_minute=0.3)  # 1 req/3s = 20/min max
LIMITER_SMALL = SimpleRateLimiter(requests_per_minute=0.6)  # 1 req/1.7s = 35/min max
LIMITER_EMBED = SimpleRateLimiter(requests_per_minute=300)  # 300/min như cũ

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
MAX_CONCURRENT_TASKS = 3  # Tăng từ 2 lên 3 (vì đã fix rate limiter)
TIMEOUT_PER_QUESTION = 120  # Giảm từ 300s xuống 120s (2 phút)
THRESHOLD_SMALL_CONTEXT = 20000  # Context < 20k chars -> dùng Small model
TOP_K = 12
ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"

# ==============================================================================
# 3. SMART SAFETY CHECK (OFFLINE)
# ==============================================================================
def is_sensitive_topic(question):
    """
    Kiểm tra nhạy cảm thông minh:
    - Block: Các từ khóa đen.
    - Allow: Nếu có từ khóa học thuật/pháp luật đi kèm thì CHO PHÉP.
    """
    q_lower = question.lower()
    
    # Danh sách đen
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "bạo động", "lật đổ", "phản động", "khủng bố", "biểu tình", "chống phá",
        "giết người", "tự tử", "ma túy", "buôn lậu", "vũ khí", "bạo lực",
        "xúc phạm", "lăng mạ", "đảng cộng sản", "xuyên tạc", "cờ bạc", "cá độ"
    ]
    
    # Danh sách trắng (Bảo vệ các câu hỏi học thuật)
    whitelist = [
        "luật", "nghị định", "quy định", "thông tư", "pháp luật", "hiến pháp",
        "lịch sử", "chiến tranh", "kháng chiến", "vụ án", "tòa án", "xét xử",
        "tác hại", "phòng chống", "ngăn chặn", "khái niệm", "định nghĩa"
    ]

    has_bad_word = any(w in q_lower for w in blacklist)
    has_good_word = any(w in q_lower for w in whitelist)

    # Nếu có từ xấu NHƯNG cũng có từ học thuật -> An toàn
    if has_bad_word and has_good_word:
        return False
    
    return has_bad_word

def find_refusal_key(options_map):
    """Tìm đáp án từ chối trong options"""
    keywords = ["không thể trả lời", "từ chối", "vi phạm", "nhạy cảm", 
                "không phù hợp", "tác động tiêu cực"]
    for label, text in options_map.items():
        if any(kw in str(text).lower() for kw in keywords):
            return label
    return None

# ==============================================================================
# 4. RETRIEVER (DEPENDENCY INJECTION)
# ==============================================================================
class HybridRetriever:
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.bm25 = None
        if BM25_FILE.exists():
            try:
                with open(BM25_FILE, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info(f"✅ BM25 loaded: {len(self.bm25.get('chunk_ids', []))} chunks")
            except Exception as e:
                logger.error(f"❌ BM25 Load Error: {e}")

    async def search_qdrant_retry(self, query_vec, top_k, max_retries=3):
        """Search Qdrant với retry"""
        for i in range(max_retries):
            try:
                response = await self.client.query_points(
                    collection_name=Config.COLLECTION_NAME,
                    query=query_vec,
                    limit=top_k,
                    with_payload=True
                )
                return response.points
            except Exception as e:
                if i == max_retries - 1:
                    logger.error(f"❌ Qdrant Fail: {e}")
                    return []
                await asyncio.sleep(1)
        return []

    async def search(self, session, query, top_k=TOP_K):
        """Hybrid search: Vector + BM25"""
        # Vector Search
        query_vec = await get_embedding_async(session, query)
        vec_hits = []
        if query_vec:
            vec_hits = await self.search_qdrant_retry(query_vec, top_k)
        
        # BM25 Search
        bm25_hits = []
        if self.bm25:
            try:
                tokens = word_tokenize(query.lower())
                scores = self.bm25['bm25_obj'].get_scores(tokens)
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
                
                # Dynamic threshold
                batch_scores = [scores[i] for i in top_idxs]
                threshold = 1.0
                if batch_scores:
                    dynamic_thresh = batch_scores[int(len(batch_scores) * 0.3)]
                    threshold = max(1.0, dynamic_thresh)

                for idx in top_idxs:
                    if scores[idx] >= threshold:
                        bm25_hits.append({
                            "id": self.bm25['chunk_ids'][idx],
                            "score": scores[idx],
                            "text": self.bm25['texts'][idx],
                            "title": self.bm25['titles'][idx]
                        })
            except Exception as e:
                logger.error(f"❌ BM25 Error: {e}")

        # Fusion
        fused = {}
        max_v = max([h.score for h in vec_hits]) if vec_hits else 1.0
        for h in vec_hits:
            fused[h.payload['chunk_id']] = {
                "text": h.payload['text'],
                "title": h.payload['title'],
                "score": (h.score/max_v) * ALPHA_VECTOR
            }
        
        max_b = max([h['score'] for h in bm25_hits]) if bm25_hits else 1.0
        for h in bm25_hits:
            norm = (h['score']/max_b) * (1 - ALPHA_VECTOR)
            cid = h['id']
            if cid in fused:
                fused[cid]['score'] += norm
            else:
                fused[cid] = {"text": h['text'], "title": h['title'], "score": norm}
        
        return sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:top_k]

# ==============================================================================
# 5. API CLIENTS
# ==============================================================================
async def call_llm_generic(session, messages, model_name, max_tokens=1024):
    """
    Gọi LLM với smart rate limiting.
    KHÔNG retry - chỉ gọi 1 lần duy nhất để tiết kiệm quota.
    """
    # Chọn limiter theo model
    if "large" in model_name.lower():
        limiter = LIMITER_LARGE
    else:
        limiter = LIMITER_SMALL
    
    # Acquire token (tự động đợi nếu hết quota)
    await limiter.acquire()
    
    try:
        creds = Config.VNPT_CREDENTIALS.get(model_name)
        url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
        headers = {
            'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
            'Token-id': creds['token_id'],
            'Token-key': creds['token_key'],
            'Content-Type': 'application/json'
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.95,
            "max_completion_tokens": max_tokens
        }
        
        async with session.post(url, json=payload, headers=headers, timeout=90) as resp:
            if resp.status == 200:
                d = await resp.json()
                if 'choices' in d:
                    return d['choices'][0]['message']['content']
            
            # Log lỗi API
            if resp.status >= 400:
                error_text = await resp.text()
                logger.error(f"❌ API Error {resp.status} ({model_name}): {error_text[:100]}")
            
    except asyncio.TimeoutError:
        logger.warning(f"⏰ LLM Timeout ({model_name})")
    except Exception as e:
        logger.warning(f"🔌 Network Error ({model_name}): {str(e)[:50]}")
    
    return None

async def get_embedding_async(session, text):
    """Get embedding với rate limit"""
    await LIMITER_EMBED.acquire()
    
    model = Config.MODEL_EMBEDDING_API
    creds = Config.VNPT_CREDENTIALS.get(model)
    headers = {
        'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
        'Token-id': creds['token_id'],
        'Token-key': creds['token_key'],
        'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "input": text,
        "encoding_format": "float"
    }
    
    for i in range(2):  # Chỉ retry 1 lần
        try:
            async with session.post(Config.VNPT_EMBEDDING_URL, json=payload, 
                                   headers=headers, timeout=30) as r:
                if r.status == 200:
                    d = await r.json()
                    if 'data' in d:
                        return d['data'][0]['embedding']
                elif r.status in [429, 500] and i == 0:
                    await asyncio.sleep(2)
        except Exception as e:
            if i == 0:
                await asyncio.sleep(1)
    
    return None

# ==============================================================================
# 6. PROMPT BUILDER
# ==============================================================================
def build_prompt(question, options_text, docs):
    """Build prompt cho LLM"""
    context_str = ""
    for i, doc in enumerate(docs):
        context_str += f"--- TÀI LIỆU #{i+1} ({doc['title']}) ---\n{doc['text']}\n\n"

    system_prompt = """Bạn là trợ lý AI chuyên gia về Việt Nam (STEM & Xã hội).
NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm.
ĐỊNH DẠNG TRẢ LỜI BẮT BUỘC:
### SUY LUẬN:
[Phân tích ngắn gọn]
### ĐÁP ÁN:
[Chỉ viết 1 ký tự: A, B, C...]

QUY TẮC:
1. Ưu tiên thông tin 2024-2025.
2. Nếu là câu hỏi Toán/Lý/Hóa -> Tự tính toán từng bước.
3. Nếu vi phạm an toàn -> Chọn đáp án TỪ CHỐI.
4. Nếu hỏi về Luật/Lịch sử có chứa từ nhạy cảm -> Vẫn trả lời theo kiến thức pháp luật."""

    user_prompt = f"""DỮ LIỆU:\n{context_str}\n\nCÂU HỎI: {question}\nLỰA CHỌN:\n{options_text}\n\nTRẢ LỜI THEO ĐÚNG ĐỊNH DẠNG (### SUY LUẬN... ### ĐÁP ÁN...):"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ==============================================================================
# 7. ANSWER EXTRACTION
# ==============================================================================
def get_dynamic_options(row):
    """Extract options từ row (hỗ trợ nhiều format)"""
    options = []
    if 'choices' in row and isinstance(row['choices'], list):
        options = row['choices']
    elif 'options' in row and isinstance(row['options'], list):
        options = row['options']
    else:
        i = 1
        while True:
            val = row.get(f"option_{i}")
            if not val or str(val).lower() == 'nan':
                break
            options.append(str(val))
            i += 1
    
    mapped = {}
    for idx, text in enumerate(options):
        mapped[chr(65 + idx)] = str(text)
    return mapped

def extract_answer_two_step(text, options_map):
    """Extract đáp án từ LLM response"""
    valid_keys = list(options_map.keys())
    fallback = valid_keys[0]
    
    if not text:
        return fallback
    
    text = text.strip()
    
    # Pattern 1: ### ĐÁP ÁN: X
    match_strict = re.search(r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])', text, re.IGNORECASE)
    if match_strict and match_strict.group(1).upper() in valid_keys:
        return match_strict.group(1).upper()
    
    # Pattern 2: Các pattern khác
    patterns = [
        r'(?:đáp án|chọn|là)[:\s\*\-\.\[\(]*([A-Z])[\]\)\*\.]*$',
        r'\*\*([A-Z])\*\*',
        r'^([A-Z])[\.\)]\s'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match and match.group(1).upper() in valid_keys:
            return match.group(1).upper()
    
    return fallback

def heuristic_answer(options_map):
    """Fallback: Chọn đáp án dài nhất (thường đúng trong trắc nghiệm)"""
    return max(options_map.items(), key=lambda x: len(x[1]))[0]

# ==============================================================================
# 8. PROCESSOR
# ==============================================================================
async def process_row_safe(sem, session, retriever, row, stats):
    """Process 1 câu hỏi với smart retry strategy"""
    try:
        async with sem:
            qid = row.get('qid', row.get('id', 'unknown'))
            question = row.get('question', '')
            true_label = row.get('answer', None)

            options_map = get_dynamic_options(row)
            options_text = "\n".join([f"{k}. {v}" for k, v in options_map.items()])

            # 1. Smart Safety Check (Offline)
            if is_sensitive_topic(question):
                logger.info(f"🚫 Q:{qid} -> Sensitive (Offline Check).")
                refusal_key = find_refusal_key(options_map)
                pred = refusal_key if refusal_key else "A"
                stats['sensitive'] += 1
                return {
                    "qid": qid,
                    "answer": pred,
                    "is_correct": pred == true_label if true_label else None
                }

            # 2. Retrieval
            docs = await retriever.search(session, question, top_k=TOP_K)
            messages = build_prompt(question, options_text, docs)
            context_chars = sum([len(d['text']) for d in docs])

            # 3. Smart Model Selection
            # Ưu tiên SMALL nếu context nhỏ (tiết kiệm quota large)
            if context_chars < THRESHOLD_SMALL_CONTEXT:
                model = Config.LLM_MODEL_SMALL
                logger.debug(f"Q:{qid} -> Small (ctx={context_chars})")
                stats['used_small'] += 1
            else:
                model = Config.LLM_MODEL_LARGE
                logger.debug(f"Q:{qid} -> Large (ctx={context_chars})")
                stats['used_large'] += 1

            # 4. Gọi model chính (CHỈ 1 LẦN)
            raw_ans = await call_llm_generic(session, messages, model)
            
            # 5. Fallback nếu model chính fail
            if not raw_ans:
                fallback_model = (Config.LLM_MODEL_SMALL if model == Config.LLM_MODEL_LARGE 
                                 else Config.LLM_MODEL_LARGE)
                logger.warning(f"Q:{qid} -> Fallback to {fallback_model.split('_')[-1]}")
                raw_ans = await call_llm_generic(session, messages, fallback_model)
                stats['fallback'] += 1
                
                if fallback_model == Config.LLM_MODEL_SMALL:
                    stats['used_small'] += 1
                else:
                    stats['used_large'] += 1

            # 6. Extract answer hoặc dùng heuristic
            if not raw_ans:
                logger.error(f"Q:{qid} -> Both models failed. Use heuristic.")
                final_key = heuristic_answer(options_map)
                stats['heuristic'] += 1
            else:
                final_key = extract_answer_two_step(raw_ans, options_map)

            # 7. Log result
            status = f"| {'✅' if final_key == true_label else '❌'}" if true_label else ""
            logger.info(f"Q:{qid} | Ans:{final_key} {status}")
            
            return {
                "qid": qid,
                "answer": final_key,
                "is_correct": final_key == true_label if true_label else None
            }

    except Exception as e:
        logger.error(f"❌ Crash Q:{qid}: {e}")
        stats['crashed'] += 1
        return {"qid": qid, "answer": "A", "is_correct": False}

async def process_with_timeout(sem, session, retriever, row, stats):
    """Wrapper với timeout"""
    try:
        return await asyncio.wait_for(
            process_row_safe(sem, session, retriever, row, stats),
            timeout=TIMEOUT_PER_QUESTION
        )
    except asyncio.TimeoutError:
        qid = row.get('qid', 'unknown')
        logger.error(f"⏰ Timeout Q:{qid}")
        stats['timeout'] += 1
        return {"qid": qid, "answer": "A", "is_correct": False}

# ==============================================================================
# 9. MAIN
# ==============================================================================
async def main():
    # Load data
    files = [
        Config.BASE_DIR / "data" / "val.json",
        Config.BASE_DIR / "data" / "test.json"
    ]
    input_file = next((f for f in files if f.exists()), None)
    if not input_file:
        logger.error("❌ No input file found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize clients
    qdrant_client = AsyncQdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        timeout=20
    )
    retriever = HybridRetriever(qdrant_client)

    # Stats tracking
    stats = {
        'used_large': 0,
        'used_small': 0,
        'fallback': 0,
        'sensitive': 0,
        'heuristic': 0,
        'timeout': 0,
        'crashed': 0
    }
    
    # Progress tracking
    start_time = time.time()
    completed = 0
    
    async def track_progress(task):
        nonlocal completed
        result = await task
        completed += 1
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (len(data) - completed) if completed > 0 else 0
        logger.info(f"📊 Progress: {completed}/{len(data)} | ETA: {eta/60:.1f}min")
        return result

    # Setup connection
    conn = aiohttp.TCPConnector(limit=10, force_close=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    print(f"🔥 STARTING: {input_file.name}")
    print(f"📝 Total: {len(data)} questions | Concurrent: {MAX_CONCURRENT_TASKS}")
    print(f"⏱️  Timeout: {TIMEOUT_PER_QUESTION}s/question")
    print(f"🎯 Strategy: Small first (ctx<{THRESHOLD_SMALL_CONTEXT})")
    print("-" * 60)
    
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            track_progress(process_with_timeout(sem, session, retriever, row, stats))
            for row in data
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Close client
    await qdrant_client.close()

    # Process results
    clean_results = []
    correct = 0
    has_label = False
    
    for r in results:
        if isinstance(r, dict):
            clean_results.append(r)
            if r.get('is_correct') is not None:
                has_label = True
                if r['is_correct']:
                    correct += 1
        else:
            logger.error(f"❌ Invalid result: {r}")
            clean_results.append({"qid": "unknown", "answer": "A"})

    # Print statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    if has_label and len(clean_results) > 0:
        print(f"✅ Correct: {correct}/{len(clean_results)} ({(correct/len(clean_results))*100:.2f}%)")
    print(f"🤖 Model Usage:")
    print(f"   - Large: {stats['used_large']} calls")
    print(f"   - Small: {stats['used_small']} calls")
    print(f"   - Fallback: {stats['fallback']} times")
    print(f"🛡️  Safety: {stats['sensitive']} sensitive questions")
    print(f"🎲 Heuristic: {stats['heuristic']} times")
    print(f"⏰ Timeout: {stats['timeout']} questions")
    print(f"💥 Crashed: {stats['crashed']} questions")
    print(f"⏱️  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print("=" * 60)

    # Save results
    output_file = Config.BASE_DIR / "output" / "submission.csv"
    pd.DataFrame(clean_results)[['qid', 'answer']].to_csv(output_file, index=False)
    print(f"💾 Saved to: {output_file}")
    print("✅ DONE!")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())