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
from aiolimiter import AsyncLimiter
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
# 1. RATE LIMITER & QUOTA CONFIGURATION
# ==============================================================================
# Tốc độ (Speed Limit): Giới hạn số request gửi đi trong 1 phút để tránh bị Server chặn
# Large: 20 req/phút (Trung bình 3s/req) -> Rất an toàn
LIMITER_LARGE = AsyncLimiter(20, 60)
# Small: 50 req/phút -> Nhanh hơn
LIMITER_SMALL = AsyncLimiter(50, 60)
# Embed: 300 req/phút -> Tối đa cho phép
LIMITER_EMBED = AsyncLimiter(300, 60)

# Ngân sách (Daily Quota): Tổng số request tối đa được dùng trong ngày
QUOTA_LARGE = 500
QUOTA_SMALL = 1000

# Cấu hình chạy
MAX_CONCURRENT_TASKS = 1      # Chạy 2 câu cùng lúc (An toàn cho mạng)
TIMEOUT_PER_QUESTION = 300    # 2 phút timeout (Đủ cho cả retry)
THRESHOLD_SMALL_CONTEXT = 25000 # Nếu context < 25k chars -> Dùng Small trước cho tiết kiệm
TOP_K = 12
ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"

# ==============================================================================
# 2. UTILS: SAFETY & PARSING
# ==============================================================================
def is_sensitive_topic(question):
    """Kiểm tra nhạy cảm thông minh: Block blacklist trừ khi có whitelist học thuật"""
    q_lower = question.lower()
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "bạo động", "lật đổ", "phản động", "khủng bố", "biểu tình", "chống phá",
        "giết người", "tự tử", "ma túy", "buôn lậu", "vũ khí", "bạo lực",
        "xúc phạm", "lăng mạ", "đảng cộng sản", "xuyên tạc", "cờ bạc", "cá độ"
    ]
    whitelist = [
        "luật", "nghị định", "quy định", "thông tư", "pháp luật", "hiến pháp",
        "lịch sử", "chiến tranh", "kháng chiến", "vụ án", "tòa án", "xét xử",
        "tác hại", "phòng chống", "ngăn chặn", "khái niệm", "định nghĩa"
    ]
    has_bad = any(w in q_lower for w in blacklist)
    has_good = any(w in q_lower for w in whitelist)
    return has_bad and not has_good

def find_refusal_key(options_map):
    keywords = ["không thể trả lời", "từ chối", "vi phạm", "nhạy cảm", "không phù hợp"]
    for label, text in options_map.items():
        if any(kw in str(text).lower() for kw in keywords): return label
    return None

def get_dynamic_options(row):
    options = []
    if 'choices' in row and isinstance(row['choices'], list): options = row['choices']
    elif 'options' in row and isinstance(row['options'], list): options = row['options']
    else:
        i = 1
        while True:
            val = row.get(f"option_{i}")
            if not val or str(val).lower() == 'nan': break
            options.append(str(val))
            i += 1
    return {chr(65 + i): str(text) for i, text in enumerate(options)}

def extract_answer_two_step(text, options_map):
    valid_keys = list(options_map.keys())
    fallback = valid_keys[0]
    if not text: return fallback
    text = text.strip()
    
    # Ưu tiên bắt format chuẩn
    match = re.search(r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])', text, re.IGNORECASE)
    if match and match.group(1).upper() in valid_keys:
        return match.group(1).upper()
    
    # Fallback: Tìm ký tự đáp án cuối cùng xuất hiện
    matches = re.findall(r'\b([A-Z])\b', text)
    for m in reversed(matches):
        if m.upper() in valid_keys: return m.upper()
    return fallback

def heuristic_answer(options_map):
    # Mẹo: Chọn đáp án dài nhất nếu không biết chọn gì
    return max(options_map.items(), key=lambda x: len(x[1]))[0]

# ==============================================================================
# 3. RETRIEVER
# ==============================================================================
class HybridRetriever:
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.bm25 = None
        if BM25_FILE.exists():
            try:
                with open(BM25_FILE, "rb") as f: self.bm25 = pickle.load(f)
                logger.info(f"✅ BM25 loaded: {len(self.bm25.get('chunk_ids', []))} chunks")
            except Exception as e: logger.error(f"❌ BM25 Load Error: {e}")

    async def search(self, session, query, top_k=TOP_K):
        # 1. Vector Search
        query_vec = await get_embedding_async(session, query)
        vec_hits = []
        if query_vec:
            for _ in range(3): # Retry Qdrant
                try:
                    res = await self.client.query_points(
                        Config.COLLECTION_NAME, query=query_vec, limit=top_k, with_payload=True
                    )
                    vec_hits = res.points
                    break
                except: await asyncio.sleep(1)

        # 2. BM25 Search
        bm25_hits = []
        if self.bm25:
            try:
                tokens = word_tokenize(query.lower())
                scores = self.bm25['bm25_obj'].get_scores(tokens)
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
                
                batch_scores = [scores[i] for i in top_idxs]
                threshold = max(1.0, batch_scores[int(len(batch_scores)*0.3)]) if batch_scores else 1.0

                for idx in top_idxs:
                    if scores[idx] >= threshold:
                        bm25_hits.append({
                            "id": self.bm25['chunk_ids'][idx], "score": scores[idx],
                            "text": self.bm25['texts'][idx], "title": self.bm25['titles'][idx]
                        })
            except: pass

        # 3. Fusion
        fused = {}
        max_v = max([h.score for h in vec_hits]) if vec_hits else 1.0
        for h in vec_hits:
            fused[h.payload['chunk_id']] = {
                "text": h.payload['text'], "title": h.payload['title'], "score": (h.score/max_v)*ALPHA_VECTOR
            }
        
        max_b = max([h['score'] for h in bm25_hits]) if bm25_hits else 1.0
        for h in bm25_hits:
            norm = (h['score']/max_b)*(1-ALPHA_VECTOR)
            cid = h['id']
            if cid in fused: fused[cid]['score'] += norm
            else: fused[cid] = {"text": h['text'], "title": h['title'], "score": norm}
        
        return sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:top_k]

# ==============================================================================
# 4. API CLIENTS
# ==============================================================================
async def get_embedding_async(session, text):
    await LIMITER_EMBED.acquire() # Rate Limit check
    creds = Config.VNPT_CREDENTIALS.get(Config.MODEL_EMBEDDING_API)
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": Config.MODEL_EMBEDDING_API, "input": text, "encoding_format": "float"}
    
    for i in range(2):
        try:
            async with session.post(Config.VNPT_EMBEDDING_URL, json=payload, headers=headers, timeout=30) as r:
                if r.status == 200:
                    d = await r.json()
                    if 'data' in d: return d['data'][0]['embedding']
                elif r.status in [429, 500]: await asyncio.sleep(2)
        except: await asyncio.sleep(1)
    return None

async def call_llm_generic(session, messages, model_name, stats, max_tokens=1024):
    """Gọi LLM với logic kiểm tra Quota ngân sách"""
    
    # 1. Check Quota (Ngân sách)
    if "large" in model_name.lower():
        if stats['used_large'] >= QUOTA_LARGE:
            logger.warning("💰 Hết quota Large -> Bỏ qua")
            return None
        limiter = LIMITER_LARGE
    else:
        if stats['used_small'] >= QUOTA_SMALL:
            logger.warning("💰 Hết quota Small -> Bỏ qua")
            return None
        limiter = LIMITER_SMALL

    # 2. Rate Limit (Tốc độ) - Chờ slot
    await limiter.acquire()
    
    try:
        creds = Config.VNPT_CREDENTIALS.get(model_name)
        url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
        headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
        payload = {"model": model_name, "messages": messages, "temperature": 0.1, "top_p": 0.95, "max_completion_tokens": max_tokens}
        
        async with session.post(url, json=payload, headers=headers, timeout=90) as resp:
            if resp.status == 200:
                d = await resp.json()
                if 'choices' in d: return d['choices'][0]['message']['content']
            elif resp.status >= 400:
                logger.warning(f"⚠️ API {model_name} Error {resp.status}")
    except asyncio.TimeoutError:
        logger.warning(f"⏰ Timeout gọi API {model_name}")
    except Exception as e:
        logger.warning(f"🔌 Lỗi mạng gọi API {model_name}: {str(e)[:50]}")
    
    return None

def build_prompt(question, options_text, docs):
    context = "".join([f"--- TÀI LIỆU #{i+1} ({d['title']}) ---\n{d['text']}\n\n" for i, d in enumerate(docs)])
    sys_prompt = """Bạn là trợ lý AI chuyên gia về Việt Nam (STEM & Xã hội).
NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm.
ĐỊNH DẠNG TRẢ LỜI BẮT BUỘC:
### SUY LUẬN:
[Phân tích ngắn gọn 1-2 dòng]
### ĐÁP ÁN:
[Chỉ viết 1 ký tự: A, B, C...]

QUY TẮC:
1. Ưu tiên thông tin 2024-2025.
2. Nếu là câu hỏi Toán/Lý/Hóa -> Tự tính toán từng bước.
3. Nếu vi phạm an toàn -> Chọn đáp án TỪ CHỐI."""
    
    user_prompt = f"DỮ LIỆU:\n{context}\n\nCÂU HỎI: {question}\nLỰA CHỌN:\n{options_text}\n\nTRẢ LỜI THEO ĐÚNG ĐỊNH DẠNG (### SUY LUẬN... ### ĐÁP ÁN...):"
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

# ==============================================================================
# 5. MAIN PROCESS FLOW
# ==============================================================================
async def process_row_safe(sem, session, retriever, row, stats):
    try:
        async with sem:
            qid = row.get('qid', row.get('id', 'unknown'))
            question = row.get('question', '')
            true_label = row.get('answer', None)
            
            opts = get_dynamic_options(row)
            opt_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])

            # 1. Safety Check
            if is_sensitive_topic(question):
                ans = find_refusal_key(opts) or "A"
                stats['sensitive'] += 1
                logger.info(f"🚫 Q:{qid} Sensitive")
                return {"qid": qid, "answer": ans, "is_correct": ans == true_label if true_label else None}

            # 2. Retrieval
            docs = await retriever.search(session, question, top_k=TOP_K)
            msgs = build_prompt(question, opt_text, docs)
            ctx_len = sum([len(d['text']) for d in docs])

            # 3. Model Select
            # Ưu tiên Small nếu context ngắn, hoặc nếu Large đã hết tiền
            if ctx_len < THRESHOLD_SMALL_CONTEXT or stats['used_large'] >= QUOTA_LARGE:
                model = Config.LLM_MODEL_SMALL
            else:
                model = Config.LLM_MODEL_LARGE

            # 4. Inference & Fallback
            raw = await call_llm_generic(session, msgs, model, stats)
            
            # Nếu model chính fail -> Thử model còn lại
            if not raw:
                logger.warning(f"Q:{qid} {model} Fail -> Switch model")
                fallback_model = Config.LLM_MODEL_SMALL if model == Config.LLM_MODEL_LARGE else Config.LLM_MODEL_LARGE
                raw = await call_llm_generic(session, msgs, fallback_model, stats)
                if raw: 
                    stats['fallback'] += 1
                    # Ghi nhận quota
                    if fallback_model == Config.LLM_MODEL_LARGE: stats['used_large'] += 1
                    else: stats['used_small'] += 1
            else:
                # Ghi nhận quota model chính
                if model == Config.LLM_MODEL_LARGE: stats['used_large'] += 1
                else: stats['used_small'] += 1

            # 5. Extract Result
            if not raw:
                stats['heuristic'] += 1
                final_key = heuristic_answer(opts)
                logger.error(f"Q:{qid} Both models failed -> Heuristic")
            else:
                final_key = extract_answer_two_step(raw, opts)

            is_correct = (final_key == true_label) if true_label else None
            status_icon = "✅" if is_correct else ("❌" if is_correct is False else "")
            logger.info(f"Q:{qid} | Ans:{final_key} {status_icon}")

            return {"qid": qid, "answer": final_key, "is_correct": is_correct}

    except Exception as e:
        logger.error(f"❌ Crash Q:{qid}: {e}")
        stats['crashed'] += 1
        return {"qid": qid, "answer": "A", "is_correct": False}

async def main():
    # 1. Load Data
    files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    input_file = next((f for f in files if f.exists()), None)
    if not input_file: return
    with open(input_file, 'r', encoding='utf-8') as f: data = json.load(f)

    # 2. Setup Resources
    qdrant_client = AsyncQdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=20)
    retriever = HybridRetriever(qdrant_client)
    stats = {'used_large': 0, 'used_small': 0, 'fallback': 0, 'sensitive': 0, 'heuristic': 0, 'timeout': 0, 'crashed': 0}
    
    # 3. Connection Config
    # limit=2: Khớp với MAX_CONCURRENT_TASKS để không bị nghẽn socket
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT_TASKS, force_close=True, enable_cleanup_closed=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    print(f"🔥 STARTING: {input_file.name} | Questions: {len(data)}")
    print(f"⚙️  Concurrent: {MAX_CONCURRENT_TASKS} | Timeout: {TIMEOUT_PER_QUESTION}s")
    
    start_time = time.time()
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for row in data:
            # Wrap timeout riêng cho từng task
            async def wrapper(r):
                try:
                    return await asyncio.wait_for(process_row_safe(sem, session, retriever, r, stats), timeout=TIMEOUT_PER_QUESTION)
                except asyncio.TimeoutError:
                    stats['timeout'] += 1
                    logger.error(f"⏰ Timeout Q:{r.get('qid')}")
                    return {"qid": r.get('qid'), "answer": "A", "is_correct": False}
            tasks.append(wrapper(row))
            
        results = await asyncio.gather(*tasks)

    await qdrant_client.close()
    
    # 4. Final Stats
    correct = sum([1 for r in results if r.get('is_correct')])
    total = len(results)
    print("\n" + "="*40)
    print(f"📊 ACCURACY: {correct}/{total} ({(correct/total)*100:.2f}%)")
    print(f"📈 Stats: {stats}")
    print(f"⏱️  Time: {(time.time()-start_time)/60:.1f} min")
    print("="*40)
    
    pd.DataFrame(results)[['qid', 'answer']].to_csv(Config.BASE_DIR / "output" / "submission.csv", index=False)
    print("✅ DONE!")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())