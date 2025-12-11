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

LIMITER_EMBED = AsyncLimiter(300, 60) # 300 req/phút
LIMITER_LLM = AsyncLimiter(50, 60)    # 50 req/phút (An toàn tuyệt đối)

MAX_CONCURRENT_TASKS = 4 # Giảm xuống 4 để tránh nghẽn
TIMEOUT_PER_QUESTION = 240 
THRESHOLD_LARGE_CHARS = 40000 
TOP_K = 12 
ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"

# ==============================================================================
# 1. SMART SAFETY CHECK (OFFLINE - KHÔNG GỌI API)
# ==============================================================================
def is_sensitive_topic(question):
    """
    Kiểm tra nhạy cảm thông minh:
    - Block: Các từ khóa đen.
    - Allow: Nếu có từ khóa học thuật/pháp luật đi kèm thì CHO PHÉP.
    """
    q_lower = question.lower()
    
    # 1. Danh sách đen
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "bạo động", "lật đổ", "phản động", "khủng bố", "biểu tình", "chống phá",
        "giết người", "tự tử", "ma túy", "buôn lậu", "vũ khí", "bạo lực",
        "xúc phạm", "lăng mạ", "đảng cộng sản", "xuyên tạc", "cờ bạc", "cá độ"
    ]
    
    # 2. Danh sách trắng (Bảo vệ các câu hỏi học thuật)
    whitelist = [
        "luật", "nghị định", "quy định", "thông tư", "pháp luật", "hiến pháp",
        "lịch sử", "chiến tranh", "kháng chiến", "vụ án", "tòa án", "xét xử",
        "tác hại", "phòng chống", "ngăn chặn", "khái niệm", "định nghĩa"
    ]

    has_bad_word = any(w in q_lower for w in blacklist)
    has_good_word = any(w in q_lower for w in whitelist)

    # Nếu có từ xấu NHƯNG cũng có từ học thuật -> Coi là AN TOÀN (False positive)
    if has_bad_word and has_good_word:
        return False # Safe
    
    return has_bad_word # Unsafe

def find_refusal_key(options_map):
    keywords = ["không thể trả lời", "từ chối", "vi phạm", "nhạy cảm", "không phù hợp", "tác động tiêu cực"]
    for label, text in options_map.items():
        if any(kw in str(text).lower() for kw in keywords):
            return label
    return None

# ==============================================================================
# 2. RETRIEVER (DEPENDENCY INJECTION)
# ==============================================================================
class HybridRetriever:
    def __init__(self, qdrant_client):
        # Nhận client từ bên ngoài vào (Dependency Injection)
        self.client = qdrant_client
        self.bm25 = None
        if BM25_FILE.exists():
            try:
                with open(BM25_FILE, "rb") as f: self.bm25 = pickle.load(f)
            except Exception as e:
                logger.error(f"BM25 Load Error: {e}")

    async def search_qdrant_retry(self, query_vec, top_k, max_retries=3):
        for i in range(max_retries):
            try:
                # Dùng query_points (API mới)
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
                logger.error(f"BM25 Error: {e}")

        # Fusion
        fused = {}
        max_v = max([h.score for h in vec_hits]) if vec_hits else 1.0
        for h in vec_hits:
            fused[h.payload['chunk_id']] = {"text": h.payload['text'], "title": h.payload['title'], "score": (h.score/max_v)*ALPHA_VECTOR}
        
        max_b = max([h['score'] for h in bm25_hits]) if bm25_hits else 1.0
        for h in bm25_hits:
            norm = (h['score']/max_b)*(1-ALPHA_VECTOR)
            cid = h['id']
            if cid in fused: fused[cid]['score'] += norm
            else: fused[cid] = {"text": h['text'], "title": h['title'], "score": norm}
            
        return sorted(fused.values(), key=lambda x: x['score'], reverse=True)[:top_k]

# ==============================================================================
# 3. API CLIENTS & PROMPTS
# ==============================================================================
async def call_llm_generic(session, messages, model_name, max_tokens=1024, retry=3):
    creds = Config.VNPT_CREDENTIALS.get(model_name)
    url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": model_name, "messages": messages, "temperature": 0.1, "top_p": 0.95, "max_completion_tokens": max_tokens}
    
    for _ in range(retry):
        try:
            async with LIMITER_LLM:
                # Tăng timeout lên 60s
                async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
                    if resp.status == 200:
                        d = await resp.json()
                        if 'choices' in d: return d['choices'][0]['message']['content']
                    elif resp.status in [429, 500, 502, 503]: 
                        await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"LLM Net Error: {str(e)[:50]}")
            await asyncio.sleep(1)
    return None

async def get_embedding_async(session, text):
    model = Config.MODEL_EMBEDDING_API
    creds = Config.VNPT_CREDENTIALS.get(model)
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": model, "input": text, "encoding_format": "float"}
    for i in range(3):
        try:
            async with LIMITER_EMBED:
                async with session.post(Config.VNPT_EMBEDDING_URL, json=payload, headers=headers, timeout=30) as r:
                    if r.status == 200: 
                        d = await r.json()
                        if 'data' in d: return d['data'][0]['embedding']
                    elif r.status in [429, 500, 401]: await asyncio.sleep(2 * (i+1))
        except: await asyncio.sleep(1)
    return None

def build_prompt(question, options_text, valid_keys_str, docs):
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
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

# ==============================================================================
# 4. PROCESSOR
# ==============================================================================
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
    
    mapped = {}
    for idx, text in enumerate(options): mapped[chr(65 + idx)] = str(text)
    return mapped

def extract_answer_two_step(text, options_map):
    valid_keys = list(options_map.keys())
    fallback = valid_keys[0]
    if not text: return fallback
    text = text.strip()
    match_strict = re.search(r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])', text, re.IGNORECASE)
    if match_strict and match_strict.group(1).upper() in valid_keys: return match_strict.group(1).upper()
    patterns = [r'(?:đáp án|chọn|là)[:\s\*\-\.\[\(]*([A-Z])[\]\)\*\.]*$', r'\*\*([A-Z])\*\*', r'^([A-Z])[\.\)]\s']
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match and match.group(1).upper() in valid_keys: return match.group(1).upper()
    return fallback

async def process_row_safe(sem, session, retriever, row):
    try:
        async with sem:
            qid = row.get('qid', row.get('id', 'unknown'))
            question = row.get('question', '')
            true_label = row.get('answer', None)

            options_map = get_dynamic_options(row)
            options_text = "\n".join([f"{k}. {v}" for k, v in options_map.items()])
            valid_keys_str = ", ".join(options_map.keys())

            # 1. Smart Safety Check
            if is_sensitive_topic(question):
                logger.info(f"🚫 Q:{qid} -> Sensitive (Offline Check).")
                refusal_key = find_refusal_key(options_map)
                pred = refusal_key if refusal_key else "A"
                return {"qid": qid, "answer": pred, "is_correct": pred == true_label if true_label else None}

            # 2. Retrieval
            docs = await retriever.search(session, question, top_k=TOP_K)
            
            # 3. Gen Answer
            messages = build_prompt(question, options_text, valid_keys_str, docs)
            context_chars = sum([len(d['text']) for d in docs])
            
            model = Config.LLM_MODEL_LARGE
            if context_chars > THRESHOLD_LARGE_CHARS: model = Config.LLM_MODEL_SMALL
            
            raw_ans = await call_llm_generic(session, messages, model)
            if not raw_ans and model == Config.LLM_MODEL_LARGE:
                raw_ans = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL)

            final_key = extract_answer_two_step(raw_ans, options_map)
            
            status = f"| {'✅' if final_key == true_label else '❌'}" if true_label else ""
            logger.info(f"Q:{qid} | Ans:{final_key} {status}")
            return {"qid": qid, "answer": final_key, "is_correct": final_key == true_label if true_label else None}

    except Exception as e:
        logger.error(f"❌ Crash Q:{qid}: {e}")
        return {"qid": qid, "answer": "A", "is_correct": False}

async def process_with_timeout(sem, session, retriever, row):
    try:
        return await asyncio.wait_for(process_row_safe(sem, session, retriever, row), timeout=TIMEOUT_PER_QUESTION)
    except asyncio.TimeoutError:
        qid = row.get('qid', 'unknown')
        logger.error(f"⏰ Timeout Q:{qid}")
        return {"qid": qid, "answer": "A", "is_correct": False}

# ==============================================================================
# 5. MAIN
# ==============================================================================
async def main():
    files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    input_file = next((f for f in files if f.exists()), None)
    if not input_file: return
    
    with open(input_file, 'r', encoding='utf-8') as f: data = json.load(f)
    
    # [QUAN TRỌNG] Khởi tạo Qdrant Client 1 lần ở đây
    qdrant_client = AsyncQdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=20)
    retriever = HybridRetriever(qdrant_client) # Truyền client vào

    conn = aiohttp.TCPConnector(limit=10, force_close=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    print(f"🔥 STARTING: {input_file.name} | Tasks: {MAX_CONCURRENT_TASKS}")
    
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [process_with_timeout(sem, session, retriever, row) for row in data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # [QUAN TRỌNG] Chỉ đóng client khi TẤT CẢ đã xong
    await qdrant_client.close()

    # Xử lý kết quả
    clean_results = []
    correct = 0
    has_label = False
    
    for r in results:
        if isinstance(r, dict):
            clean_results.append(r)
            if r.get('is_correct') is not None:
                has_label = True
                if r['is_correct']: correct += 1
        else:
            clean_results.append({"qid": "unknown", "answer": "A"})

    if has_label and len(clean_results) > 0:
        print(f"📊 SCORE: {correct}/{len(clean_results)} ({(correct/len(clean_results))*100:.2f}%)")

    pd.DataFrame(clean_results)[['qid', 'answer']].to_csv(Config.BASE_DIR / "output" / "submission.csv", index=False)
    print("DONE!")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())