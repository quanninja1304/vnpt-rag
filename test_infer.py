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
import random
from pathlib import Path
from aiolimiter import AsyncLimiter
from qdrant_client import AsyncQdrantClient
from underthesea import word_tokenize
from config import Config

# ==============================================================================
# 0. CẤU HÌNH CHIẾN THUẬT (Tactical Config)
# ==============================================================================
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File lưu kết quả (Dùng để Resume)
OUTPUT_FILE = Config.BASE_DIR / "output" / "submission.csv"

# Cấu hình chạy an toàn tuyệt đối
MAX_CONCURRENT_TASKS = 1      # Chạy từng câu một (Chậm nhưng chắc 100%)
TIMEOUT_PER_QUESTION = 120    # 2 phút/câu (Đủ để retry nếu mạng lag)

# Rate Limit (Tốc độ)
LIMITER_LARGE = AsyncLimiter(20, 60)   # 20 req/phút
LIMITER_SMALL = AsyncLimiter(50, 60)   # 50 req/phút
LIMITER_EMBED = AsyncLimiter(300, 60)  # 300 req/phút

# Ngân sách (Quota) - Để theo dõi
QUOTA_LARGE = 500
QUOTA_SMALL = 1000

# Constants
THRESHOLD_SMALL_CONTEXT = 20000 
TOP_K = 12
ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'inference_resume.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VNPT_BOT")

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ (UTILS)
# ==============================================================================
def is_sensitive_topic(question):
    q_lower = question.lower()
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "phản động", "khủng bố", "giết người", "ma túy", "buôn lậu", "vũ khí", "bạo lực",
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
    
    # Ưu tiên 1: Format chuẩn
    match = re.search(r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])', text, re.IGNORECASE)
    if match and match.group(1).upper() in valid_keys: return match.group(1).upper()
    
    # Ưu tiên 2: Markdown
    match = re.search(r'\*\*([A-Z])\*\*', text)
    if match and match.group(1).upper() in valid_keys: return match.group(1).upper()

    # Fallback: Tìm ký tự cuối cùng
    matches = re.findall(r'\b([A-Z])\b', text)
    for m in reversed(matches):
        if m.upper() in valid_keys: return m.upper()
    return fallback

def heuristic_answer(options_map):
    # Chọn đáp án dài nhất
    return max(options_map.items(), key=lambda x: len(str(x[1])))[0]

def build_prompt(question, options_text, docs):
    context = "".join([f"--- TÀI LIỆU #{i+1} ({d['title']}) ---\n{d['text']}\n\n" for i, d in enumerate(docs)])
    sys_prompt = """Bạn là trợ lý AI chuyên gia (STEM & Xã hội).
NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm dựa trên dữ liệu.
ĐỊNH DẠNG:
### SUY LUẬN:
[Phân tích ngắn]
### ĐÁP ÁN:
[Chỉ viết 1 ký tự A, B, C...]"""
    user_prompt = f"DỮ LIỆU:\n{context}\n\nCÂU HỎI: {question}\nLỰA CHỌN:\n{options_text}\n\nTRẢ LỜI THEO ĐÚNG ĐỊNH DẠNG:"
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

# ==============================================================================
# 2. RETRIEVER & API CLIENTS
# ==============================================================================
class HybridRetriever:
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.bm25 = None
        if BM25_FILE.exists():
            try:
                with open(BM25_FILE, "rb") as f: self.bm25 = pickle.load(f)
                logger.info(f"BM25 loaded: {len(self.bm25.get('chunk_ids', []))} chunks")
            except: pass

    async def search(self, session, query, top_k=TOP_K):
        # 1. Embed
        query_vec = await get_embedding_async(session, query)
        vec_hits = []
        if query_vec:
            for _ in range(3):
                try:
                    res = await self.client.query_points(Config.COLLECTION_NAME, query=query_vec, limit=top_k, with_payload=True)
                    vec_hits = res.points
                    break
                except: await asyncio.sleep(1)

        # 2. BM25
        bm25_hits = []
        if self.bm25:
            try:
                tokens = word_tokenize(query.lower())
                scores = self.bm25['bm25_obj'].get_scores(tokens)
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
                thresh = max(1.0, scores[top_idxs[int(len(top_idxs)*0.3)]]) if top_idxs else 1.0
                for idx in top_idxs:
                    if scores[idx] >= thresh:
                        bm25_hits.append({"id": self.bm25['chunk_ids'][idx], "score": scores[idx], "text": self.bm25['texts'][idx], "title": self.bm25['titles'][idx]})
            except: pass

        # 3. Fusion
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

async def get_embedding_async(session, text):
    await LIMITER_EMBED.acquire()
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
    limiter = LIMITER_LARGE if "large" in model_name.lower() else LIMITER_SMALL
    await limiter.acquire()
    
    # Ghi nhận dùng quota
    if "large" in model_name.lower(): stats['used_large'] += 1
    else: stats['used_small'] += 1

    try:
        creds = Config.VNPT_CREDENTIALS.get(model_name)
        url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
        headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
        payload = {"model": model_name, "messages": messages, "temperature": 0.1, "top_p": 0.95, "max_completion_tokens": max_tokens}
        
        # Jitter để tránh dồn toa
        await asyncio.sleep(random.uniform(0.5, 1.5))

        async with session.post(url, json=payload, headers=headers, timeout=90) as resp:
            if resp.status == 200:
                d = await resp.json()
                if 'choices' in d: return d['choices'][0]['message']['content']
            elif resp.status >= 400:
                logger.warning(f"⚠️ API {model_name} Error {resp.status}")
    except Exception as e:
        logger.warning(f"🔌 Net Error {model_name}: {str(e)[:30]}")
    return None

# ==============================================================================
# 3. CORE LOGIC (PROCESS SINGLE ROW)
# ==============================================================================
async def process_row_logic(session, retriever, row, stats):
    """Xử lý 1 dòng, trả về kết quả"""
    qid = row.get('qid', row.get('id', 'unknown'))
    question = row.get('question', '')
    true_label = row.get('answer', None)
    
    opts = get_dynamic_options(row)
    opt_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])

    # 1. Safety
    if is_sensitive_topic(question):
        ans = find_refusal_key(opts) or "A"
        logger.info(f"🚫 Q:{qid} Sensitive")
        return {"qid": qid, "answer": ans, "is_correct": ans == true_label if true_label else None}

    # 2. Retrieval
    docs = await retriever.search(session, question, top_k=TOP_K)
    msgs = build_prompt(question, opt_text, docs)
    ctx_len = sum([len(d['text']) for d in docs])

    # 3. Model
    model = Config.LLM_MODEL_LARGE
    if ctx_len < THRESHOLD_SMALL_CONTEXT: model = Config.LLM_MODEL_SMALL

    # 4. Infer
    raw = await call_llm_generic(session, msgs, model, stats)
    
    if not raw:
        # Fallback
        fallback_model = Config.LLM_MODEL_SMALL if model == Config.LLM_MODEL_LARGE else Config.LLM_MODEL_LARGE
        logger.warning(f"⚠️ Q:{qid} Fallback -> {fallback_model}")
        raw = await call_llm_generic(session, msgs, fallback_model, stats)

    # 5. Extract
    if not raw:
        ans = heuristic_answer(opts)
        logger.error(f"0 Q:{qid} Failed all models -> Heuristic")
    else:
        ans = extract_answer_two_step(raw, opts)

    is_correct = (ans == true_label) if true_label else None
    status = "1" if is_correct else ("0" if is_correct is False else "")
    logger.info(f"Q:{qid} | Ans:{ans} {status}")
    
    return {"qid": qid, "answer": ans, "is_correct": is_correct}

# ==============================================================================
# 4. MAIN LOOP WITH RESUME
# ==============================================================================
async def main():
    # 1. Load Data
    files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    input_file = next((f for f in files if f.exists()), None)
    if not input_file: return
    with open(input_file, 'r', encoding='utf-8') as f: data = json.load(f)

    # 2. Check Resume (Đọc file đã lưu)
    processed_ids = set()
    if OUTPUT_FILE.exists():
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(df_done['qid'].astype(str))
            logger.info(f"RESUMING... Found {len(processed_ids)} processed questions.")
        except: pass
    
    # Lọc câu chưa làm
    data_to_process = [r for r in data if str(r.get('qid', r.get('id'))) not in processed_ids]
    
    if not data_to_process:
        logger.info("ALL DONE! Nothing to process.")
        return

    logger.info(f"REMAINING: {len(data_to_process)}/{len(data)} questions")

    # 3. Setup
    qdrant_client = AsyncQdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=30)
    retriever = HybridRetriever(qdrant_client)
    stats = {'used_large': 0, 'used_small': 0}
    
    # 4. Run Sequential (Vòng lặp đơn luồng)
    conn = aiohttp.TCPConnector(limit=1, force_close=True, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=conn) as session:
        
        for i, row in enumerate(data_to_process):
            qid = row.get('qid', row.get('id'))
            
            # Retry loop cho từng câu
            for attempt in range(3):
                try:
                    # Timeout cứng
                    result = await asyncio.wait_for(
                        process_row_logic(session, retriever, row, stats),
                        timeout=TIMEOUT_PER_QUESTION
                    )
                    
                    # --- WRITE TO DISK IMMEDIATELY ---
                    df_res = pd.DataFrame([result])
                    # Nếu file chưa có thì ghi header, có rồi thì append không header
                    need_header = not OUTPUT_FILE.exists()
                    df_res[['qid', 'answer']].to_csv(OUTPUT_FILE, mode='a', header=need_header, index=False)
                    
                    break # Success -> Next question
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout Q:{qid} (Attempt {attempt+1})")
                    if attempt == 2:
                        # Fail hẳn -> Ghi 'A' để lần sau không bị kẹt
                        pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)
                except Exception as e:
                    logger.error(f"Error Q:{qid}: {e}")
                    await asyncio.sleep(5)

            # Nghỉ ngơi giữa các câu để server thở
            await asyncio.sleep(1)

    await qdrant_client.close()
    logger.info("BATCH COMPLETED!")

    if OUTPUT_FILE.exists():
        print("\n" + "="*40)
        print("TỔNG KẾT TOÀN BỘ (CUMULATIVE STATS)")
        print("="*40)
        
        try:
            # 1. Đọc toàn bộ kết quả đã lưu trong CSV
            df_results = pd.read_csv(OUTPUT_FILE)
            
            # 2. Tạo từ điển đáp án đúng (Ground Truth) từ file input gốc
            # Lưu ý: Chỉ lấy những câu có trường 'answer' (đề phòng file Test không có)
            ground_truth = {
                str(r.get('qid', r.get('id'))): str(r.get('answer')).strip() 
                for r in data if r.get('answer')
            }
            
            if not ground_truth:
                print("Đây là tập Test (không có đáp án) -> Bỏ qua tính điểm.")
            else:
                correct_count = 0
                total_checked = 0
                
                # 3. So khớp từng câu trong CSV với đáp án gốc
                for _, row in df_results.iterrows():
                    qid = str(row['qid'])
                    # Chuyển về string và strip để so sánh chính xác
                    pred = str(row['answer']).strip()
                    
                    if qid in ground_truth:
                        total_checked += 1
                        true_label = ground_truth[qid]
                        
                        # So sánh
                        if pred == true_label:
                            correct_count += 1
                
                # 4. In kết quả
                if total_checked > 0:
                    acc = (correct_count / total_checked) * 100
                    print(f"Đã làm: {total_checked}/{len(ground_truth)} câu")
                    print(f"Đúng  : {correct_count} câu")
                    print(f"Tỷ lệ : {acc:.2f}%")
                else:
                    print("⚠️ Chưa có câu nào khớp ID với tập dữ liệu gốc.")
                    
        except Exception as e:
            print(f"Lỗi tính điểm: {e}")

        print(f"File kết quả: {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())