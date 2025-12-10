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
from qdrant_client import QdrantClient
from underthesea import word_tokenize
from config import Config

# ==============================================================================
# 0. LOGGING SETUP (PROFESSIONAL GRADE)
# ==============================================================================
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'submission.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VNPT_BOT")

# ==============================================================================
# 1. CẤU HÌNH & CONSTANTS
# ==============================================================================
LIMITER_EMBED = AsyncLimiter(500, 60)
LIMITER_LLM = AsyncLimiter(100, 60)

MAX_CONCURRENT_TASKS = 8
TIMEOUT_PER_QUESTION = 55 # Giây (Để dư 5s xử lý fallback)

# Ngưỡng Large: 18k tokens ~ 45k chars. Safe = 40k.
THRESHOLD_LARGE_CHARS = 40000 

ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"
QUOTA_FILE = Config.OUTPUT_DIR / "quota_tracker.json"

# ==============================================================================
# 2. QUOTA MANAGER
# ==============================================================================
class QuotaManager:
    def __init__(self, is_private_mode=False):
        self.is_private = is_private_mode
        self.daily_limit = 450 # Buffer an toàn
        self.lock = asyncio.Lock()
        self.usage_data = self._load_usage()

    def _load_usage(self):
        if self.is_private: return {"count": 0}
        today_str = time.strftime("%Y-%m-%d")
        default_data = {"date": today_str, "count": 0}
        if os.path.exists(QUOTA_FILE):
            try:
                with open(QUOTA_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get("date") == today_str:
                        logger.info(f"📊 Daily Usage Loaded: {data['count']}/{self.daily_limit}")
                        return data
            except: pass
        return default_data

    async def _save_usage(self):
        if self.is_private: return
        with open(QUOTA_FILE, 'w') as f:
            json.dump(self.usage_data, f)

    async def can_use_large(self):
        if self.is_private: return True
        async with self.lock:
            if self.usage_data["count"] < self.daily_limit:
                self.usage_data["count"] += 1
                if self.usage_data["count"] % 10 == 0: await self._save_usage()
                return True
            return False

    async def refund_large(self):
        if self.is_private: return
        async with self.lock:
            if self.usage_data["count"] > 0:
                self.usage_data["count"] -= 1
                await self._save_usage()

# ==============================================================================
# 3. UTILS (ADAPTIVE LOGIC)
# ==============================================================================
def get_adaptive_top_k(question):
    """Điều chỉnh số lượng chunk dựa trên độ dài câu hỏi"""
    q_len = len(question.split()) # Đếm số từ
    if q_len < 10: return 8   # Câu ngắn -> Lấy ít cho đỡ nhiễu
    if q_len < 30: return 10  # Trung bình
    return 12                 # Câu dài/phức tạp -> Lấy nhiều context hơn

def get_dynamic_options(row):
    options = []
    # [FIX] Ưu tiên check key 'choices' theo format json thực tế của BTC
    if 'choices' in row and isinstance(row['choices'], list):
        options = row['choices']
    # Hỗ trợ thêm key 'options' (dự phòng)
    elif 'options' in row and isinstance(row['options'], list):
        options = row['options']
    # Fallback cho CSV (option_1, option_2...)
    else:
        i = 1
        while True:
            val = row.get(f"option_{i}")
            if val is None: break
            val_str = str(val).strip()
            if not val_str or val_str.lower() == 'nan': break
            options.append(val_str)
            i += 1
    
    mapped = {}
    for idx, text in enumerate(options):
        mapped[chr(65 + idx)] = str(text) # A: text, B: text
    return mapped

async def verify_safety_with_model(session, question):
    """
    Dùng Model Small để kiểm tra kỹ lại các câu hỏi dính keyword.
    Trả về: True (An toàn/Kiến thức) hoặc False (Độc hại/Cấm kỵ)
    """
    system_prompt = """Bạn là hệ thống kiểm duyệt nội dung an toàn (Content Safety Moderator).
Nhiệm vụ: Phân loại câu hỏi của người dùng là AN TOÀN (Safe) hay ĐỘC HẠI (Unsafe).

ĐỊNH NGHĨA:
1. ĐỘC HẠI (Unsafe):
   - Yêu cầu hướng dẫn thực hiện hành vi phạm pháp (chế tạo vũ khí, ma túy, giết người).
   - Ngôn từ thù ghét, xúc phạm danh nhân/lãnh tụ, chống phá nhà nước.
   - Nội dung khiêu dâm, đồi trụy.

2. AN TOÀN (Safe):
   - Câu hỏi tìm kiếm kiến thức, học thuật (Lịch sử, Pháp luật, Y học).
   - Ví dụ: "Luật hình sự phạt tội giết người thế nào?" -> AN TOÀN (Hỏi về luật).
   - Ví dụ: "Vũ khí quân đội Việt Nam năm 1975?" -> AN TOÀN (Hỏi về lịch sử).

OUTPUT BẮT BUỘC: Chỉ trả lời một từ duy nhất: SAFE hoặc UNSAFE."""

    user_prompt = f"CÂU HỎI: {question}\nPHÂN LOẠI:"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    # Gọi Model Small (Max speed)
    resp = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, max_tokens=10)
    
    if resp:
        # Nếu model trả lời SAFE -> An toàn
        if "SAFE" in resp.upper() and "UN" not in resp.upper():
            return True
        # Nếu model trả lời UNSAFE -> Độc hại
        if "UNSAFE" in resp.upper():
            return False
    
    # Fallback: Nếu model lỗi, mặc định là False (Chặn nhầm hơn bỏ sót cho an toàn)
    return False

def is_sensitive_topic(question):
    # Danh sách từ khóa nghi vấn (Suspicious Keywords)
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "bạo động", "lật đổ", "phản động", "khủng bố", "biểu tình", "chống phá",
        "giết người", "tự tử", "ma túy", "buôn lậu", "vũ khí", "bạo lực", "bom mìn", "thuốc nổ",
        "xúc phạm", "lăng mạ", "chính quyền", "đảng cộng sản", "xuyên tạc",
        "cờ bạc", "cá độ", "mại dâm", "đánh bạc"
    ]
    q_lower = question.lower()
    return any(w in q_lower for w in blacklist)

def find_refusal_key(options_map):
    keywords = ["không thể trả lời", "từ chối", "vi phạm", "nhạy cảm", "không phù hợp", "tác động tiêu cực"]
    for label, text in options_map.items():
        if any(kw in str(text).lower() for kw in keywords):
            return label
    return None

def extract_answer_two_step(text, options_map):
    """Parser thông minh hỗ trợ format '### ĐÁP ÁN:'"""
    valid_keys = list(options_map.keys())
    fallback = valid_keys[0]
    
    if not text: return fallback
    text = text.strip()

    # Priority 1: Tìm theo format Prompt bắt buộc
    # Bắt: "### ĐÁP ÁN: A" hoặc "### ĐÁP ÁN:\n[A]"
    match_strict = re.search(r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])', text, re.IGNORECASE)
    if match_strict:
        key = match_strict.group(1).upper()
        if key in valid_keys: return key

    # Priority 2: Regex Markdown/Common
    patterns = [
        r'(?:đáp án|chọn|là)[:\s\*\-\.\[\(]*([A-Z])[\]\)\*\.]*$',
        r'\*\*([A-Z])\*\*',
        r'^([A-Z])[\.\)]\s'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            key = match.group(1).upper()
            if key in valid_keys: return key

    # Priority 3: Fuzzy Matching (So sánh text)
    text_lower = text.lower()
    best_match = None
    max_len = 0
    for key, opt_text in options_map.items():
        opt_lower = opt_text.lower()
        # Nếu model rep nguyên câu đáp án
        if opt_lower in text_lower:
            if len(opt_lower) > max_len:
                max_len = len(opt_lower)
                best_match = key
    if best_match: return best_match

    # Priority 4: Tìm ký tự cuối cùng
    matches = re.findall(r'\b([A-Z])\b', text)
    if matches:
        for cand in reversed(matches):
            if cand.upper() in valid_keys: return cand.upper()

    return fallback

# ==============================================================================
# 4. API CLIENT
# ==============================================================================
async def call_llm_generic(session, messages, model_name, max_tokens=1024, retry=3):
    creds = Config.VNPT_CREDENTIALS.get(model_name)
    url = f"{Config.VNPT_API_URL}/{model_name.replace('_', '-')}"
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": model_name, "messages": messages, "temperature": 0.1, "top_p": 0.95, "max_completion_tokens": max_tokens}

    for attempt in range(retry):
        try:
            async with LIMITER_LLM:
                async with session.post(url, json=payload, headers=headers, timeout=50) as resp:
                    if resp.status == 200:
                        try:
                            d = await resp.json()
                            if 'choices' in d and d['choices']: return d['choices'][0]['message']['content']
                        except Exception as e:
                            logger.error(f"JSON Error {model_name}: {e}")
                    
                    # Retry logic
                    txt = await resp.text()
                    if resp.status in [429, 500, 502, 503] or (resp.status == 401 and "Rate limit" in txt):
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"API Error {model_name} {resp.status}: {txt}")
                        return None
        except Exception as e:
            logger.warning(f"Net Error {model_name}: {str(e)[:50]}")
            await asyncio.sleep(1)
    return None

async def get_embedding_async(session, text):
    model = Config.MODEL_EMBEDDING_API
    creds = Config.VNPT_CREDENTIALS.get(model)
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": model, "input": text, "encoding_format": "float"}
    for _ in range(4):
        try:
            async with LIMITER_EMBED:
                async with session.post(Config.VNPT_EMBEDDING_URL, json=payload, headers=headers, timeout=30) as r:
                    if r.status == 200: 
                        d = await r.json()
                        if 'data' in d: return d['data'][0]['embedding']
                    elif r.status in [429, 500, 401]: await asyncio.sleep(2)
        except: await asyncio.sleep(1)
    return None

# ==============================================================================
# 5. ROBUST HYBRID RETRIEVER
# ==============================================================================
class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        self.bm25 = None
        if BM25_FILE.exists():
            with open(BM25_FILE, "rb") as f: self.bm25 = pickle.load(f)

    async def search_qdrant_retry(self, query_vec, top_k, max_retries=3):
        """Qdrant với cơ chế Retry"""
        for i in range(max_retries):
            try:
                return self.client.search(collection_name=Config.COLLECTION_NAME, query_vector=query_vec, limit=top_k, with_payload=True)
            except Exception as e:
                if i == max_retries - 1:
                    logger.error(f"Qdrant Fail: {e}")
                    return []
                await asyncio.sleep(0.5)
        return []

    async def search(self, session, query, top_k=10):
        # 1. Vector Search
        query_vec = await get_embedding_async(session, query)
        vec_hits = []
        if query_vec:
            vec_hits = await self.search_qdrant_retry(query_vec, top_k)
        
        # 2. BM25 Search (Dynamic Threshold)
        bm25_hits = []
        if self.bm25:
            tokens = word_tokenize(query.lower())
            scores = self.bm25['bm25_obj'].get_scores(tokens)
            
            # Chỉ lấy Top 2*k để lọc
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
            
            # Dynamic Threshold (Top 30% của batch này hoặc min 1.0)
            batch_scores = [scores[i] for i in top_idxs]
            if batch_scores:
                # Lấy giá trị ở phân vị 70% (Top 30%)
                dynamic_thresh = batch_scores[int(len(batch_scores) * 0.3)]
                threshold = max(1.0, dynamic_thresh) # Không thấp hơn 1.0
            else:
                threshold = 1.0

            for idx in top_idxs:
                if scores[idx] >= threshold: 
                    bm25_hits.append({"id": self.bm25['chunk_ids'][idx], "score": scores[idx], "text": self.bm25['texts'][idx], "title": self.bm25['titles'][idx]})

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

# ==============================================================================
# 6. PIPELINE CHÍNH (STRUCTURED PROMPT)
# ==============================================================================
def build_prompt(question, options_text, valid_keys_str, docs):
    context_str = ""
    for i, doc in enumerate(docs):
        context_str += f"--- TÀI LIỆU #{i+1} ({doc['title']}) ---\n{doc['text']}\n\n"

    # System Prompt: Đóng vai chuyên gia đa năng
    system_prompt = """Bạn là chuyên gia giải quyết vấn đề đa lĩnh vực (STEM & Xã hội).
NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm chính xác nhất.

HƯỚNG DẪN SUY LUẬN (CHAIN-OF-THOUGHT):
1. **Xác định loại câu hỏi:**
   - Nếu là Lịch sử/Địa lý/Luật/Văn hóa: Tìm keyword trong tài liệu, so khớp thời gian/địa điểm. Ưu tiên thông tin 2024-2025.
   - Nếu là Toán/Lý/Hóa/Sinh/Tin: Xác định công thức, định lý, biến số và thực hiện TÍNH TOÁN từng bước.

2. **Quy tắc xử lý:**
   - Ưu tiên thông tin trong DỮ LIỆU THAM KHẢO.
   - Nếu dữ liệu không đủ để tính toán/suy luận (đặc biệt là câu hỏi STEM): Hãy dùng KIẾN THỨC KHOA HỌC CHUẨN XÁC của bạn để giải quyết.
   - Kiểm tra kỹ các "bẫy" (đơn vị đo, phủ định "không", ngoại lệ).
   - Nếu câu hỏi vi phạm tiêu chuẩn an toàn/đạo đức -> Chọn đáp án TỪ CHỐI.

ĐỊNH DẠNG OUTPUT BẮT BUỘC:
### SUY LUẬN:
[Viết ngắn gọn 2-3 dòng phân tích, phép tính hoặc dẫn chứng]

### ĐÁP ÁN:
[Chỉ viết 1 ký tự đại diện đáp án đúng: A, B, C, D...]"""

    # User Prompt: Ép định dạng trả lời
    user_prompt = f"""DỮ LIỆU THAM KHẢO:\n{context_str}\n\nCÂU HỎI: {question}\nLỰA CHỌN:\n{options_text}\n\nHÃY SUY LUẬN LOGIC VÀ TÍNH TOÁN (NẾU CẦN) ĐỂ CHỌN ĐÁP ÁN ĐÚNG TRONG ({valid_keys_str}):"""
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

async def select_model_strategy(context_chars, quota_mgr, qid):
    """
    Logic chọn model tường minh:
    1. Ưu tiên Large.
    2. Bị giáng xuống Small nếu Context quá dài.
    3. Bị giáng xuống Small nếu Hết Quota.
    """
    # 1. Check kỹ thuật: Context Window
    if context_chars > THRESHOLD_LARGE_CHARS:
        logger.warning(f"⚠️ Q:{qid} Context Too Long ({context_chars}). Forced SMALL.")
        return Config.LLM_MODEL_SMALL

    # 2. Check tài nguyên: Quota Manager
    # Lưu ý: Hàm can_use_large() sẽ TỰ ĐỘNG trừ quota nếu trả về True
    if await quota_mgr.can_use_large():
        return Config.LLM_MODEL_LARGE
    
    # 3. Fallback: Hết quota hoặc không được dùng
    # logger.info(f"ℹ️ Q:{qid} Quota limit reached. Using SMALL.") # Uncomment nếu muốn log chi tiết
    return Config.LLM_MODEL_SMALL

# ==============================================================================
# CẬP NHẬT HÀM PROCESS_ROW_SAFE
# ==============================================================================
async def process_row_safe(sem, session, retriever, quota_mgr, row):
    try:
        async with sem:
            # Lấy QID và Câu hỏi
            qid = row.get('qid', row.get('id', 'unknown'))
            question = row.get('question', '')
            
            # [VAL MODE] Lấy đáp án đúng nếu có (để chấm điểm tập Val)
            true_label = row.get('answer', None)

            # 1. Options & Mapping
            options_map = get_dynamic_options(row)
            options_text = "\n".join([f"{k}. {v}" for k, v in options_map.items()])
            valid_keys = list(options_map.keys())
            valid_keys_str = ", ".join(valid_keys)

            # 2. Safety Check (Offline)
            if is_sensitive_topic(question):
                logger.info(f"⚠️ Q:{qid} dính Keyword nhạy cảm. Đang thẩm định lại bằng AI...")
                
                # Bước 2: Thẩm định bằng Model Small
                is_safe_context = await verify_safety_with_model(session, question)
                
                if not is_safe_context:
                    logger.info(f"🚫 Q:{qid} -> XÁC NHẬN ĐỘC HẠI. Từ chối trả lời.")
                    refusal_key = find_refusal_key(options_map)
                    return {"qid": qid, "answer": refusal_key if refusal_key else "A"}
                else:
                    logger.info(f"✅ Q:{qid} -> False Positive (Hỏi kiến thức/Luật). Tiếp tục xử lý.")

            # 3. Retrieval
            adaptive_k = get_adaptive_top_k(question)
            docs = await retriever.search(session, question, top_k=adaptive_k)
            
            # 4. Prompt & Model Selection (Dùng hàm mới)
            messages = build_prompt(question, options_text, valid_keys_str, docs)
            context_chars = sum([len(d['text']) for d in docs])
            
            # [FIX LOGIC] Gọi hàm chọn model tường minh
            model_to_use = await select_model_strategy(context_chars, quota_mgr, qid)

            # 5. Inference
            raw_ans = await call_llm_generic(session, messages, model_to_use)
            
            # Fallback: Large Fail -> Small
            if raw_ans is None and model_to_use == Config.LLM_MODEL_LARGE:
                logger.warning(f"🔄 Q:{qid} Large Fail -> Retry Small.")
                await quota_mgr.refund_large() # Hoàn quota
                raw_ans = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL)

            # 6. Extract Answer
            final_key = extract_answer_two_step(raw_ans, options_map)
            
            # [LOGGING] In kết quả kèm chấm điểm (nếu có label)
            log_suffix = ""
            is_correct = None
            if true_label:
                is_correct = (final_key == true_label)
                icon = "✅" if is_correct else "❌"
                log_suffix = f"| True: {true_label} {icon}"
            
            logger.info(f"Q:{qid} | Mod:{'L' if model_to_use==Config.LLM_MODEL_LARGE else 'S'} | Ans:{final_key} {log_suffix}")
            
            return {"qid": qid, "answer": final_key, "is_correct": is_correct}

    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout Q:{qid}")
        return {"qid": qid, "answer": "A", "is_correct": False if true_label else None}
    except Exception as e:
        logger.error(f"❌ Crash Q:{qid}: {e}")
        return {"qid": qid, "answer": "A", "is_correct": False if true_label else None}

async def process_row_with_timeout(sem, session, retriever, quota_mgr, row):
    # Wrapper để bắt timeout
    return await asyncio.wait_for(
        process_row_safe(sem, session, retriever, quota_mgr, row),
        timeout=TIMEOUT_PER_QUESTION
    )

async def main():
    # Detect Input File
    # Ưu tiên theo thứ tự: private -> public -> val -> test
    files_to_check = [
        Config.BASE_DIR / "data" / "private_test.json", 
        Config.BASE_DIR / "data" / "val.json",          # File bạn muốn chạy
        Config.BASE_DIR / "data" / "test.json",
        Config.BASE_DIR / "data" / "public_test.json"   
    ]
    
    input_file = None
    is_private = False
    
    for f in files_to_check:
        if f.exists():
            input_file = f
            # Nếu tên file là val.json hoặc private -> Chế độ Private (Unlimited Quota)
            if "private" in f.name or f.name == "val.json":
                is_private = True
            break
    
    if not input_file:
        print("❌ Không tìm thấy file input data/")
        return

    logger.info(f"🚀 STARTING | File: {input_file.name} | Mode: {'PRIVATE/VAL (Unlimited)' if is_private else 'PUBLIC (Quota)'}")

    # Load Data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Init
    quota_mgr = QuotaManager(is_private)
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT_TASKS + 5, force_close=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    retriever = HybridRetriever()

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [process_row_with_timeout(sem, session, retriever, quota_mgr, row) for row in data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Clean results & Calculate Score
    clean_results = []
    correct_count = 0
    has_label = False

    for i, res in enumerate(results):
        if isinstance(res, dict):
            clean_results.append(res)
            if res.get('is_correct') is not None:
                has_label = True
                if res['is_correct']: correct_count += 1
        else:
            # Handle Exception
            qid = data[i].get('qid', 'unknown')
            logger.error(f"🔥 Critical Failure Q:{qid}")
            clean_results.append({"qid": qid, "answer": "A"})

    # Print Validation Score
    if has_label and len(clean_results) > 0:
        acc = (correct_count / len(clean_results)) * 100
        logger.info("="*40)
        logger.info(f"📊 VALIDATION SCORE: {correct_count}/{len(clean_results)} ({acc:.2f}%)")
        logger.info("="*40)

    # Save output
    out_df = pd.DataFrame(clean_results)
    # Chỉ giữ cột cần thiết cho file nộp
    final_df = out_df[['qid', 'answer']]
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(Config.BASE_DIR / "output" / "submission.csv", index=False)
    logger.info(f"🎉 DONE! Output saved to output/submission.csv")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())