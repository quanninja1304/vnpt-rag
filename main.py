import asyncio
import aiohttp
import pandas as pd
import json
import re
import pickle
import sys
import os
from pathlib import Path
from aiolimiter import AsyncLimiter
from qdrant_client import QdrantClient
from underthesea import word_tokenize
from config import Config

# ==============================================================================
# 1. CẤU HÌNH & QUOTA MANAGER
# ==============================================================================
# Rate Limit (Tốc độ)
LIMITER_EMBED = AsyncLimiter(300, 60) # 300 req/phút
LIMITER_LLM = AsyncLimiter(100, 60)   # 100 req/phút

# Concurrency Limit (Số luồng song song - Tránh tràn RAM)
MAX_CONCURRENT_TASKS = 15 

# Quota Limit (Tổng số request/ngày - Tránh hết tiền)
# Trừ hao 20 request để test hoặc lỗi mạng
MAX_QUOTA_LARGE = 480 
MAX_QUOTA_SMALL = 980

class QuotaManager:
    """Quản lý số lượng request để không bị hết quota giữa chừng"""
    def __init__(self):
        self.large_used = 0
        self.small_used = 0
        self.lock = asyncio.Lock()

    async def check_and_increment(self, model_type):
        async with self.lock:
            if model_type == Config.LLM_MODEL_LARGE:
                if self.large_used < MAX_QUOTA_LARGE:
                    self.large_used += 1
                    return True # Cho phép dùng Large
                return False # Hết quota Large
            else:
                self.small_used += 1
                return True # Small cứ dùng (hoặc check limit nếu cần)

QUOTA_MGR = QuotaManager()

TOP_K = 6
ALPHA_VECTOR = 0.7
BM25_FILE = Config.OUTPUT_DIR / "bm25_index.pkl"

# ==============================================================================
# 2. CLIENT GỌI API (ROBUST)
# ==============================================================================
async def call_llm_generic(session, messages, model_name, max_tokens=1024, retry=3):
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

    for attempt in range(retry):
        try:
            async with LIMITER_LLM:
                # [FIX CONNECTION] force_close=True để tránh lỗi Server disconnected
                async with session.post(url, json=payload, headers=headers, timeout=45) as resp:
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            # [FIX CRASH] Kiểm tra kỹ xem có 'choices' không
                            if 'choices' in data and len(data['choices']) > 0:
                                return data['choices'][0]['message']['content']
                            else:
                                # API trả 200 nhưng nội dung lỗi
                                print(f"⚠️ API {model_name} Weird Response: {data}")
                                return None
                        except Exception as json_err:
                            print(f"❌ JSON Parse Error: {json_err}")
                            return None
                    
                    text_resp = await resp.text()
                    # Retry nếu lỗi Rate Limit hoặc Server
                    if resp.status in [429, 500, 502, 503] or (resp.status == 401 and "Rate limit" in text_resp):
                        wait = 2 * (attempt + 1)
                        print(f"⚠️ {model_name} Busy ({resp.status}). Wait {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        print(f"❌ Error {model_name} ({resp.status}): {text_resp}")
                        return None
        except Exception as e:
            # Lỗi mạng thuần túy (disconnect)
            print(f"⚠️ Net Error {model_name}: {str(e)[:50]}...") # In ngắn gọn
            await asyncio.sleep(1)
            
    return None

async def get_embedding_async(session, text):
    model = Config.MODEL_EMBEDDING_API
    creds = Config.VNPT_CREDENTIALS.get(model)
    headers = {'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}', 'Token-id': creds['token_id'], 'Token-key': creds['token_key'], 'Content-Type': 'application/json'}
    payload = {"model": model, "input": text, "encoding_format": "float"}

    for attempt in range(3):
        async with LIMITER_EMBED:
            try:
                async with session.post(Config.VNPT_EMBEDDING_URL, json=payload, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data: return data['data'][0]['embedding']
                    
                    text_resp = await resp.text()
                    if resp.status in [429, 500] or (resp.status == 401 and "Rate limit" in text_resp):
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
            except:
                await asyncio.sleep(1)
    return None

# ==============================================================================
# 3. HYBRID RETRIEVER
# ==============================================================================
class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        self.bm25_data = None
        if BM25_FILE.exists():
            with open(BM25_FILE, "rb") as f: self.bm25_data = pickle.load(f)

    async def search(self, session, query, top_k=TOP_K):
        # 1. Vector
        query_vec = await get_embedding_async(session, query)
        vec_hits = []
        if query_vec:
            try:
                vec_hits = self.client.search(collection_name=Config.COLLECTION_NAME, query_vector=query_vec, limit=top_k, with_payload=True)
            except: pass
        
        # 2. BM25
        bm25_hits = []
        if self.bm25_data:
            tokens = word_tokenize(query.lower())
            scores = self.bm25_data['bm25_obj'].get_scores(tokens)
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for idx in top_idxs:
                if scores[idx] > 0.8:
                    bm25_hits.append({"id": self.bm25_data['chunk_ids'][idx], "score": scores[idx], "text": self.bm25_data['texts'][idx], "title": self.bm25_data['titles'][idx]})

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
# 4. LOGIC TRẢ LỜI & QUOTA ROUTING
# ==============================================================================
def is_sensitive(question):
    blacklist = ["sex", "khiêu dâm", "bạo động", "lật đổ", "phản động", "giết người", "khủng bố"]
    return any(w in question.lower() for w in blacklist)

def build_advanced_prompt(question, options_text, docs):
    context_str = ""
    for i, doc in enumerate(docs):
        context_str += f"--- TÀI LIỆU #{i+1} ({doc['title']}) ---\n{doc['text']}\n\n"

    system_prompt = """Bạn là trợ lý AI chuyên gia về Việt Nam. Nhiệm vụ:
1. Đọc tài liệu tham khảo.
2. Trả lời câu hỏi trắc nghiệm.
3. Nếu tài liệu mâu thuẫn thời gian, ƯU TIÊN THÔNG TIN MỚI NHẤT (2024-2025)."""

    user_prompt = f"""
DỮ LIỆU:
{context_str}

CÂU HỎI: {question}
LỰA CHỌN:
{options_text}

HƯỚNG DẪN:
Suy luận từng bước:
1. Tìm từ khóa & mốc thời gian.
2. Tìm thông tin trong tài liệu.
3. So sánh lựa chọn.
4. CHỈ TRẢ VỀ KÝ TỰ ĐÁP ÁN (A, B, C, D)."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

async def generate_answer(session, question, options_text, docs):
    messages = build_advanced_prompt(question, options_text, docs)
    
    # Ước lượng độ dài Context
    context_len = sum([len(d['text']) for d in docs]) * 1.5
    total_tokens = context_len + 1000 
    
    # --- LOGIC CHỌN MODEL (SMART ROUTER) ---
    selected_model = Config.LLM_MODEL_SMALL # Mặc định dùng Small (cho an toàn quota)
    
    # 1. Nếu Context quá lớn -> BẮT BUỘC dùng Small
    if total_tokens > 18000:
        use_large = False
    else:
        # 2. Nếu Context vừa phải -> Check Quota Large xem còn không?
        use_large = await QUOTA_MGR.check_and_increment(Config.LLM_MODEL_LARGE)
    
    if use_large:
        selected_model = Config.LLM_MODEL_LARGE
    
    # Gọi Model
    answer = await call_llm_generic(session, messages, selected_model)
    
    # Fallback: Nếu Large fail -> Gọi Small
    if answer is None and selected_model == Config.LLM_MODEL_LARGE:
        print("🔄 Fallback to SMALL model.")
        await QUOTA_MGR.check_and_increment(Config.LLM_MODEL_SMALL) # Count usage small
        answer = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL)
        
    return answer if answer else "A"

def extract_key(text):
    match = re.search(r'(?:đáp án|chọn)[:\s]*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1].upper() if matches else "A"

# ==============================================================================
# 5. MAIN PIPELINE (ĐÃ SỬA CONCURRENCY & INPUT)
# ==============================================================================
async def process_row_safe(sem, session, retriever, row):
    """Wrapper có Semaphore để giới hạn số luồng chạy cùng lúc"""
    async with sem:
        try:
            # Parse input JSON (Key có thể khác nhau tùy file, cần linh hoạt)
            qid = row.get('id', row.get('qid', 'unknown'))
            question = row.get('question', '')
            
            # [FIX INPUT] Lấy options từ JSON (thường là list hoặc các field rời)
            # Giả sử format: "option_1": "...", "option_2": "..." HOẶC "options": ["...", "..."]
            if 'options' in row and isinstance(row['options'], list):
                opts = row['options']
                options_text = f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}"
            else:
                options_text = "\n".join([f"{k}. {row.get(f'option_{i}', '')}" for i, k in enumerate(['A','B','C','D'], 1)])

            if is_sensitive(question):
                return {"id": qid, "answer": "A"}

            docs = await retriever.search(session, question, top_k=TOP_K)
            raw_ans = await generate_answer(session, question, options_text, docs)
            final_key = extract_key(raw_ans)
            
            # Log tiến độ nhẹ
            print(f"✅ Q:{qid} | Ans:{final_key} | Docs:{len(docs)} | LargeUsed:{QUOTA_MGR.large_used}")
            return {"id": qid, "answer": final_key}
            
        except Exception as e:
            print(f"❌ Error processing QID {row.get('id')}: {e}")
            return {"id": row.get('id'), "answer": "A"}

async def main():
    input_file = Config.BASE_DIR / "data" / "test.json" 

    print(f"📂 Reading JSON: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # [FIX 2] Semaphore để kiểm soát concurrency
    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    retriever = HybridRetriever()
    
    print(f"🔥 Processing {len(data)} questions with {MAX_CONCURRENT_TASKS} concurrent tasks...")
    
    connector = aiohttp.TCPConnector(force_close=True, limit=MAX_CONCURRENT_TASKS)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_row_safe(sem, session, retriever, row) for row in data]
        results = await asyncio.gather(*tasks)
    
    # [OUTPUT] Format CSV theo yêu cầu: qid, answer
    output_csv = Config.BASE_DIR / "output" / "pred.csv"
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    out_df = pd.DataFrame(results)
    # Rename cột id -> qid nếu cần thiết cho đúng format nộp
    if 'id' in out_df.columns:
        out_df.rename(columns={'id': 'qid'}, inplace=True)
        
    out_df = out_df[['qid', 'answer']] # Chỉ lấy 2 cột cần thiết
    out_df.to_csv(output_csv, index=False)
    print(f"💾 Done! Saved to {output_csv}")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

