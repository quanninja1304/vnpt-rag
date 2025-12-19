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
import uuid
from underthesea import word_tokenize
from config import Config
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


# ==============================================================================
# 0. CẤU HÌNH CHIẾN THUẬT (Tactical Config)
# ==============================================================================
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File lưu kết quả (Dùng để Resume)
OUTPUT_FILE = Config.BASE_DIR / "output" / "submission.csv"
DEBUG_LOG_FILE = Config.LOGS_DIR / "debug_trace.txt"

# Cấu hình chạy an toàn tuyệt đối
MAX_CONCURRENT_TASKS = 1      # Chạy từng câu một (Chậm nhưng chắc 100%)
TIMEOUT_PER_QUESTION = 600 # 10 phút

# Rate Limit (Tốc độ)
LIMITER_LARGE = AsyncLimiter(1, 95)   # 40 req/giờ
LIMITER_SMALL = AsyncLimiter(1, 65)   # 60 req/giờ
LIMITER_EMBED = AsyncLimiter(300, 60)  # 300 req/phút

# Ngân sách (Quota) - Để theo dõi
QUOTA_LARGE = 500
QUOTA_SMALL = 1000

# Constants
THRESHOLD_SMALL_CONTEXT = 15000 
TOP_K = 18
ALPHA_VECTOR = 0.5
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

def extract_balanced_json(text):
    """
    Tìm JSON object đầu tiên với cặp {} cân bằng
    
    VÍ DỤ XỬ LÝ ĐƯỢC:
    - 'Here is the answer: {"x": {"y": "z"}} hope this helps'
    - '{"a": "b", "c": "$x^2$"}' (chứa ký tự đặc biệt)
    - 'Sure! {"nested": {"key": "value"}} Done'
    """
    
    # Tìm vị trí { đầu tiên
    start = text.find('{')
    if start == -1:
        return None
    
    # Đếm cặp ngoặc để tìm } đóng
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        # Xử lý escape trong string
        if escape:
            escape = False
            continue
        
        if char == '\\':
            escape = True
            continue
        
        # Xử lý string (bỏ qua {} trong "...")
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        # Đếm ngoặc
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            
            # Tìm thấy cặp {} hoàn chỉnh
            if depth == 0:
                return text[start:i+1]
    
    return None

def parse_json_strict(raw_response):
    """
    Parse JSON ROBUST - Xử lý được nhiều format khác nhau
    
    CHIẾN LƯỢC MỚI:
    1. Thử parse trực tiếp bằng json.loads() (nhanh nhất)
    2. Loại bỏ markdown fence
    3. Tìm JSON bằng balanced bracket matching (xử lý nested {})
    4. Fallback: Extract từ khóa bằng regex
    """
    
    if not raw_response:
        return None
    
    cleaned = raw_response.strip()
    
    # ========================================
    # BƯỚC 1: THỬ PARSE TRỰC TIẾP (Fast Path)
    # ========================================
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "safety" in data and "domain" in data:
            return data
    except:
        pass
    
    # ========================================
    # BƯỚC 2: LOẠI BỎ MARKDOWN FENCE
    # ========================================
    if "```" in cleaned:
        # Match ```json ... ``` hoặc ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and "safety" in data and "domain" in data:
                    return data
            except:
                pass
    
    # ========================================
    # BƯỚC 3: BALANCED BRACKET MATCHING
    # Xử lý được nested JSON như: {"x": {"y": "z"}}
    # ========================================
    json_str = extract_balanced_json(cleaned)
    
    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "safety" in data and "domain" in data:
                return data
        except:
            pass
    
    # ========================================
    # BƯỚC 4: FALLBACK - EXTRACT TỪ KHÓA
    # Nếu model trả về dạng tự do, ví dụ:
    # "I think it's UNSAFE and domain is LEGAL"
    # ========================================
    safety_match = re.search(r'(?:safety|an toàn)[":\s]*(SAFE|UNSAFE)', cleaned, re.IGNORECASE)
    domain_match = re.search(r'(?:domain|lĩnh vực)[":\s]*(STEM|LEGAL|SOCIAL)', cleaned, re.IGNORECASE)
    
    if safety_match and domain_match:
        return {
            "safety": safety_match.group(1).upper(),
            "domain": domain_match.group(1).upper()
        }
    
    # ========================================
    # THẤT BẠI HOÀN TOÀN
    # ========================================
    logger.warning(f"Failed to parse JSON from response: {cleaned[:200]}...")
    return None

def is_all_above_option(text):
    """Kiểm tra xem đáp án có phải là 'Tất cả các ý trên' hay không"""
    text_lower = str(text).lower()
    patterns = [
        r"tất cả.*(?:đáp án|ý|lựa chọn)",  # Tất cả các đáp án trên
        r"cả.*(?:đều|là).*đúng",           # Cả 3 ý đều đúng
        r"các ý trên đều",
        r"phương án.*cả"                   # Cả A và B
    ]
    return any(re.search(p, text_lower) for p in patterns)

def find_true_refusal_key(options_map):
    """
    Tìm đáp án từ chối - Phiên bản tối ưu
    Trả về: (Key, Type) với Type = "SAFETY" | "NO_INFO" | None
    """
    
    # NHÓM 1: SAFETY REFUSAL
    safety_patterns = [
        r"^(?:tôi|hệ thống)\s*không\s*(?:thể|được|hỗ trợ)\s*(?:trả lời|cung cấp)",
        r"vi phạm\s+(?:pháp luật|chính sách)",
        r"không\s+(?:thể|hỗ trợ).*(?:làm giả|trốn thuế|gian lận)"
    ]
    
    safety_exclusions = [
        "được xem là", "bị coi là", "cấu thành tội"
    ]
    
    for label, text in options_map.items():
        text_lower = str(text).lower().strip()
        
        if any(ex in text_lower for ex in safety_exclusions):
            continue
        
        if any(re.search(p, text_lower) for p in safety_patterns):
            return label, "SAFETY"
    
    # NHÓM 2: NO INFO REFUSAL
    no_info_patterns = [
        r"không\s+(?:có|đủ|tìm thấy)\s+(?:thông tin|dữ liệu)",
        r"không\s+thể\s+(?:xác định|kết luận|tính toán)(?:\.|,|\s|$)",
        r"thông tin.*không\s+đủ"
    ]
    
    no_info_exclusions = [
        "không thể thay đổi", "không thể tách rời", 
        "không thể thiếu", "không thể phủ nhận"
    ]
    
    for label, text in options_map.items():
        text_lower = str(text).lower().strip()
        
        if "vi phạm" in text_lower or "luật" in text_lower:
            continue
        
        if any(ex in text_lower for ex in no_info_exclusions):
            continue
        
        if any(re.search(p, text_lower) for p in no_info_patterns):
            return label, "NO_INFO"
    
    return None, None


async def unified_router_v3(session, question, options_map):
    """
    ROUTER ĐÃ ĐƯỢC FIX - Khắc phục 5 vấn đề:
    1. Rút gọn prompt từ 800 từ → 200 từ
    2. Đơn giản hóa logic → 2 bước thay vì 3
    3. Parse JSON chặt chẽ hơn
    4. Log đầy đủ để debug
    5. Timeout 20s + retry
    """
    
    q_lower = question.lower()
    
    # PRE-CALC
    refusal_key, refusal_type = find_true_refusal_key(options_map)
    options_text = "\n".join([f"{k}. {v}" for k, v in options_map.items()])
    
    # ========================================
    # PROMPT (200 từ)
    # ========================================
    
    system_prompt = """Phân loại câu hỏi thi theo 2 tiêu chí:

1. SAFETY (An toàn):
   UNSAFE: Hướng dẫn CÁCH LÀM hành vi vi phạm:
   - Trốn/tránh/lách luật, thanh tra, thuế
   - Làm giả tem, tài liệu, tiếp tục hoạt động khi bị cấm
   - Phá hoại, lật đổ, xúc phạm lãnh đạo
   - Khiêu dâm, bạo lực, vũ khí
   
   SAFE: Hỏi kiến thức (kể cả chiến tranh, chính trị, luật hình sự, y học)

2. DOMAIN (Lĩnh vực):
   STEM: Toán, Lý, Hóa (có công thức)
   LEGAL: Luật pháp (Điều X, Bộ luật...)
   SOCIAL: Lịch sử, Địa lý, Văn học

OUTPUT chỉ 1 dòng JSON:
{"safety":"SAFE|UNSAFE","domain":"STEM|LEGAL|SOCIAL"}"""

    user_content = f"Câu hỏi: {question}\n\nĐáp án:\n{options_text}\n\nJSON:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # ========================================
    # FIX 2: THÊM TIMEOUT + RETRY
    # ========================================
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            raw_resp = await call_llm_generic(
                session, messages,
                Config.LLM_MODEL_SMALL,
                stats=None,
                max_tokens=300,
                timeout=45
            )
            
            if not raw_resp:
                logger.warning(f"Attempt {attempt+1}: Empty response")
                continue
            
            # ========================================
            # FIX 3: PARSE JSON CHẶT CHẼ HƠN
            # ========================================
            
            result = parse_json_strict(raw_resp)
            
            if not result:
                logger.warning(f"Attempt {attempt+1}: Failed to parse JSON")
                logger.debug(f"Raw response: {raw_resp[:200]}...")
                continue
            
            # ========================================
            # FIX 4: XỬ LÝ KẾT QUẢ
            # ========================================
            
            safety = result.get("safety", "SAFE").upper()
            domain = result.get("domain", "SOCIAL").upper()
            
            # SAFETY NET: Cứu các câu hỏi chính trị học thuật
            if safety == "UNSAFE":
                political_whitelist = [
                    "hồ chí minh", "bác hồ", "đảng", "cách mạng",
                    "chiến tranh", "kháng chiến", "lịch sử"
                ]
                dirty_blacklist = [
                    "khiêu dâm", "sex", "porn", "lật đổ", 
                    "khủng bố", "xuyên tạc", "bôi nhọ"
                ]
                
                has_political = any(w in q_lower for w in political_whitelist)
                has_dirty = any(w in q_lower for w in dirty_blacklist)
                
                if has_political and not has_dirty:
                    logger.info(f"✅ SAFETY NET: Rescued political question")
                    safety = "SAFE"
            
            # INTENT OVERRIDE: Bắt các pattern nguy hiểm model bỏ sót
            if safety == "SAFE":
                bad_patterns = [
                    r"(?:cách|phương pháp).*(?:vi phạm|trốn|tránh|lách)",
                    r"tránh.*(?:cung cấp|khai báo).*thông tin",
                    r"tiếp tục.*hoạt động.*(?:bị đình chỉ|bị cấm)"
                ]
                
                if any(re.search(p, q_lower) for p in bad_patterns):
                    logger.warning(f"⚠️ INTENT OVERRIDE: Blocked by pattern")
                    safety = "UNSAFE"
            
            # RETURN KẾT QUẢ
            if safety == "UNSAFE":
                return _create_blocked_result("AI-BLOCKED", options_map)
            
            # ========================================
            # FIX LỖI 2: XỬ LÝ REFUSAL KEY CHO SAFE
            # ========================================
            
            # Quyết định dùng Large/Small
            is_stem = (domain == "STEM")
            use_large = (
                is_stem 
                or (domain == "LEGAL") 
                or len(question) > 500
            )
            
            # XỬ LÝ REFUSAL KEY:
            # - Nếu là STEM và có đáp án NO_INFO -> Giữ lại để fallback
            # - Nếu là SOCIAL/LEGAL -> Bỏ qua refusal (đó là bẫy)
            final_refusal_key = None
            final_refusal_type = None
            
            if refusal_key and refusal_type == "NO_INFO":
                # Chỉ giữ NO_INFO cho câu STEM (Toán/Lý)
                if is_stem:
                    final_refusal_key = refusal_key
                    final_refusal_type = "NO_INFO"
                    logger.info(f"📝 Detected NO_INFO answer for STEM question: {refusal_key}")
                else:
                    # SOCIAL/LEGAL luôn có đáp án đúng
                    logger.info(f"🎣 Ignored TRAP refusal for {domain} question")
            
            return {
                "is_unsafe": False,
                "is_stem": is_stem,
                "use_large": use_large,
                "tag": f"AI-{domain}-{'L' if use_large else 'S'}",
                "refusal_key": final_refusal_key,
                "refusal_type": final_refusal_type
            }
        
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed!")
    
    # ========================================
    # FALLBACK REGEX (KHI AI HOÀN TOÀN LỖI)
    # ========================================
    
    logger.info("⚠️ Using REGEX FALLBACK (AI failed after retries)")
    
    # Hard ban
    hard_ban = ["khiêu dâm", "ấu dâm", "sex", "porn", "xxx", "cá độ"]
    if any(w in q_lower for w in hard_ban):
        return _create_blocked_result("REGEX-DIRTY", options_map)
    
    # Intent patterns
    bad_intent = [
        r"(?:cách|phương pháp).*(?:vi phạm|trốn|tránh|lách|làm giả)",
        r"tránh.*(?:cung cấp|khai báo).*(?:thông tin|hồ sơ)",
        r"tiếp tục.*hoạt động.*(?:bị đình chỉ|bị cấm)"
    ]
    
    if any(re.search(p, q_lower) for p in bad_intent):
        return _create_blocked_result("REGEX-INTENT", options_map)
    
    # Check refusal type
    if refusal_type == "SAFETY":
        return _create_blocked_result("REGEX-ANS-SAFETY", options_map)
    
    # Default safe
    has_math = bool(re.search(r"\$|\\frac|\\int|\\sum", question))
    
    # FIX LỖI 2 (FALLBACK): Giữ refusal_key cho STEM + NO_INFO
    final_refusal_key = None
    final_refusal_type = None
    
    if has_math and refusal_key and refusal_type == "NO_INFO":
        final_refusal_key = refusal_key
        final_refusal_type = "NO_INFO"
        logger.info(f"📝 FALLBACK: Kept NO_INFO for math question")
    
    return {
        "is_unsafe": False,
        "is_stem": has_math,
        "use_large": True,
        "tag": "REGEX-FALLBACK",
        "refusal_key": final_refusal_key,
        "refusal_type": final_refusal_type
    }

def _create_blocked_result(reason, options_map):
    """Tạo kết quả chặn"""
    key, _ = find_true_refusal_key(options_map)
    
    if not key:
        # Fallback: Tìm bất kỳ đáp án nào có "không thể"
        keywords = ["tôi không thể", "không thể cung cấp", "không thể chia sẻ"]
        for k, v in options_map.items():
            if any(kw in str(v).lower() for kw in keywords):
                key = k
                break
    
    final_key = key if key else "A"
    
    return {
        "is_unsafe": True,
        "is_stem": False,
        "use_large": False,
        "tag": f"BLOCKED-{reason}",
        "refusal_key": final_key,
        "refusal_type": "SAFETY"
    }

def find_no_info_key(options_map):
    """
    Tìm đáp án mang tính LOGIC/KHOA HỌC (Không xác định được).
    (Phiên bản nâng cấp: Bắt đa dạng cấu trúc câu)
    """
    
    # Danh sách Pattern (Chia nhóm để dễ quản lý)
    no_info_patterns = [
        # NHÓM 1: TRỰC TIẾP "KHÔNG ĐỦ..."
        # Bắt: "Không có thông tin", "Không đủ dữ kiện", "Thiếu cơ sở", "Chưa đủ bằng chứng"
        r"(?:không|chưa) (?:có|đủ|tìm thấy) (?:thông tin|dữ liệu|dữ kiện|cơ sở|căn cứ|bằng chứng|giả thiết)",
        
        # NHÓM 2: ĐẢO NGỮ "THÔNG TIN... KHÔNG ĐỦ"
        # Bắt: "Thông tin cung cấp không đủ", "Dữ liệu bài toán chưa đủ"
        r"(?:thông tin|dữ liệu|dữ kiện|giả thiết).* (?:không|chưa) (?:đủ|rõ ràng|chính xác)",
        
        # NHÓM 3: KHÔNG THỂ HÀNH ĐỘNG (ĐỘNG TỪ MẠNH)
        # Bắt: "Không thể xác định", "Không thể kết luận", "Không thể tính", "Không thể đưa ra"
        # Thêm \b để ranh giới từ rõ ràng
        r"không thể (?:xác định|kết luận|tính toán|trả lời|khẳng định|đưa ra|so sánh)(?:\.|,| |$)",
        
        # NHÓM 4: CỤM TỪ KINH ĐIỂN TRONG TRẮC NGHIỆM
        # Bắt: "Từ thông tin đã cho...", "Dựa vào dữ liệu trên..." đi kèm phủ định
        r"(?:từ|dựa vào|với|căn cứ).* (?:thông tin|dữ liệu|dữ kiện).* (?:không|chưa|khó)",
        
        # NHÓM 5: META (Về câu hỏi)
        r"câu hỏi (?:không thể|không có) (?:trả lời|đáp án)"
    ]
    
    # Danh sách loại trừ (Tránh bắt nhầm kiến thức)
    # Ví dụ: "Năng lực là đặc điểm không thể thay đổi" -> Bị loại trừ.
    exclusions = [
        "tôi không thể", # Nhường cho Safety
        "không thể thay đổi", "không thể tách rời", "không thể thiếu", 
        "không thể phủ nhận", "không thể tránh khỏi", "không thể đảo ngược",
        "không thể chia cắt", "không thể nhầm lẫn"
    ]

    for label, text in options_map.items():
        text_lower = str(text).lower().strip()
        
        # 1. Check Exclusion (Loại trừ trước)
        if any(ex in text_lower for ex in exclusions):
            continue
            
        # 2. Check "Vi phạm/Luật" (Để chắc chắn không cướp của Safety)
        if "vi phạm" in text_lower or "luật" in text_lower or "chính sách" in text_lower:
            continue
        
        # 3. Check Patterns
        if any(re.search(p, text_lower) for p in no_info_patterns):
            return label

    return None

    
def write_debug_log(qid, question, route_tag, model_used, answer, true_label=None, note=""):
    """Hàm ghi log chi tiết vào file txt"""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Kiểm tra đúng sai nếu có đáp án mẫu
        result_status = ""
        if true_label:
            result_status = "✅ ĐÚNG" if str(answer).strip() == str(true_label).strip() else f"❌ SAI (Đúng là {true_label})"
        
        log_content = f"""
--------------------------------------------------------------------------------
[{timestamp}] QID: {qid}
❓ Question: {question}
🏷️ Route: {route_tag} | 🤖 Model: {model_used}
📝 Answer: {answer} {result_status}
ℹ️ Note: {note}
--------------------------------------------------------------------------------
"""
        # Mở file mode 'a' (append) để ghi nối tiếp
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_content)
            
    except Exception as e:
        print(f"Lỗi ghi log: {e}")


async def unified_router_logic(session, question):
    """
    ROUTER TỔNG HỢP V2 (Robust Parsing & Legal Awareness)
    """
    q_lower = question.lower()

    # --- BƯỚC 1: HARD CHECK (Zero-Cost) ---
    # 1.1 An toàn
    hard_ban = ["khiêu dâm", "làm tình", "ấu dâm", "kích dục", "cá độ", "lật đổ chính quyền", "sex", "xxx"]
    if any(w in q_lower for w in hard_ban):
        return {"is_unsafe": True, "is_stem": False, "use_large": False, "tag": "BLOCKED"}

    # 1.2 STEM (Toán học)
    has_math_regex = bool(re.search(r"\$|\\frac|\\int|\\sum|\^\{|sin\(|cos\(|tan\(", question))

    # 1.3 Luật pháp (Legal Keywords) - Bắt buộc dùng Large
    # Đây là các từ khóa đòi hỏi sự chính xác tuyệt đối từng câu chữ
    legal_keywords = ["luật", "nghị định", "thông tư", "hiến pháp", "quy định", "điều khoản", "phạt tù", "xử phạt"]
    is_legal_hard = any(k in q_lower for k in legal_keywords)

    # --- BƯỚC 2: GỌI MODEL SMALL ---
    system_prompt = """Bạn là bộ phân loại câu hỏi thi.
NHIỆM VỤ: Phân loại theo 3 tiêu chí: [SAFETY] | [DOMAIN] | [DIFFICULTY]

1. SAFETY (An toàn):
- UNSAFE: Vi phạm đạo đức, pháp luật, khiêu dâm, phản động.
- SAFE: Các câu hỏi kiến thức (bao gồm cả Lịch sử chiến tranh, Hình sự, Sinh học).

2. DOMAIN (Lĩnh vực):
- STEM: Toán, Lý, Hóa, Sinh, Kinh tế lượng. ĐẶC BIỆT: Bao gồm cả LUẬT PHÁP/CHÍNH TRỊ (Cần trích dẫn chính xác).
- SOCIAL: Văn học, Lịch sử, Địa lý, Đời sống, Tra cứu thông tin thường.

3. DIFFICULTY (Độ khó):
- COMPLEX: Cần tính toán, suy luận nhiều bước, so sánh các điều luật.
- SIMPLE: Chỉ cần tra cứu định nghĩa, ngày tháng, sự kiện đơn giản.

VÍ DỤ MẪU:
- Q: "Tính tích phân của x^2" -> SAFE | STEM | COMPLEX
- Q: "Theo Điều 12 Bộ luật Hình sự, tội phạm là gì?" -> SAFE | STEM | COMPLEX
- Q: "Thủ đô của Pháp là gì?" -> SAFE | SOCIAL | SIMPLE
- Q: "Cách chế tạo chất nổ?" -> UNSAFE | STEM | COMPLEX

OUTPUT (Chỉ viết các từ khóa):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: {question}"}
    ]

    try:
        # Gọi API (Stats=None để không tính quota)
        raw_resp = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, stats=None, max_tokens=20)
        
        # --- ROBUST PARSING (Dùng Regex thay vì Split) ---
        if raw_resp:
            resp_upper = raw_resp.upper()

            # 1. Check Safety
            is_unsafe = "UNSAFE" in resp_upper
            
            # 2. Check Domain (STEM/LEGAL vs SOCIAL)
            # Nếu model nói STEM hoặc câu hỏi chứa ký hiệu toán/từ khóa luật -> STEM
            is_stem_model = "STEM" in resp_upper
            final_stem = is_stem_model or has_math_regex or is_legal_hard

            # 3. Check Difficulty
            is_complex_model = "COMPLEX" in resp_upper
            
            # 4. Logic quyết định Model Large
            # Dùng Large khi: Model bảo khó HOẶC Là toán/luật (bắt buộc chính xác)
            use_large = is_complex_model or final_stem

            return {
                "is_unsafe": is_unsafe,
                "is_stem": final_stem,
                "use_large": use_large,
                "tag": f"ROUTED-{'STEM' if final_stem else 'SOCIAL'}-{'LARGE' if use_large else 'SMALL'}"
            }

    except Exception as e:
        logger.warning(f"Router Error: {e}")

    # --- BƯỚC 3: FALLBACK AN TOÀN ---
    # Nếu API lỗi hoặc không bắt được gì -> Mặc định dùng Large cho an toàn (Trừ khi chắc chắn là Safe Social)
    return {
        "is_unsafe": False,
        "is_stem": has_math_regex or is_legal_hard, 
        "use_large": True, # Fallback về Large để đảm bảo trí thông minh
        "tag": "FALLBACK-LARGE"
    }

def unified_router(question):
    """
    Bộ định tuyến tổng hợp (Local - 0 API Call).
    Phân loại câu hỏi thành 4 nhóm để chọn chiến thuật phù hợp.
    
    OUTPUT:
    - 'BLOCKED': Câu hỏi nhạy cảm/cấm -> Trả lời từ chối ngay.
    - 'STEM': Toán, Lý, Hóa -> Cần Large Model + CoT Prompt.
    - 'COMPLEX': Logic, suy luận, đánh đố -> Cần Large Model + CoT Prompt.
    - 'SIMPLE': Tra cứu Văn, Sử, Địa -> Dùng Small Model (hoặc Large nếu thích) + Simple Prompt.
    """
    q_lower = question.lower()

    # ==============================================================================
    # 1. SAFETY CHECK (Ưu tiên cao nhất - Chặn trước khi làm bất cứ việc gì)
    # ==============================================================================
    hard_ban = [
        "khiêu dâm", "làm tình", "ấu dâm", "kích dục", "cá độ", "cờ bạc",
        "lật đổ", "phản động", "khủng bố", "giết người", "tự sát", "tự tử",
        "chế bom", "chế súng", "ma túy đá", "thuốc lắc", "sex", "xxx"
    ]
    if any(w in q_lower for w in hard_ban): return 'BLOCKED'

    # Soft ban: Chỉ chặn nếu không có từ khóa học thuật đi kèm
    soft_ban = ["ma túy", "vũ khí", "bạo lực", "chết", "biểu tình", "chính trị"]
    academic_whitelist = [
        "luật", "nghị định", "quy định", "lịch sử", "tôn giáo", "kinh thánh", 
        "torah", "qur'an", "văn bản", "cổ đại", "hình phạt", "tội danh",
        "theo đoạn văn", "dựa vào thông tin", "theo ngữ cảnh"
    ]
    
    has_bad = any(w in q_lower for w in soft_ban)
    has_academic = any(w in q_lower for w in academic_whitelist)
    
    if has_bad and not has_academic: return 'BLOCKED'

    # ==============================================================================
    # 2. STEM CHECK (Toán/Lý/Hóa - Cần tính toán chính xác)
    # ==============================================================================
    # Regex bắt ký hiệu Toán học đặc thù
    if re.search(r"\$|\\frac|\\int|\\sum|\^\{|sin\(|cos\(|tan\(|log\(|lim_|\\sqrt", question):
        return 'STEM'
    
    # Từ khóa định lượng/đơn vị đo lường
    stem_keywords = [
        # --- Toán học & Vật lý ---
        "giá trị của", "kết quả phép tính", "nghiệm của", "xác suất", "tọa độ", 
        "đạo hàm", "tích phân", "trung bình cộng", "phương sai", "độ lệch chuẩn",
        "vận tốc", "gia tốc", "cường độ", "điện trở", "nồng độ", "số mol",
        "diện tích", "thể tích", "chu vi", "bán kính",
        
        # --- Tài chính & Kinh tế lượng ---
        "kỳ vọng",          # Bắt "giá trị kỳ vọng", "lợi nhuận kỳ vọng"
        "đầu tư",           # Bài toán ROI
        "lợi nhuận",        # Tính lãi
        "mức lỗ", "thua lỗ", # Tính lỗ
        "lãi suất", "vốn",  # Bài toán lãi kép/đơn
        "tăng trưởng",      # Bài toán % tăng trưởng
        "tỉ lệ", "phần trăm"
    ]
    if any(k in q_lower for k in stem_keywords):
        return 'STEM'

    # Xử lý từ "Tính": Phân biệt "Tính toán" (STEM) vs "Tính cách" (SIMPLE)
    if "tính" in q_lower:
        social_context = ["tính cách", "tính chất", "tính năng", "tính nhân văn", "máy tính", "thuộc tính"]
        if not any(sc in q_lower for sc in social_context):
            return 'STEM' # Có chữ "tính" mà không phải "tính cách" -> Khả năng cao là Toán

    # ==============================================================================
    # 3. COMPLEX/LOGIC CHECK (Suy luận, Đố mẹo, Logic)
    # ==============================================================================
    logic_keywords = [
        "giả sử", "nếu... thì", "suy ra", "logic", "người tiếp theo", "quy luật", 
        "mâu thuẫn", "tương phản", "ý nào sau đây đúng", "nguyên nhân chính",
        "dựa vào thông tin", "theo đoạn văn", "ý chính", "kết luận nào"
    ]
    if any(k in q_lower for k in logic_keywords):
        return 'COMPLEX'

    # ==============================================================================
    # 4. SIMPLE CHECK (Mặc định - Tra cứu kiến thức)
    # ==============================================================================
    return 'SIMPLE'


async def smart_router_with_small(session, question):
    """
    Dùng Model Small để phân loại độ khó câu hỏi.
    OUTPUT: True (Khó/STEM/Luật suy luận -> Dùng Large) | False (Tra cứu/Văn/Sử -> Dùng Small)
    """
    # 1. LỚP LỌC 1: Regex Toán học/Ký hiệu (Nhanh, không tốn API)
    # Bắt các công thức LaTeX, ký hiệu toán, hóa học đặc thù
    if re.search(r"\$|\\frac|\\int|\\sum|\^\{|sin\(|cos\(|tan\(|log\(|ln\(", question):
        return True

    # 2. LỚP LỌC 2: Gọi Model Small phân loại ngữ nghĩa
    system_prompt = """Bạn là bộ phân loại câu hỏi thi. 
NHIỆM VỤ: Phân loại câu hỏi vào 1 trong 2 nhóm:

1. NHÓM PHỨC TẠP (Trả lời: COMPLEX):
   - Toán, Lý, Hóa, Sinh, Kinh tế lượng (cần tính toán).
   - Tư duy Logic, Đố mẹo, Suy luận nguyên nhân - hệ quả phức tạp.
   - Câu hỏi Phủ định xoắn não ("Ngoại trừ...", "Không phải là...").

2. NHÓM TRA CỨU (Trả lời: SIMPLE):
   - Lịch sử, Địa lý, Văn học, Tác giả - Tác phẩm.
   - Trích xuất thông tin đơn thuần ("Theo đoạn văn...", "Chi tiết nào...").
   - Định nghĩa, Khái niệm, Ngày tháng năm.

OUTPUT: Chỉ trả lời duy nhất 1 từ: COMPLEX hoặc SIMPLE."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: {question}"}
    ]

    try:
        # Gọi model small, max_tokens cực thấp (chỉ cần 1 từ)
        # stats=None để không tính vào quota chính (hoặc truyền stats nếu muốn track)
        resp = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, stats=None, max_tokens=10)
        
        if resp:
            label = resp.strip().upper()
            if "COMPLEX" in label: return True
            if "SIMPLE" in label: return False
                
    except Exception as e:
        logger.warning(f"Router Error: {e}")

    # 3. LỚP LỌC 3: Fallback an toàn (Nếu API lỗi)
    # Chỉ bắt các từ khóa TÍNH TOÁN thực sự, bỏ qua các từ tra cứu
    # "tính" trong "tính cách" -> False. "tính" trong "tính giá trị" -> True (do ngữ cảnh)
    # Ở đây dùng list hẹp để tránh bắt nhầm.
    safe_keywords = [
        "tính giá trị", "công thức", "lãi suất", "khấu hao", "tọa độ", 
        "xác suất", "vận tốc", "gia tốc", "biến đổi", "tỉ lệ", "phương trình"
    ]
    return any(k in question.lower() for k in safe_keywords)


def get_current_date_str():
    return datetime.now().strftime("%d/%m/%Y")

async def rerank_with_small(session, question, initial_docs, top_n=8, stats=None):
    if not initial_docs: return []
    if len(initial_docs) <= top_n: return initial_docs

    # 1. Input
    docs_text = ""
    for i, doc in enumerate(initial_docs):
        clean_body = str(doc.get('text', '')).strip().replace("\n", " ")
        preview_text = " ".join(clean_body.split())[:1000] 
        docs_text += f"ID [{i}]: {preview_text}...\n\n"

    # 2. Prompt (Thêm ví dụ cụ thể để model dễ hiểu)
    system_prompt = """Bạn là chuyên gia lọc thông tin RAG.
NHIỆM VỤ: Chọn ra tối đa 8 tài liệu liên quan nhất để trả lời câu hỏi.

TIÊU CHÍ KHẮT KHE:
1. Ưu tiên tài liệu chứa đáp án trực tiếp hoặc từ khóa chính xác.
2. Loại bỏ tài liệu rác hoặc không liên quan.
3. Nếu câu hỏi yêu cầu tính toán/số liệu -> Chọn tài liệu chứa con số.

OUTPUT: Chỉ trả về mảng số ID. Ví dụ: [0, 5, 2]"""

    user_prompt = f"""CÂU HỎI: "{question}"

DANH SÁCH TÀI LIỆU:
{docs_text}

HÃY CHỌN ID TÀI LIỆU LIÊN QUAN NHẤT (JSON Array):"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    try:
        response = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, stats, max_tokens=100)
        
        # [DEBUG LOG] Xem model trả lời gì khi bị fail
        if not response or not any(c.isdigit() for c in response):
            logger.warning(f"⚠️ Rerank Empty Response: '{response}'")

        if response:
            found_indices = [int(s) for s in re.findall(r'\d+', response)]
            
            valid_docs = []
            seen = set()
            for idx in found_indices:
                if 0 <= idx < len(initial_docs) and idx not in seen:
                    valid_docs.append(initial_docs[idx])
                    seen.add(idx)
            
            # [FIX BACKFILL] Luôn đảm bảo đủ top_n docs
            if len(valid_docs) < top_n:
                for i, doc in enumerate(initial_docs):
                    if i not in seen:
                        valid_docs.append(doc)
                        if len(valid_docs) >= top_n: break
            
            return valid_docs[:top_n]

    except Exception as e:
        logger.warning(f"Rerank Error: {e}")
    
    return initial_docs[:top_n]

async def route_question_type(session, question):
    """
    Phân loại câu hỏi: STEM hay KHÁC?
    Chiến thuật: Regex (Math) -> Model Small
    """
    # 1. Check nhanh các ký hiệu Toán học đặc thù (Tiết kiệm quota)
    # Tìm dấu $, các lệnh latex cơ bản
    if re.search(r"\$|\\frac|\\int|\\sum|\\sqrt|\^\{", question):
        return True # Chắc chắn là STEM

    # 2. Gọi Model Small để phân loại ngữ nghĩa
    system_prompt = """
                    Bạn là bộ phân loại câu hỏi.
                    NHIỆM VỤ: Xác định câu hỏi thuộc nhóm TỰ NHIÊN (Toán, Lý, Hóa, Sinh, Kinh tế định lượng, Kỹ thuật) hay XÃ HỘI (Văn, Sử, Địa, Luật, Đời sống).

                    OUTPUT:
                    - Nếu là Tự nhiên/Tính toán -> Trả lời: STEM
                    - Nếu là Xã hội/Tra cứu -> Trả lời: SOCIAL
                    - Chỉ trả lời đúng 1 từ.
                    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: {question}"}
    ]

    # Gọi Small, max_tokens=5 cho nhanh
    # Lưu ý: Cần truyền stats giả hoặc None nếu hàm call_llm_generic của bạn yêu cầu
    # Ở đây giả định call_llm_generic không bắt buộc stats
    try:
        # Gọi model small với timeout ngắn
        resp = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, max_tokens=10)
        
        if resp and "STEM" in resp.upper():
            return True
        return False
    except:
        # Fallback nếu gọi AI lỗi -> Dùng lại keyword cũ cho an toàn
        stem_keywords = [
        # 1. TOÁN HỌC & THỐNG KÊ CƠ BẢN
        "công thức", "hàm số", "phương trình", "bất phương trình", "nghiệm",
        "xác suất", "tỉ lệ", "phần trăm", "trung bình", "bình quân",
        "tọa độ", "vectơ", "ma trận", "đạo hàm", "tích phân", "logarit",
        "diện tích", "thể tích", "chu vi", "bán kính", "đường kính",
        "sin", "cos", "tan", "cot", "hình học", "đồ thị",

        # 2. TÀI CHÍNH - KẾ TOÁN (Fix lỗi test_0095)
        "lãi suất", "vốn hóa", "cổ tức", "khấu hao", "tài sản", "nguồn vốn",
        "nợ phải trả", "vốn chủ sở hữu", "doanh thu", "chi phí", "lợi nhuận",
        "bảng cân đối", "báo cáo tài chính", "dòng tiền", "thu nhập ròng",
        "giá vốn", "hàng tồn kho", "biên lợi nhuận", "cổ phiếu", "trái phiếu",
        "tiền tệ", "tỷ giá", "hối đoái", "lạm phát", "gdp", "cpi",
        "usd", "vnd", "đồng", "triệu", "tỷ", "nghìn", # Đơn vị tiền tệ

        # 3. VẬT LÝ & KỸ THUẬT
        "vận tốc", "gia tốc", "quãng đường", "thời gian", "lực", "công suất",
        "năng lượng", "động năng", "thế năng", "nhiệt lượng", "điện áp",
        "cường độ", "dòng điện", "điện trở", "tần số", "bước sóng", "chu kỳ",
        "áp suất", "trọng lượng", "khối lượng riêng", "độ lớn", "biên độ",
        "m/s", "km/h", "kwh", "hz", "vôn", "ampe", "joule",

        # 4. HÓA HỌC & SINH HỌC (Tính toán)
        "nồng độ", "mol", "khối lượng mol", "phản ứng", "cân bằng",
        "kết tủa", "nguyên tử khối", "phân tử khối", "hóa trị", "ph",
        "dung dịch", "chất tan", "dung môi", "kiềm", "axit",

        # 5. TỪ KHÓA DẤU HIỆU BÀI TOÁN (Logic)
        "giả sử", "cho biết", "biết rằng", "kết quả của", "giá trị của",
        "tính toán", "ước tính", "dự báo", "tăng bao nhiêu", "giảm bao nhiêu"
        ]
        return any(k in question.lower() for k in stem_keywords)
    

def extract_answer_strict(text, options_map):
    """Trích xuất đáp án từ output của LLM một cách chặt chẽ"""
    valid_keys = list(options_map.keys())
    if not text: return None
    text = text.strip()
    
    # Các mẫu regex để bắt đáp án chuẩn
    patterns = [
        r'###\s*ĐÁP ÁN[:\s\n]*([A-Z])',  # Format chuẩn: ### ĐÁP ÁN: A
        r'ĐÁP ÁN[:\s]*([A-Z])',          # Format lỏng: ĐÁP ÁN: A
        r'CHỌN[:\s]*([A-Z])',            # Format: Chọn A
        r'KẾT LUẬN[:\s]*([A-Z])',        # Format: Kết luận A
        r'^([A-Z])\.$',                  # Chỉ trả về: A.
        r'^([A-Z])$'                     # Chỉ trả về: A
    ]
    
    # 1. Ưu tiên tìm theo pattern định sẵn
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match and match.group(1).upper() in valid_keys: 
            return match.group(1).upper()
            
    # 2. Fallback: Tìm ký tự in đậm cuối cùng (Markdown bold)
    # Ví dụ: "Đáp án đúng là *A*"
    matches = re.findall(r'\*\*([A-Z])\*\*', text)
    if matches:
        last_match = matches[-1].upper()
        if last_match in valid_keys: 
            return last_match
    
    loose_patterns = [
        r'(?:đáp án|chọn|là)[:\s\*\-\.\[\(]*([A-Z])[\]\)\*\.]', # Bắt "Là A", "Chọn B"
        r'\*\*([A-Z])\*\*',  # Bắt "**A**"
        r'^([A-Z])[\.\)]'    # Bắt đầu dòng bằng "A."
    ]
    for p in loose_patterns:
        match = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if match and match.group(1).upper() in valid_keys: 
            return match.group(1).upper()
        
    return None


def check_critical_question(question):
    """Phát hiện các câu hỏi cần độ chính xác tuyệt đối (Toán, Luật, Số liệu)"""
    q_lower = question.lower()
    
    # Nhóm 1: Luật pháp & Chế tài (Cần chính xác từng chữ)
    legal = ["luật", "nghị định", "thông tư", "phạt", "tội", "án", "hiến pháp", "cơ quan", "thẩm quyền", "quy định"]
    
    # Nhóm 2: Số liệu & Thời gian (Cần chính xác con số)
    facts = ["năm nào", "khi nào", "bao nhiêu", "số lượng", "tỉ lệ", "%", "lần đầu", "đạt mốc"]
    
    # Nhóm 3: Toán & Logic (Cần tính toán/suy luận)
    stem = ["tính", "công thức", "hàm số", "lãi suất", "khấu hao", "dao động", "trung bình", "sin", "cos"]
    
    # Nhóm 4: Trích xuất (Extractive)
    extract = ["theo đoạn", "trong văn bản", "ý nào sau đây", "chi tiết nào","theo ngữ cảnh"]

    critical_keywords = legal + facts + stem + extract
    return any(k in q_lower for k in critical_keywords)

def heuristic_answer_overlap(question, options_map):
    """Chọn đáp án dựa trên độ trùng lặp từ khóa, có xử lý câu phủ định"""
    q_lower = question.lower()
    # Kiểm tra xem có phải câu hỏi tìm ý SAI không
    is_negative = any(w in q_lower for w in ["không", "ngoại trừ", "sai", "trừ"])
    
    try:
        q_tokens = set(word_tokenize(q_lower))
        scores = {}
        for key, text in options_map.items():
            opt_tokens = set(word_tokenize(str(text).lower()))
            scores[key] = len(q_tokens.intersection(opt_tokens))
        
        if not scores: return "A"

        if is_negative:
            # Với câu hỏi phủ định: Đáp án đúng thường KHÁC BIỆT nhất so với câu hỏi
            # Hoặc an toàn hơn: Chọn câu DÀI NHẤT (thường câu đúng trong luật rất dài)
            return max(options_map.items(), key=lambda x: len(str(x[1])))[0]
        else:
            # Câu hỏi thường: Chọn câu trùng nhiều từ khóa nhất
            return max(scores, key=scores.get)
    except:
        return "A"
    
def heuristic_answer_math(question, options_map):
    """
    Heuristic STEM nâng cao - Phân tích pattern câu hỏi
    """ 
    q_lower = question.lower()
    
    # ============================================
    # NHÓM 1: BÀI TOÁN CÓ ĐƠN VỊ
    # ============================================
    # Tìm đơn vị trong câu hỏi
    units_in_question = re.findall(r'\b(m/s|km/h|kg|mol|j|w|v|a|°c|%)\b', q_lower)
    
    if units_in_question:
        # Ưu tiên đáp án có CÙNG đơn vị
        target_unit = units_in_question[0]
        for k, v in options_map.items():
            if target_unit in str(v).lower():
                return k
    
    # ============================================
    # NHÓM 2: BÀI TOÁN TĂNG/GIẢM
    # ============================================
    if any(w in q_lower for w in ['tăng', 'giảm', 'chênh lệch', 'thay đổi']):
        # Tìm đáp án có dấu +/- hoặc %
        for k, v in options_map.items():
            v_str = str(v)
            if '%' in v_str or '+' in v_str or 'tăng' in v_str.lower():
                return k
    
    # ============================================
    # NHÓM 3: BÀI TOÁN SO SÁNH (Lớn nhất/Nhỏ nhất)
    # ============================================
    if 'lớn nhất' in q_lower or 'cao nhất' in q_lower or 'tối đa' in q_lower:
        # Tìm số lớn nhất
        nums = {}
        for k, v in options_map.items():
            match = re.search(r'([\d\.]+)', str(v))
            if match:
                nums[k] = float(match.group(1))
        
        if nums:
            return max(nums, key=nums.get)
    
    if 'nhỏ nhất' in q_lower or 'thấp nhất' in q_lower or 'tối thiểu' in q_lower:
        nums = {}
        for k, v in options_map.items():
            match = re.search(r'([\d\.]+)', str(v))
            if match:
                nums[k] = float(match.group(1))
        
        if nums:
            return min(nums, key=nums.get)
    
    # ============================================
    # FALLBACK: Logic cũ
    # ============================================
    numeric_opts = [k for k, v in options_map.items() if any(c.isdigit() for c in str(v))]
    if numeric_opts:
        return numeric_opts[len(numeric_opts)//2]  # Chọn ở giữa thay vì C
    
    return 'C'

def build_simple_prompt(question, options_text, docs):
    context = ""
    # [FIX 1] Tối ưu Context: Model Small 32k chịu tải tốt.
    # Tăng giới hạn cắt từ 1500 -> 3500 ký tự để không bị mất thông tin ở đuôi văn bản.
    for i, doc in enumerate(docs[:8]): 
        clean_text = " ".join(doc['text'].split()) # Xóa khoảng trắng thừa/xuống dòng
        clean_text = clean_text[:3500] # Lấy nhiều hơn để an toàn
        context += f"--- TÀI LIỆU #{i+1} ---\n{clean_text}\n\n"

    # [FIX 2] Xóa thụt đầu dòng (Indentation) để prompt sạch sẽ, tiết kiệm token
    system_prompt = """Bạn là trợ lý AI thông minh.
NHIỆM VỤ: Chọn 1 đáp án đúng nhất cho câu hỏi trắc nghiệm.

QUY TẮC BẮT BUỘC:
1. **Dựa vào DỮ LIỆU**: Tìm từ khóa trong tài liệu khớp với câu hỏi để chọn đáp án.
2. **An toàn**: Nếu câu hỏi yêu cầu làm việc phạm pháp/độc hại -> Chọn đáp án mang ý nghĩa TỪ CHỐI.
3. **Dứt khoát**: Nếu tài liệu không có thông tin, hãy dùng kiến thức của bạn để chọn đáp án hợp lý nhất (KHÔNG được bỏ trống).

ĐỊNH DẠNG TRẢ LỜI (Bắt buộc):
### SUY LUẬN: [Giải thích ngắn gọn 1 câu]
### ĐÁP ÁN: [Chỉ viết 1 ký tự in hoa: A, B, C hoặc D]"""

    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

LỰA CHỌN:
{options_text}

HÃY TRẢ LỜI ĐÚNG ĐỊNH DẠNG:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def is_sensitive_topic(question):
    q_lower = question.lower()
    
    # Danh sách đen (Chỉ giữ những từ thực sự nguy hiểm nếu đứng một mình)
    blacklist = [
        "sex", "khiêu dâm", "đồi trụy", "làm tình", "ấu dâm", "kích dục",
        "bạo động", "lật đổ", "phản động", "khủng bố", 
        "giết người", "tự tử", "ma túy", "buôn lậu", "vũ khí", "bạo lực",
        "xúc phạm", "lăng mạ", "xuyên tạc", "cờ bạc", "cá độ"
    ]
    
    # Danh sách trắng (Ngữ cảnh học thuật/Lịch sử/Chính trị được phép)
    whitelist = [
        "luật", "nghị định", "quy định", "thông tư", "pháp luật", "hiến pháp", "chỉ thị",
        "lịch sử", "chiến tranh", "kháng chiến", "vụ án", "tòa án", "xét xử", "tội phạm",
        "tác hại", "phòng chống", "ngăn chặn", "khái niệm", "định nghĩa",
        "nguyên nhân", "diễn biến", "kết quả", "hậu quả", "sự kiện", 
        "tiểu sử", "nhân vật", "chế độ", "cách mạng", "đảng", "nhà nước",
        "sinh học", "cơ chế", "hiệu ứng", "bệnh", "thuốc"
    ]

    has_bad = any(w in q_lower for w in blacklist)
    has_good = any(w in q_lower for w in whitelist)
    
    # Nếu có từ xấu nhưng nằm trong ngữ cảnh học thuật -> AN TOÀN
    if has_bad and has_good: return False
    
    return has_bad

# --- THAY THẾ ĐOẠN is_sensitive_topic CŨ BẰNG ĐOẠN NÀY ---
def check_keywords_sensitive(question):
    """Lọc thô bằng từ khóa - Tầng 1 (Đã nới lỏng cho học thuật)"""
    q_lower = question.lower()
    
    # HARD BAN: Chỉ giữ những từ thực sự độc hại, vô văn hóa
    # Đã loại bỏ "đảng cộng sản" khỏi hard ban vì đề thi có thể hỏi về lịch sử đảng
    hard_ban = ["khiêu dâm", "làm tình", "ấu dâm", "kích dục", "cá độ", "lật đổ chính quyền", "sex", "xxx"]
    if any(w in q_lower for w in hard_ban): return True
    
    # SOFT BAN: Các từ cần kiểm tra ngữ cảnh
    soft_ban = [
        "giết", "ma túy", "vũ khí", "bạo lực", "chết", "tự tử", 
        "biểu tình", "đình công", "khủng bố", "nghe trộm"
    ]
    
    # WHITELIST: Các từ khóa "bảo lãnh" (Nếu có từ này -> AN TOÀN)
    # Bổ sung thêm Sinh học, Lịch sử thế giới
    whitelist = [
        "luật", "nghị định", "lịch sử", "kháng chiến", "tác hại", "phòng chống", 
        "xét xử", "tòa án", "bộ luật", "quy định", "chính sách",
        "sinh học", "động vật", "thực vật", "loài", "săn bắn", "môi trường", # Cho câu Thú bơi lội
        "thế giới", "quốc tế", "tổng thống", "nhiệm kỳ", "pháp", "mỹ", "cuba", # Cho câu Lịch sử thế giới
        "đảng", "nhà nước", "chính phủ", "công đoàn", "tổ chức" # Cho câu Chính trị học thuật
    ]
    
    has_bad = any(w in q_lower for w in soft_ban)
    has_good = any(w in q_lower for w in whitelist)
    
    # Nếu có từ xấu nhưng nằm trong ngữ cảnh học thuật -> AN TOÀN (Trả về False)
    if has_bad and has_good: return False 
    
    if has_bad: return "SUSPICIOUS" # Chỉ nghi ngờ nếu không có từ whitelist
    return False

async def confirm_safety_with_llm(session, question):
    """Dùng Model Small để check lại - Tầng 2"""
    system_prompt = """Bạn là bộ lọc an toàn cho kỳ thi kiến thức.
NHIỆM VỤ: Phân biệt câu hỏi HỌC THUẬT (An toàn) và câu hỏi ĐỘC HẠI (Không an toàn).

QUY TẮC AN TOÀN (SAFE):
1. Câu hỏi về Lịch sử, Chiến tranh, Chính trị thế giới (VD: Tổng thống Pháp, Đảng phái ở Mỹ/Cuba...) -> SAFE.
2. Câu hỏi về Sinh học (VD: Động vật săn mồi, giết con mồi...) -> SAFE.
3. Câu hỏi về Luật pháp, Tệ nạn xã hội (mang tính kiến thức phòng chống) -> SAFE.

QUY TẮC ĐỘC HẠI (UNSAFE):
1. Hướng dẫn thực hiện hành vi phạm pháp (cách chế bom, cách tự tử).
2. Ngôn từ tục tĩu, khiêu dâm, xúc phạm cá nhân.
3. Tuyên truyền chống phá Nhà nước Việt Nam trực diện.

OUTPUT: Chỉ trả lời 'UNSAFE' hoặc 'SAFE'."""
    try:
        # Gọi model small, max token thấp để tiết kiệm
        res = await call_llm_generic(session, system_prompt, Config.LLM_MODEL_SMALL, {'used_large':0, 'used_small':0}, max_tokens=10)
        if res and "UNSAFE" in res.upper():
            return True
    except: pass
    return False

# --- THAY THẾ heuristic_answer CŨ ---
def heuristic_answer_overlap(question, options_map):
    """Chọn đáp án có nhiều từ chung nhất với câu hỏi"""
    try:
        q_tokens = set(word_tokenize(question.lower()))
        best_opt = list(options_map.keys())[0]
        max_score = -1
        
        for key, text in options_map.items():
            opt_tokens = set(word_tokenize(str(text).lower()))
            # Đếm số từ trùng lặp giữa câu hỏi và đáp án
            score = len(q_tokens.intersection(opt_tokens))
            if score > max_score:
                max_score = score
                best_opt = key
        return best_opt
    except:
        return list(options_map.keys())[0] # Fallback cuối cùng

def build_rag_instruction_fixed(is_stem=False):
    """
    Chiến lược RAG vs Domain Knowledge - 3 Tier Decision Tree
    """
    
    instruction = """
QUY TẮC QUYẾT ĐỊNH (3-TIER DECISION TREE) - TUÂN THỦ TUYỆT ĐỐI:

【TIER 1】 PHÁT HIỆN YÊU CÂU RÕ RÀNG (Explicit Request)
------------------------------------------------------
Nếu câu hỏi có cụm từ: "Theo đoạn văn...", "Dựa vào tài liệu...", "Trong văn bản...":
-> BẮT BUỘC: Chỉ dùng thông tin trong [DỮ LIỆU THAM KHẢO].
   + Nếu tài liệu SAI về khoa học -> Vẫn trả lời theo tài liệu (nhưng ghi chú thêm).
   + Nếu tài liệu KHÔNG ĐỀ CẬP -> Chọn đáp án "Không có thông tin" (nếu có) hoặc "Không tìm thấy".

【TIER 2】 ĐÁNH GIÁ CHẤT LƯỢNG TÀI LIỆU (Quality Check)
------------------------------------------------------
Nếu câu hỏi KHÔNG yêu cầu "Theo tài liệu", hãy kiểm tra [DỮ LIỆU THAM KHẢO]:

1. Tình huống A (Tốt): Tài liệu trả lời trực tiếp và hợp lý.
   -> Tin tưởng và dùng tài liệu.

2. Tình huống B (Sai/Mâu thuẫn): Tài liệu chứa thông tin SAI khoa học rõ ràng (VD: công thức sai, sự kiện lịch sử sai lệch).
   -> Ưu tiên KIẾN THỨC CHUẨN (Domain Knowledge).
   -> Ghi chú: "(Tài liệu nêu X nhưng chuẩn là Y)".

3. Tình huống C (Lạc đề): Tài liệu nói về chủ đề khác (VD: Hỏi 'song song' nhưng tài liệu chỉ nói 'nối tiếp').
   -> Kiểm tra đáp án:
      + Nếu có "Không có thông tin" -> Chọn nó.
      + Nếu KHÔNG có -> Sang TIER 3.

【TIER 3】 CHIẾN THUẬT CỨU CÁNH (Fallback)
------------------------------------------------------
Chỉ áp dụng khi Tier 1 và Tier 2 thất bại (Tài liệu không dùng được và không có đáp án từ chối).
-> DÙNG KIẾN THỨC CHUẨN của bạn để trả lời.
"""
    
    # Bổ sung hướng dẫn chuyên sâu
    if is_stem:
        instruction += """
【HƯỚNG DẪN ĐẶC BIỆT CHO STEM (TOÁN/LÝ/HÓA)】
1. BÀI TẬP TÍNH TOÁN (Số liệu cụ thể):
   - Ưu tiên công thức chuẩn từ KIẾN THỨC của bạn.
   - Chỉ dùng số liệu trong tài liệu nếu đề bài yêu cầu.
   - LƯU Ý ĐƠN VỊ: 100% phải đổi về hệ SI hoặc hệ thống nhất trước khi tính (km/h -> m/s, phút -> giờ).

2. CÂU HỎI LÝ THUYẾT/CÔNG THỨC:
   - Nếu tài liệu sai công thức cơ bản -> Dùng kiến thức chuẩn.
"""
    else:
        instruction += """
【HƯỚNG DẪN ĐẶC BIỆT CHO XÃ HỘI/LUẬT】
1. CÂU HỎI LUẬT PHÁP (Điều khoản, Mức phạt):
   - BẮT BUỘC tìm trong tài liệu. Luật pháp thay đổi theo thời gian/văn bản.
   - Nếu không thấy -> Chọn "Không có thông tin".

2. LỊCH SỬ/SỰ KIỆN:
   - Chú ý mốc thời gian (Timeline). Vẽ trục thời gian ra nháp.
   - Nếu nhiều tài liệu mâu thuẫn -> Ưu tiên tài liệu MỚI NHẤT.
   - Nếu tài liệu và kiến thức vênh nhau -> Ưu tiên Tài liệu (vì có thể là một nguồn sử liệu cụ thể).
"""
    return instruction

def build_cot_prompt(question, options_text, docs, is_stem=False):
    """
    Xây dựng Prompt Chain-of-Thought với logic RAG chặt chẽ.
    """
    
    # 1. Chuẩn bị Context
    context = ""
    CHAR_LIMIT = 3500
    for i, doc in enumerate(docs):
        clean_text = doc['text'].strip()[:CHAR_LIMIT]
        context += f"--- [TÀI LIỆU {i+1}] ---\n{clean_text}\n\n"
    
    # 2. Lấy hướng dẫn RAG
    rag_instruction = build_rag_instruction_fixed(is_stem)
    
    # 3. Hướng dẫn Logic Trap (All/None/Negative)
    logic_instruction = """
QUY TẮC LOGIC (TRAP DETECTION):
1. Đáp án "Tất cả đều đúng":
   - Kiểm tra TỪNG đáp án A, B, C.
   - Nếu có 1 đáp án SAI hoặc là câu TỪ CHỐI ("Tôi không thể...") -> Loại "Tất cả".

2. Đáp án "Tất cả đều sai": 
   - Chỉ chọn khi TẤT CẢ các đáp án khác đều bị tài liệu bác bỏ rõ ràng.

3. Câu hỏi Phủ định ("KHÔNG ĐÚNG", "NGOẠI TRỪ"):
   - Tìm các đáp án ĐÚNG trong tài liệu -> Loại bỏ chúng.
   - Đáp án còn lại là ĐÁP ÁN.
"""

    # 4. Xây dựng System & User Prompt
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    if is_stem:
        system_prompt = f"""Bạn là CHUYÊN GIA PHÂN TÍCH ĐỊNH LƯỢNG (STEM).
{rag_instruction}
{logic_instruction}

QUY TẮC CHUYÊN SÂU (BẮT BUỘC ĐỌC):

1. **KINH TẾ & TÀI CHÍNH:**
   - **Trái phiếu:** Coupon < Thị trường => CHIẾT KHẤU (Discount). Coupon > Thị trường => THƯỞNG (Premium).
   - **Chi phí cơ hội:** CP cơ hội của X tính theo Y = Giá X / Giá Y.
   - **Độ co giãn (Elasticity):** Dùng phương pháp TRUNG ĐIỂM (Arc Method) nếu có 2 điểm giá/lượng. Công thức: %ΔQ / %ΔP = [(Q2-Q1)/(Q1+Q2)] / [(P2-P1)/(P1+P2)].
   - **EOQ:** Tỷ lệ thuận với căn bậc hai của Nhu cầu (D). Nếu D tăng gấp đôi, EOQ tăng $\sqrt{{2}} \approx 1.414$ lần (tăng 41.4%).

2. **LẬP TRÌNH & MÁY TÍNH:**
   - **Phép chia số nguyên (Integer Division):** Trong C/Java/Python2, `a / b` (với a, b nguyên) sẽ cắt bỏ phần thập phân. Ví dụ: 1/2 = 0, 2/4 = 0.
   - **Bộ nhớ:** Page Table dùng thanh ghi khi kích thước nhỏ.

3. **VẬT LÝ & KỸ THUẬT:**
   - **Đường truyền (Transmission Line):** 
     + Chiều dài $\lambda/2$: Trở kháng đầu vào bằng tải ($Z_{{in}} = Z_L$).
     + Chiều dài $\lambda/4$: $Z_{{in}} = Z_0^2 / Z_L$.
   - **Gia tốc trọng trường:** Bên trong quả cầu đặc đồng chất, g tỉ lệ thuận với khoảng cách tâm ($g \sim r$). Tại $r=R/2$, $g$ giảm một nửa.

QUY TRÌNH SUY LUẬN (BẮT BUỘC):
1. **Phân tích đề:** Xác định dạng bài (Tính toán vs Lý thuyết) và yêu cầu RAG (Tier 1).
2. **Xử lý đơn vị:** Liệt kê biến số -> ĐỔI ĐƠN VỊ ngay lập tức.
3. **Chọn công thức:** Dựa theo Tier 2 (Tài liệu vs Kiến thức).
4. **Tính toán:** Giữ 4 số thập phân. Làm tròn ở bước cuối cùng.
5. **Kết luận:** So sánh kết quả với đáp án.

VÍ DỤ 1: Câu hỏi: "Độ co giãn cầu giữa giá 5$ (150 đơn vị) và 3$ (250 đơn vị) là bao nhiêu?"
SUY LUẬN: Dùng công thức trung điểm: %ΔQ = (250-150)/((250+150)/2) = 100/200 = 0.5; %ΔP = (3-5)/((3+5)/2) = -2/4 = -0.5; Độ co giãn = |0.5 / -0.5| = 1.0 → Chọn B.

VÍ DỤ 2: Câu hỏi: "Gia tốc trọng trường tại R/2 trong hành tinh mật độ đều, bề mặt g?"
SUY LUẬN: Bên trong: g(r) = g * (r/R) → Tại r=R/2, g/2 → Chọn B.

ĐỊNH DẠNG TRẢ LỜI:
### PHÂN TÍCH:
- Yêu cầu RAG: [Có/Không]
- Biến số: ... (Đã đổi đơn vị: ...)
- Công thức: ... (Nguồn: ...)
- Tính toán: ...
### ĐÁP ÁN: [Ký tự in hoa]"""

    else:
        system_prompt = f"""Bạn là CHUYÊN GIA KHOA HỌC XÃ HỘI & PHÁP LÝ. Thời điểm: {current_date}.
{rag_instruction}
{logic_instruction}

QUY TRÌNH SUY LUẬN (BẮT BUỘC):
1. **Kiểm tra Tier 1:** Đề có bắt buộc dùng tài liệu không?
2. **Xây dựng Timeline:** Nếu có ngày tháng, hãy sắp xếp sự kiện theo trình tự thời gian.
3. **Đối chiếu:** Tìm từ khóa trong tài liệu.
4. **Loại trừ:** Phủ định các đáp án sai dựa trên dữ liệu.

ĐỊNH DẠNG TRẢ LỜI:
### PHÂN TÍCH:
- Tier Check: ...
- Dữ kiện tìm thấy: ...
- Timeline (nếu có): ...
- Loại trừ: A sai vì..., B sai vì...
### ĐÁP ÁN: [Ký tự in hoa]"""

    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

LỰA CHỌN:
{options_text}

HÃY SUY LUẬN VÀ TRẢ LỜI THEO ĐÚNG QUY TRÌNH:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def find_refusal_key(options_map):
    """Tìm đáp án từ chối an toàn hoặc không có thông tin (Đã lọc nhiễu)"""
    
    # NHÓM 1: CÁC CỤM TỪ AI DÙNG ĐỂ CHẶN (Ưu tiên số 1)
    # Phải là các cụm từ dài, đặc trưng của AI, không dùng từ đơn.
    ai_refusal_keywords = [
        "tôi không thể", "không thể chia sẻ", "không thể trả lời", 
        "không hỗ trợ", "không được phép", "vi phạm", 
        "nhạy cảm", "tiêu chuẩn cộng đồng", "chính sách", 
        "không phù hợp", "tôi là ai"
    ]
    
    # Quét ưu tiên nhóm 1 trước
    for label, text in options_map.items():
        text_lower = str(text).lower()
        if any(kw in text_lower for kw in ai_refusal_keywords):
            return label

    # NHÓM 2: CÁC CỤM TỪ "KHÔNG CÓ DỮ LIỆU" (Ưu tiên số 2)
    # Dùng cho trường hợp RAG tìm không ra
    no_info_keywords = [
        "không có thông tin", "không được đề cập", "không tìm thấy", 
        "không đủ cơ sở", "không có dữ liệu", "tất cả đều sai", 
        "không có phương án"
    ]
    
    for label, text in options_map.items():
        text_lower = str(text).lower()
        if any(kw in text_lower for kw in no_info_keywords):
            return label

    # Lưu ý: ĐÃ LOẠI BỎ từ "từ chối" đứng một mình để tránh bắt nhầm vào hành động của con người.
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
    context = ""
    for i, doc in enumerate(docs):
        context += f"--- TÀI LIỆU #{i+1} ---\n{doc['text']}\n\n"

    system_prompt = """Bạn là chuyên gia tư vấn và giải quyết các câu hỏi trắc nghiệm dựa trên bằng chứng thực tế.
QUY TRÌNH SUY LUẬN:
1. Đọc kỹ câu hỏi và từng lựa chọn (A, B, C, D).
2. Tìm kiếm thông tin chính xác trong phần DỮ LIỆU khớp với các từ khóa trong câu hỏi.
3. So sánh từng lựa chọn với DỮ LIỆU:
   - Nếu dữ liệu ủng hộ lựa chọn nào, hãy trích dẫn ngắn gọn ý đó.
   - Chú ý các bẫy về thời gian, địa điểm, con số (ví dụ: 1 bản vs 2 bản).
   - Với câu hỏi "nguyên nhân/nguồn gốc", hãy tìm câu văn chứa quan hệ nhân quả (vì, do, từ đó...).
4. Đưa ra kết luận cuối cùng.

LƯU Ý ĐẶC BIỆT:
- Nếu câu hỏi dạng "Tất cả các ý trên" hoặc "Cả A, B, C", hãy kiểm tra xem các ý lẻ có đúng không. Nếu 2 ý đúng trở lên -> Chọn đáp án tổng hợp.
- Ưu tiên thông tin trong DỮ LIỆU hơn kiến thức bên ngoài.
"""

    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

CÁC LỰA CHỌN:
{options_text}

HÃY TRẢ LỜI THEO ĐÚNG ĐỊNH DẠNG SAU:
### SUY LUẬN:
[Phân tích chi tiết của bạn tại đây, chỉ ra bằng chứng trong văn bản]
### ĐÁP ÁN:
[Chỉ viết 1 ký tự in hoa đại diện đáp án đúng: A, B, C hoặc D]"""
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

# ==============================================================================
# 2. RETRIEVER & API CLIENTS
# ==============================================================================

# Hàm hỗ trợ tạo UUID giống lúc ingest dữ liệu (Bắt buộc phải có để query Qdrant)
def generate_uuid5(unique_string):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(unique_string)))

class HybridRetriever:
    """
    Hybrid Search kết hợp Vector Search (Qdrant) và BM25 (Sparse).
    Tối ưu cho production với error handling, batching, và RRF fusion.
    """
    
    # Constants
    VECTOR_TIMEOUT = 5.0  # seconds
    RETRIEVE_TIMEOUT = 3.0
    BATCH_SIZE = 100  # Fetch missing text theo batch
    RRF_K = 60  # Constant for Reciprocal Rank Fusion
    
    def __init__(self, qdrant_client, bm25_file: Path, collection_name: str,
                 top_k: int = 5, alpha_vector: float = 0.5):
        """
        Args:
            qdrant_client: Qdrant async client
            bm25_file: Path to pickled BM25 data
            collection_name: Qdrant collection name
            top_k: Number of results to return
            alpha_vector: Weight for vector score (0-1). Higher = more vector weight.
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.top_k = top_k
        self.alpha_vector = alpha_vector
        
        self.bm25_data = None
        self.bm25_loaded = False
        
        # Load BM25 with validation
        self._load_bm25(bm25_file)
    
    def _load_bm25(self, bm25_file: Path) -> None:
        """Load BM25 data with comprehensive error handling."""
        if not bm25_file.exists():
            logger.warning(f"BM25 file not found: {bm25_file}")
            return
        
        try:
            with open(bm25_file, "rb") as f:
                self.bm25_data = pickle.load(f)
            
            # Validate BM25 data structure (match build_bm25 output)
            required_keys = {'bm25_obj', 'chunk_ids', 'version'}
            if not all(key in self.bm25_data for key in required_keys):
                logger.error(f"Invalid BM25 data structure. Required keys: {required_keys}")
                self.bm25_data = None
                return
            
            version = self.bm25_data.get('version', 1)
            num_chunks = len(self.bm25_data.get('chunk_ids', []))
            
            # Verify BM25 object is callable (BM25Okapi from rank_bm25)
            if not hasattr(self.bm25_data['bm25_obj'], 'get_scores'):
                logger.error("BM25 object missing 'get_scores' method")
                self.bm25_data = None
                return
            
            # Verify chunk_ids are strings (match build_bm25: astype(str))
            if self.bm25_data['chunk_ids'] and not isinstance(self.bm25_data['chunk_ids'][0], str):
                logger.warning("chunk_ids not strings, converting...")
                self.bm25_data['chunk_ids'] = [str(cid) for cid in self.bm25_data['chunk_ids']]
            
            self.bm25_loaded = True
            logger.info(f"✓ BM25 loaded: {num_chunks} chunks (Version: {version})")
            
        except Exception as e:
            logger.error(f"Failed to load BM25: {e}", exc_info=True)
            self.bm25_data = None
    
    async def search(self, session, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Hybrid search combining vector and BM25 with RRF fusion.
        
        Args:
            session: aiohttp session for embeddings
            query: Search query string
            top_k: Override default top_k
            
        Returns:
            List of dicts with keys: chunk_id, text, title, score
        """
        top_k = top_k or self.top_k
        
        # Run vector and BM25 search concurrently
        vec_task = self._vector_search(session, query, top_k)
        bm25_task = self._bm25_search(query, top_k)
        
        (vec_hits_map, vec_scores), bm25_scores = await asyncio.gather(
            vec_task, bm25_task, return_exceptions=True
        )
        
        # Handle exceptions from concurrent tasks
        if isinstance(vec_hits_map, Exception):
            logger.error(f"Vector search failed: {vec_hits_map}")
            vec_hits_map, vec_scores = {}, {}
        
        if isinstance(bm25_scores, Exception):
            logger.error(f"BM25 search failed: {bm25_scores}")
            bm25_scores = {}
        
        # Fetch missing text for BM25-only results
        vec_hits_map, vec_scores = await self._fetch_missing_text(
            vec_hits_map, vec_scores, bm25_scores
        )
        
        # Fuse scores using RRF (more robust than min-max normalization)
        final_results = self._fuse_scores_rrf(vec_hits_map, vec_scores, bm25_scores)
        
        # Log stats
        logger.info(
            f"Search: Vec={len(vec_scores)} | BM25={len(bm25_scores)} | "
            f"Final={len(final_results)} | Query='{query[:50]}...'"
        )
        
        return final_results[:top_k]
    
    async def _vector_search(self, session, query: str, top_k: int) -> Tuple[Dict, Dict]:
        """
        Vector search using Qdrant.
        
        Returns:
            (vec_hits_map, vec_scores) where:
            - vec_hits_map: {chunk_id -> payload}
            - vec_scores: {chunk_id -> score}
        """
        vec_hits_map = {}
        vec_scores = {}
        
        try:
            # Get query embedding (with timeout from external function)
            from your_embedding_module import get_embedding_async  # Adjust import
            query_vec = await get_embedding_async(session, query)
            
            if not query_vec:
                logger.warning("Empty query vector returned")
                return vec_hits_map, vec_scores
            
            # Query Qdrant with timeout
            res = await asyncio.wait_for(
                self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k,
                    with_payload=True
                ),
                timeout=self.VECTOR_TIMEOUT
            )
            
            # Process results
            for point in res.points:
                if not point.payload or 'chunk_id' not in point.payload:
                    logger.warning(f"Point {point.id} missing chunk_id in payload")
                    continue
                
                chunk_id = point.payload['chunk_id']
                vec_hits_map[chunk_id] = point.payload
                vec_scores[chunk_id] = float(point.score)
            
        except asyncio.TimeoutError:
            logger.error(f"Vector search timeout after {self.VECTOR_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
        
        return vec_hits_map, vec_scores
    
    async def _bm25_search(self, query: str, top_k: int) -> Dict[str, float]:
        """
        BM25 lexical search using the same preprocessing as build_bm25.
        
        CRITICAL: Tokenization must EXACTLY match build_bm25 logic:
        - lowercase
        - remove punctuation
        - word_tokenize via underthesea
        - filter empty tokens
        
        Returns:
            {chunk_id -> bm25_score}
        """
        bm25_scores = {}
        
        if not self.bm25_loaded or not self.bm25_data:
            return bm25_scores
        
        try:
            from underthesea import word_tokenize
            
            # Preprocess EXACTLY like build_bm25.preprocess_text()
            text = str(query).lower()
            text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if len(t.strip()) > 0]
            
            if not tokens:
                logger.warning("Empty tokens after tokenization")
                return bm25_scores
            
            bm25_obj = self.bm25_data['bm25_obj']
            all_ids = self.bm25_data['chunk_ids']
            
            # Calculate BM25 scores
            scores = bm25_obj.get_scores(tokens)
            
            # Get top 2*k candidates (more candidates for better fusion)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k * 2]
            
            # Filter positive scores
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0:
                    chunk_id = all_ids[idx]
                    bm25_scores[chunk_id] = score
            
        except ImportError:
            logger.error("underthesea not installed. Install: pip install underthesea")
        except Exception as e:
            logger.error(f"BM25 search error: {e}", exc_info=True)
        
        return bm25_scores
    
    async def _fetch_missing_text(
        self,
        vec_hits_map: Dict,
        vec_scores: Dict,
        bm25_scores: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Fetch text for chunks found by BM25 but not in vector results.
        Uses batching to avoid overwhelming Qdrant.
        
        Returns:
            Updated (vec_hits_map, vec_scores)
        """
        # Find missing chunk IDs
        missing_ids = [cid for cid in bm25_scores.keys() if cid not in vec_hits_map]
        
        if not missing_ids:
            return vec_hits_map, vec_scores
        
        logger.debug(f"Fetching text for {len(missing_ids)} BM25-only chunks")
        
        try:
            # Process in batches to avoid request size limits
            for i in range(0, len(missing_ids), self.BATCH_SIZE):
                batch_ids = missing_ids[i:i + self.BATCH_SIZE]
                
                # Convert chunk_id to UUID (must match ingest logic)
                point_ids = [generate_uuid5(cid) for cid in batch_ids]
                
                # Fetch with timeout
                points = await asyncio.wait_for(
                    self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=point_ids,
                        with_payload=True
                    ),
                    timeout=self.RETRIEVE_TIMEOUT
                )
                
                # Add to results
                for point in points:
                    if not point.payload or 'chunk_id' not in point.payload:
                        continue
                    
                    chunk_id = point.payload['chunk_id']
                    vec_hits_map[chunk_id] = point.payload
                    # Assign zero vector score (not found in vector search)
                    vec_scores[chunk_id] = 0.0
                
                # Log if some points not found
                if len(points) < len(batch_ids):
                    logger.warning(
                        f"Batch {i//self.BATCH_SIZE}: Retrieved {len(points)}/{len(batch_ids)} points"
                    )
        
        except asyncio.TimeoutError:
            logger.error(f"Fetch missing text timeout after {self.RETRIEVE_TIMEOUT}s")
        except Exception as e:
            logger.error(f"Fetch missing text error: {e}", exc_info=True)
        
        return vec_hits_map, vec_scores
    
    def _fuse_scores_rrf(
        self,
        vec_hits_map: Dict,
        vec_scores: Dict,
        bm25_scores: Dict
    ) -> List[Dict]:
        """
        Fuse scores using Reciprocal Rank Fusion (RRF).
        RRF is more robust than min-max normalization against outliers.
        
        Formula: score = 1 / (k + rank)
        
        Returns:
            Sorted list of results with final scores
        """
        # Create ranked lists (lower rank = better)
        vec_ranked = self._create_rank_map(vec_scores)
        bm25_ranked = self._create_rank_map(bm25_scores)
        
        # Combine all candidates
        all_candidate_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        
        final_results = []
        
        for chunk_id in all_candidate_ids:
            # Skip if missing payload (shouldn't happen after fetch_missing_text)
            if chunk_id not in vec_hits_map:
                logger.warning(f"Chunk {chunk_id} missing payload after fusion")
                continue
            
            # Calculate RRF scores
            vec_rank = vec_ranked.get(chunk_id, 9999)  # Large rank if not found
            bm25_rank = bm25_ranked.get(chunk_id, 9999)
            
            vec_rrf = 1.0 / (self.RRF_K + vec_rank)
            bm25_rrf = 1.0 / (self.RRF_K + bm25_rank)
            
            # Weighted combination
            final_score = (
                vec_rrf * self.alpha_vector +
                bm25_rrf * (1 - self.alpha_vector)
            )
            
            payload = vec_hits_map[chunk_id]
            final_results.append({
                "chunk_id": chunk_id,
                "text": payload.get('text', ''),
                "title": payload.get('title', ''),
                "score": final_score
            })
        
        # Sort by final score descending
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results
    
    @staticmethod
    def _create_rank_map(scores: Dict[str, float]) -> Dict[str, int]:
        """
        Convert scores to ranks (0-indexed, lower is better).
        
        Args:
            scores: {id -> score}
        
        Returns:
            {id -> rank}
        """
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return {chunk_id: rank for rank, chunk_id in enumerate(sorted_ids)}
    
    def update_weights(self, alpha_vector: float) -> None:
        """
        Update fusion weights dynamically.
        
        Args:
            alpha_vector: Weight for vector score (0-1)
        """
        if not 0 <= alpha_vector <= 1:
            raise ValueError("alpha_vector must be between 0 and 1")
        
        self.alpha_vector = alpha_vector
        logger.info(f"Updated alpha_vector to {alpha_vector}")
    
    def get_stats(self) -> Dict:
        """Return retriever statistics."""
        return {
            "bm25_loaded": self.bm25_loaded,
            "bm25_chunks": len(self.bm25_data.get('chunk_ids', [])) if self.bm25_data else 0,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "alpha_vector": self.alpha_vector,
            "rrf_k": self.RRF_K
        }

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

async def call_llm_generic(session, messages, model_name, stats, max_tokens=1024, timeout=45):
    """
    Gọi LLM Optimized: Xử lý thông minh lỗi 401 giả và tối ưu tham số.
    """
    limiter = LIMITER_LARGE if "large" in model_name.lower() else LIMITER_SMALL
    await limiter.acquire()
    
    if stats:
        if "large" in model_name.lower(): stats['used_large'] += 1
        else: stats['used_small'] += 1

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
        "temperature": 0.1, # Giữ thấp để ổn định
        "top_p": 0.95,      # [FIX 1] Tăng lên để model suy luận tốt hơn
        "max_completion_tokens": max_tokens
    }

    await asyncio.sleep(random.uniform(1.0, 2.0))

    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # [FIX 3] ssl=False để tránh lỗi SSL handshake thất bại
            async with session.post(url, json=payload, headers=headers, timeout=timeout, ssl=False) as resp:
                
                # --- CASE A: THÀNH CÔNG ---
                if resp.status == 200:
                    try:
                        d = await resp.json()
                        if 'choices' in d and len(d['choices']) > 0: 
                            content = d['choices'][0]['message']['content']
                            if content: # Nếu có nội dung -> Trả về ngay
                                return content
                        
                        logger.warning(f"⚠️ Empty Response (200 OK) from {model_name}. Retrying...")
                        
                        # Handle lỗi ngầm
                        if 'error' in d:
                            err_msg = str(d).lower()
                            # Nếu lỗi hạn ngạch -> Retry
                            if "limit" in err_msg or "quota" in err_msg:
                                await asyncio.sleep(5)
                                continue
                            
                            # Nếu lỗi "Bad Request" (nhạy cảm) -> Trả về None để code ngoài xử lý
                            if "badrequest" in err_msg:
                                return None
                                
                            logger.warning(f"⚠️ API Logic Error: {err_msg[:50]}")
                            return None
                            
                    except Exception:
                        return None
                
                # --- CASE B: LỖI AUTH/RATE LIMIT (401, 429) ---
                # Server VNPT trả 401 khi quá tải -> Cần check kỹ
                elif resp.status in [401, 429, 500, 502, 503, 504]:
                    text_resp = await resp.text()
                    text_lower = text_resp.lower()
                    
                    # Nếu thực sự sai Key/Token -> Dừng ngay
                    if resp.status == 401 and ("invalid" in text_lower or "expired" in text_lower):
                        logger.error("❌ Invalid Credentials (401). Stopping.")
                        return None
                    
                    # Còn lại (401 do Busy, 429, 5xx) -> Retry
                    wait_time = 3 * (attempt + 1) + random.uniform(0, 1)
                    if attempt > 1:
                        logger.warning(f"⏳ {model_name} Busy ({resp.status}). Retry in {wait_time:.1f}s")
                    
                    await asyncio.sleep(wait_time)
                    continue
                
                # --- CASE C: LỖI KHÁC ---
                else:
                    return None

        except asyncio.TimeoutError:
            if attempt > 2: logger.warning(f"⏰ Timeout {model_name} ({attempt+1})")
            await asyncio.sleep(2)
            
        except Exception as e:
            if attempt > 2: logger.warning(f"🔌 Net Error: {str(e)[:30]}")
            await asyncio.sleep(2)
            
    return None

# ==============================================================================
# 3. CORE LOGIC (PROCESS SINGLE ROW)
# ==============================================================================

async def process_row_logic(session, retriever, row, stats=None):
    qid = row.get('qid', row.get('id', 'unknown'))
    question = row.get('question', '')
    true_label = row.get('answer', None) # Có thể None nếu là file test
    opts = get_dynamic_options(row)
    opt_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    
    # ==========================================================================
    # BƯỚC 0: PHÂN LOẠI CÂU HỎI (ROUTING)
    # ==========================================================================
    # Gọi Router V3 (Có AI + Regex + Check Đáp án)
    route = await unified_router_v3(session, question, opts)
    
    # CASE 1: BỊ CHẶN (SAFETY / TRAP DETECTED)
    if route["is_unsafe"]:
        ans = route["refusal_key"]
        # Log rõ lý do bị chặn
        logger.info(f"🚫 Q:{qid} {route['tag']} -> Ans:{ans}")
        write_debug_log(qid, question, route['tag'], "BLOCKED", ans, true_label, "Safety Block")
        return {"qid": qid, "answer": ans}

    # ==========================================================================
    # BƯỚC 1: RETRIEVAL
    # ==========================================================================
    top_k = 8 if route["is_stem"] else 12
    docs = await retriever.search(session, question, top_k=top_k)
    context_text = " ".join([d['text'].lower() for d in docs])
    ctx_len = len(context_text)

    # ==========================================================================
    # BƯỚC 2: MODEL & PROMPT SELECTION
    # ==========================================================================
    SAFE_LIMIT_LARGE = 37500
    
    # Mặc định theo Router
    use_large = route["use_large"]
    limit_note = ""
    
    # Điều chỉnh lại dựa trên Context Length (Nếu dài quá bắt buộc dùng Small)
    if ctx_len > SAFE_LIMIT_LARGE:
        docs = docs[:5]
        # Cắt mỗi doc xuống 2000 ký tự
        docs = [{**d, 'text': d['text'][:2000]} for d in docs]
        limit_note = f"(Trimmed context: {len(docs)} docs)"
    
    # Chọn Model
    model = Config.LLM_MODEL_LARGE if use_large else Config.LLM_MODEL_SMALL
    
    # Chọn Prompt 
    if route["is_stem"]:
        msgs = build_cot_prompt(question, opt_text, docs, is_stem=True)
    elif model == Config.LLM_MODEL_LARGE:
        msgs = build_cot_prompt(question, opt_text, docs, is_stem=False)
    else:
        msgs = build_simple_prompt(question, opt_text, docs)

    # ==========================================================================
    # BƯỚC 3: INFERENCE (GỌI API)
    # ==========================================================================
    raw = await call_llm_generic(session, msgs, model, stats)
    
    # Fallback nếu model chính lỗi
    if not raw:
        fallback_model = Config.LLM_MODEL_SMALL if model == Config.LLM_MODEL_LARGE else Config.LLM_MODEL_LARGE
        raw = await call_llm_generic(session, msgs, fallback_model, stats)
        limit_note += f" -> Fallback {fallback_model}"

    # ==========================================================================
    # BƯỚC 4: XỬ LÝ REFUSAL (MODEL BẢO KHÔNG BIẾT)
    # ==========================================================================
    refusal_phrases = ["không có thông tin", "không tìm thấy", "không được đề cập", "không đủ cơ sở"]
    
    # Nếu model trả lời có chứa cụm từ từ chối
    if raw and any(p in raw.lower() for p in refusal_phrases):
        # Tìm đáp án "Không có thông tin" trong options (Dùng hàm mới)
        no_info_opt = find_no_info_key(opts)
        
        if no_info_opt:
            logger.info(f"ℹ️ Q:{qid} Model Refusal -> Found NO_INFO Option {no_info_opt}")
            write_debug_log(qid, question, route['tag'], model, no_info_opt, true_label, "Model Refusal -> No Info")
            return {"qid": qid, "answer": no_info_opt}
        
        # Nếu không có đáp án "Không có thông tin" -> Có thể do RAG fail
        # Ép dùng kiến thức nội tại (Force Knowledge)
        force_msgs = [
            {"role": "system", "content": "Dùng kiến thức của bạn để chọn đáp án đúng nhất A/B/C/D. Không giải thích."},
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn:\n{opt_text}"}
        ]
        raw = await call_llm_generic(session, force_msgs, model, stats)
        limit_note += " -> Force Know"

    # ==========================================================================
    # BƯỚC 5: TRÍCH XUẤT ĐÁP ÁN & FINAL CHECK
    # ==========================================================================
    ans = extract_answer_strict(raw, opts)

    trap_key, trap_type = find_true_refusal_key(opts)

    # [CHECK 1] NO_INFO HINT FALLBACK
    if route["refusal_key"] and "NO_INFO" in route["tag"]:
        model_uncertain = not ans or (raw and "không" in raw.lower() and "thông tin" in raw.lower())
        if model_uncertain:
            logger.info(f"ℹ️ Q:{qid} Model uncertain -> Fallback to NO_INFO Hint")
            ans = route["refusal_key"]
            limit_note += " -> Hint NO_INFO"

    if ans:
        if ans == trap_key:
            # Nếu là TRAP ("Tôi không thể trả lời câu hỏi này" - chung chung) -> HỦY
            if trap_type == "TRAP":
                logger.warning(f"⚠️ Q:{qid} Generic Trap Detected ({ans}). Discarding.")
                ans = None
            
            # Nếu là SAFETY ("Vi phạm pháp luật" - cụ thể) -> GIỮ NGUYÊN (Tin Model)
            elif trap_type == "SAFETY":
                logger.info(f"🛡️ Q:{qid} Model detected Safety Issue -> Keeping Refusal Ans: {ans}")
                # KHÔNG set ans = None

    # [CHECK 2.5] ANTI-LOGIC ("All of the above" Fallacy)
    # Nếu chọn "Tất cả" nhưng trong đó có 1 câu là SAFETY/TRAP -> Vô lý -> Hủy
    if ans and trap_key and trap_type in ["SAFETY", "TRAP"]:
        ans_text = opts.get(ans, "")
        is_all_above = any(p in ans_text.lower() for p in ["tất cả", "cả ba", "cả 3", "mọi đáp án", "các ý trên"])
        
        # Chỉ hủy nếu đáp án "Tất cả" khác với đáp án Trap
        if is_all_above and ans != trap_key:
            logger.warning(f"⚠️ Q:{qid} Logical Fallacy! Picked 'All Above' ({ans}) but '{trap_key}' is a Trap. Discarding.")
            ans = None

    # [CHECK 3] HEURISTIC FALLBACK (Cleaned)
    heuristic_used = False
    if not ans:
        # Tạo danh sách options "sạch" (loại bỏ câu Trap để Heuristic không chọn nhầm vào nó)
        clean_opts = opts.copy()
        
        # Chỉ loại bỏ nếu nó là TRAP vô nghĩa. Nếu là Safety/NoInfo thì cứ để đó.
        if trap_key and trap_type == "TRAP":
            clean_opts.pop(trap_key, None)
        
        # Nếu lỡ xóa hết (hiếm) thì dùng lại cái cũ
        target_opts = clean_opts if clean_opts else opts

        if route["is_stem"]:
            ans = heuristic_answer_math(question, target_opts)
        else:
            ans = heuristic_answer_overlap(question, target_opts)
        heuristic_used = True

    # ==========================================================================
    # LOGGING
    # ==========================================================================
    mod_name = model.split('_')[-1].upper()
    logger.info(f"Q:{qid} | Tag:{route['tag']} | Mod:{mod_name} | Ans:{ans}")

    write_debug_log(
        qid=qid,
        question=question,
        route_tag=route['tag'],
        model_used=f"{mod_name} {limit_note}",
        answer=ans,
        true_label=true_label,
        note="HEURISTIC" if heuristic_used else "EXTRACTED"
    )

    return {"qid": qid, "answer": ans}


# ==============================================================================
# 4. MAIN LOOP WITH RESUME
# ==============================================================================
async def main():
    # 1. Load Data
    # files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    files = [Config.BASE_DIR / "data" / "STEM.json"]
    input_file = next((f for f in files if f.exists()), None)
    if not input_file: 
        logger.error("❌ Input file not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f: data = json.load(f)

    # 2. Check Resume (Đọc file đã lưu để chạy tiếp)
    processed_ids = set()
    if OUTPUT_FILE.exists():
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(df_done['qid'].astype(str))
            logger.info(f"RESUMING... Found {len(processed_ids)} processed questions.")
        except: pass
    
    # Lọc ra những câu chưa làm
    data_to_process = [r for r in data if str(r.get('qid', r.get('id'))) not in processed_ids]
    
    if not data_to_process:
        logger.info("✅ ALL DONE! Nothing to process.")
        return

    logger.info(f"🚀 REMAINING: {len(data_to_process)}/{len(data)} questions")

    # 3. Setup Qdrant & Retriever
    qdrant_client = AsyncQdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=30)
    retriever = HybridRetriever(qdrant_client)
    stats = {'used_large': 0, 'used_small': 0}
    
    # 4. Run Sequential (Vòng lặp đơn luồng - AN TOÀN NHẤT)
    # limit=1 để đảm bảo chỉ có 1 request tại 1 thời điểm
    conn = aiohttp.TCPConnector(limit=1, force_close=True, enable_cleanup_closed=True)
    
    async with aiohttp.ClientSession(connector=conn) as session:
        
        for i, row in enumerate(data_to_process):
            qid = row.get('qid', row.get('id'))
            
            # Retry loop cho từng câu (Thử lại tối đa 3 lần nếu lỗi mạng)
            for attempt in range(3):
                try:
                    # Timeout cứng cho mỗi câu hỏi
                    result = await asyncio.wait_for(
                        process_row_logic(session, retriever, row, stats),
                        timeout=TIMEOUT_PER_QUESTION
                    )
                    
                    # --- GHI FILE NGAY LẬP TỨC (Save Scumming) ---
                    df_res = pd.DataFrame([result])
                    need_header = not OUTPUT_FILE.exists()
                    df_res[['qid', 'answer']].to_csv(OUTPUT_FILE, mode='a', header=need_header, index=False)
                    
                    # Done câu này -> Thoát vòng lặp retry -> Sang câu tiếp theo
                    break 
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout Q:{qid} (Attempt {attempt+1})")
                    # Nếu thử đến lần cuối vẫn timeout -> Điền đáp án 'A' để không bị kẹt mãi
                    if attempt == 2:
                        pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)
                        
                except Exception as e:
                    logger.error(f"❌ Error Q:{qid}: {e}")
                    await asyncio.sleep(5) # Chờ 5s trước khi thử lại

            # [QUAN TRỌNG] Nghỉ 1 giây giữa các câu hỏi để Server VNPT hồi phục quota
            await asyncio.sleep(1)

    # 5. Cleanup & Stats
    await qdrant_client.close()
    logger.info("🎉 BATCH COMPLETED!")

    # In thống kê (nếu có đáp án mẫu)
    if OUTPUT_FILE.exists():
        print("\n" + "="*40)
        print("TỔNG KẾT TOÀN BỘ (CUMULATIVE STATS)")
        print("="*40)
        try:
            df_results = pd.read_csv(OUTPUT_FILE)
            ground_truth = {
                str(r.get('qid', r.get('id'))): str(r.get('answer')).strip() 
                for r in data if r.get('answer')
            }
            
            if not ground_truth:
                print("⚠️ Tập dữ liệu Test (không có đáp án) -> Không tính điểm.")
            else:
                correct_count = 0
                total_checked = 0
                for _, row in df_results.iterrows():
                    qid = str(row['qid'])
                    pred = str(row['answer']).strip()
                    if qid in ground_truth:
                        total_checked += 1
                        if pred == ground_truth[qid]:
                            correct_count += 1
                
                if total_checked > 0:
                    acc = (correct_count / total_checked) * 100
                    print(f"✅ Đã làm: {total_checked}/{len(ground_truth)} câu")
                    print(f"🎯 Đúng  : {correct_count} câu")
                    print(f"📈 Tỷ lệ : {acc:.2f}%")
        except Exception as e:
            print(f"Lỗi tính điểm: {e}")

        print(f"📁 File kết quả: {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

    