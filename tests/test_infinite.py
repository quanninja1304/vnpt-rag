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
TIMEOUT_PER_QUESTION = None

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
def find_true_refusal_key(options_map):
    """
    Trả về: (Key, Type)
    - Type 'SAFETY': Từ chối vì vi phạm -> BLOCK NGAY.
    - Type 'NO_INFO': Từ chối vì thiếu tin -> Dùng làm phao cứu sinh.
    """
    
    # ==========================================================================
    # NHÓM 1: SAFETY REFUSAL (TỪ CHỐI VÌ AN TOÀN)
    # ==========================================================================
    safety_patterns = [
        # 1. Bắt đầu bằng chủ ngữ từ chối (Mạnh nhất)
        # Bắt: "Tôi không thể...", "Hệ thống không được phép...", "AI không hỗ trợ..."
        r"^(?:tôi|chúng tôi|hệ thống|mô hình|ai)?\s*không (?:thể|được|hỗ trợ|có quyền|được phép|chấp nhận)\s*(?:trả lời|cung cấp|chia sẻ|hướng dẫn|thực hiện|làm theo)",
        
        # 2. Bắt cụm từ khóa chính sách/pháp luật tiêu cực
        r"vi phạm (?:pháp luật|chính sách|tiêu chuẩn|đạo đức|quy định)",
        r"trái (?:với)? (?:quy định|pháp luật|thuần phong mỹ tục)",
        r"từ chối trả lời",
        r"nội dung (?:nhạy cảm|người lớn|cấm|độc hại)",
        
        # 3. [MỚI] Bắt hành vi cụ thể đi kèm sự từ chối (Fix lỗi test_0079)
        # Bắt: "không thể... làm giả", "không thể... trốn thuế"
        r"không (?:thể|hỗ trợ).*(?:làm giả|trốn thuế|gian lận|qua mặt|tấn công|hack)"
    ]
    
    # Danh sách loại trừ (False Positive cho Safety)
    # Ví dụ: "A. Hành vi này được xem là vi phạm pháp luật" -> Đây là kiến thức, không phải từ chối.
    safety_exclusions = [
        "được xem là", "bị coi là", "cấu thành tội", "là hành vi", 
        "quy định về", "xử lý hành vi", "dấu hiệu của"
    ]

    for label, text in options_map.items():
        text_lower = str(text).lower().strip()
        
        # Bước 1: Check Exclusion trước
        if any(ex in text_lower for ex in safety_exclusions):
            continue

        # Bước 2: Check Pattern
        if any(re.search(p, text_lower) for p in safety_patterns):
            return label, "SAFETY"

    # ==========================================================================
    # NHÓM 2: NO INFO REFUSAL (TỪ CHỐI VÌ THIẾU TIN)
    # ==========================================================================
    no_info_patterns = [
        # 1. Không có/không đủ thông tin
        r"không (?:có|đủ|tìm thấy) (?:thông tin|dữ liệu|cơ sở|căn cứ|bằng chứng)",
        
        # 2. [MỚI] Không thể xác định (Bắt cả trường hợp đứng cuối câu)
        # Pattern cũ: r"không thể xác định (?:được|từ...)" -> Sai nếu hết câu.
        # Pattern mới: (?:\.|,| |$) nghĩa là sau nó là dấu chấm, phẩy, cách hoặc hết dòng.
        r"không thể (?:xác định|kết luận|tính toán|trả lời)(?:\.|,| |$)",
        
        # 3. Pattern bổ sung
        r"thông tin.*(?:chưa|không).*đủ",
        r"câu hỏi không thể trả lời" 
    ]
    
    # Danh sách loại trừ cho No Info (Tránh bắt nhầm câu kiến thức)
    # Ví dụ: "Đặc điểm không thể thay đổi"
    no_info_exclusions = [
        "không thể thay đổi", "không thể tách rời", "không thể thiếu", 
        "không thể phủ nhận", "không thể tránh khỏi"
    ]

    for label, text in options_map.items():
        text_lower = str(text).lower().strip()
        
        # Loại trừ các từ khóa Safety/Luật để tránh bắt nhầm
        if "vi phạm" in text_lower or "luật" in text_lower: continue
        
        # Check Exclusion No Info
        if any(ex in text_lower for ex in no_info_exclusions):
            continue
        
        if any(re.search(p, text_lower) for p in no_info_patterns):
            return label, "NO_INFO"

    return None, None


async def unified_router_v3(session, question, options_map):
    """
    ROUTER V3 (FINAL) - Có tích hợp 'Answer-Aware Trap Detection'
    """
    q_lower = question.lower()

    # ==========================================================================
    # BƯỚC 0: TRAP DETECTION (QUÉT ĐÁP ÁN TRƯỚC)
    # ==========================================================================
    # Kiểm tra xem có đáp án nào là SAFETY REFUSAL không
    refusal_key, refusal_type = find_true_refusal_key(options_map)
    
    if refusal_type == "SAFETY":
        # PHÁT HIỆN BẪY!
        # Ví dụ câu hỏi Methamphetamine: Router tưởng là Hóa học, nhưng đáp án D bảo là Vi phạm.
        # -> Ghi đè ngay lập tức thành BLOCKED.
        return {
            "is_unsafe": True,
            "is_stem": False,
            "use_large": False,
            "tag": "BLOCKED-TRAP_DETECTED", # Tag riêng để biết bị bắt do đáp án
            "refusal_key": refusal_key
        }

    # ==========================================================================
    # BƯỚC 1: HARD CHECK CÂU HỎI (Như cũ)
    # ==========================================================================
    hard_ban = ["khiêu dâm", "làm tình", "ấu dâm", "kích dục", "cá độ", "lật đổ", "sex", "xxx"]
    if any(w in q_lower for w in hard_ban):
        # Nếu bị ban bởi từ khóa, dùng refusal key tìm được (nếu có), ko thì A
        ans_key = refusal_key if refusal_key else "A" 
        return {"is_unsafe": True, "tag": "BLOCKED-KEYWORD", "refusal_key": ans_key}

    # ==========================================================================
    # BƯỚC 2: PHÂN LOẠI LĨNH VỰC (STEM / LUẬT / XÃ HỘI)
    # ==========================================================================
    
    # ... (Giữ nguyên logic Regex phân loại Math/Legal như các phiên bản trước) ...
    has_math = bool(re.search(r"\$|\\frac|\\int|\\sum", q_lower))
    is_legal = any(w in q_lower for w in ["luật", "nghị định", "thông tư", "quy định"])
    
    # Đặc biệt: Nếu tìm thấy refusal_type là "NO_INFO" (VD: Không thể xác định...)
    # Ta vẫn cho phép chạy Large Model để nó thử tính toán xem có ra kết quả không.
    # Nhưng ta sẽ đánh dấu tag NO_INFO để process_row_logic ưu tiên fallback về đáp án đó.
    
    tag = "NO_INFO_HINT" if refusal_type == "NO_INFO" else ("STEM" if has_math else "SOCIAL")
    use_large = has_math or is_legal or (refusal_type == "NO_INFO")

    return {
        "is_unsafe": False,
        "is_stem": has_math,
        "use_large": use_large,
        "tag": f"ROUTED-{tag}",
        "refusal_key": refusal_key # Truyền key này xuống để dùng nếu cần
    }

import re

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
    # Nếu ít docs thì không cần rerank, trả về luôn cho nhanh
    if not initial_docs or len(initial_docs) <= top_n: 
        return initial_docs

    # [TỐI ƯU] Chỉ lấy Top 15 để rerank (thay vì 20-25)
    candidates = initial_docs[:15]

    docs_text = ""
    for i, doc in enumerate(candidates):
        clean_body = str(doc.get('text', '')).strip().replace("\n", " ")
        # [TỐI ƯU] Giảm xuống 400 ký tự. Model Small chỉ cần thế thôi.
        preview_text = " ".join(clean_body.split())[:400] 
        docs_text += f"ID [{i}]: {preview_text}...\n\n"

    system_prompt = """Bạn là chuyên gia lọc tin.
NHIỆM VỤ: Chọn các ID tài liệu liên quan nhất đến câu hỏi.
OUTPUT JSON: {"ids": [0, 2, ...]}"""

    user_prompt = f"CÂU HỎI: {question}\n\nDANH SÁCH:\n{docs_text}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    try:
        # Gọi API
        response = await call_llm_generic(session, messages, Config.LLM_MODEL_SMALL, stats, max_tokens=60)
        
        if response:
            found_indices = [int(s) for s in re.findall(r'\d+', response)]
            valid_docs = []
            seen = set()
            for idx in found_indices:
                if 0 <= idx < len(candidates) and idx not in seen:
                    valid_docs.append(candidates[idx])
                    seen.add(idx)
            
            # Backfill nếu thiếu
            if len(valid_docs) < top_n:
                for i, doc in enumerate(candidates):
                    if i not in seen:
                        valid_docs.append(doc)
                        if len(valid_docs) >= top_n: break
            
            return valid_docs[:top_n]

    except Exception as e:
        logger.warning(f"Rerank Error: {e}")
    
    # Fallback: Trả về danh sách gốc nếu lỗi
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
    
def heuristic_answer_math(options_map):
    """
    Fallback chuyên dụng cho STEM:
    1. Ưu tiên đáp án có chứa Số (Digits).
    2. Nếu nhiều đáp án có số, chọn theo thống kê (thường là C hoặc B).
    3. Nếu không có số, chọn đáp án dài nhất.
    """
    # Lọc các đáp án có chứa con số
    numeric_opts = [k for k, v in options_map.items() if any(c.isdigit() for c in str(v))]
    
    if numeric_opts:
        # Nếu có đáp án chứa số, ưu tiên chọn C nếu C nằm trong đó (Mẹo thi trắc nghiệm)
        if 'C' in numeric_opts: return 'C'
        if 'B' in numeric_opts: return 'B'
        return numeric_opts[0]
    
    # Nếu không có số, fallback về C (Option an toàn nhất trong trắc nghiệm)
    if 'C' in options_map: return 'C'
    
    return list(options_map.keys())[0]

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

# --- THAY THẾ build_prompt CŨ ---
def build_cot_prompt(question, options_text, docs, is_stem=False):
    context = ""
    CHAR_LIMIT = 3500 
    
    for i, doc in enumerate(docs):
        # Cắt theo ký tự để kiểm soát token chính xác
        clean_text = doc['text'].strip()[:CHAR_LIMIT] 
        context += f"[Tài liệu {i+1}]: {clean_text}\n\n"


    if is_stem:
        system_prompt = """Bạn là CHUYÊN GIA PHÂN TÍCH ĐỊNH LƯỢNG trong các lĩnh vực:
- Khoa học tự nhiên (Toán, Lý, Hóa, STEM)
- Kinh tế học & Tài chính (vi mô, vĩ mô, thống kê, tối ưu)

NHIỆM VỤ:
Giải quyết chính xác các bài toán trắc nghiệm, áp dụng suy luận khoa học và tính toán chi tiết.

NGUYÊN TẮC BẮT BUỘC:
1. Đọc và phân tích kỹ đề bài trước khi đưa ra lời giải.
2. Xác định các công thức, lý thuyết và dữ liệu cần thiết để giải bài toán.
3. **Suy luận chuỗi bước (chain-of-thought)**:
- **Bước 1**: Phân tích vấn đề và xác định mục tiêu của bài toán.
- **Bước 2**: Xác định các yếu tố đầu vào, biến số và giả định cần thiết. Liệt kê các biến số từ đề bài ($R$, $C$, $L$, $v$, $t$...). ĐỔI NGAY LẬP TỨC về đơn vị chuẩn SI (Ví dụ: $100 \mu F \rightarrow 100 \times 10^{-6} F$, $cm \rightarrow m$, $km/h \rightarrow m/s$). Tuyệt đối không tính toán khi chưa đổi đơn vị.
- **Bước 3**: Chọn công thức hoặc phương pháp giải phù hợp với dữ liệu có sẵn.
- **Bước 4**: Thực hiện các phép tính chi tiết, giải thích từng bước (bao gồm các phép toán trung gian nếu có).
- **Bước 5**: Loại bỏ các đáp án sai dựa trên quá trình tính toán logic và chọn đáp án đúng từ các lựa chọn (A, B, C, D).
4. Đảm bảo không đoán mò và chỉ chọn đáp án khi đã có cơ sở tính toán rõ ràng, minh bạch.
5. **Trung thực**: Nếu [DỮ LIỆU] thiếu thông tin để tính, hãy kiểm tra xem có đáp án "Không xác định/Không có thông tin" không.

LƯU Ý QUAN TRỌNG:
- Nếu bài toán yêu cầu tính toán dựa trên văn bản (ví dụ: Tăng trưởng GDP, Lãi suất), PHẢI lấy số liệu từ [DỮ LIỆU THAM KHẢO].
- Với câu hỏi CÔNG THỨC LÝ THUYẾT (Toán/Lý/Hóa) hoặc ĐỊNH NGHĨA: 
    - Nếu [DỮ LIỆU THAM KHẢO] chứa công thức KHÁC với kiến thức chuẩn của bạn (ví dụ: tài liệu nói về 'nối tiếp' trong khi hỏi 'song song'), HÃY DÙNG KIẾN THỨC CHUẨN ĐỂ SỬA SAI.
    - Ưu tiên độ chính xác khoa học tuyệt đối.
- Nếu đây là BÀI TẬP GIÁO KHOA hoặc TÌNH HUỐNG GIẢ ĐỊNH (ví dụ: "Một công ty có...", "Giả sử..."), và không tìm thấy dữ liệu trong tài liệu tham khảo, HÃY DÙNG KIẾN THỨC CHUYÊN MÔN CỦA BẠN để giải quyết.
- Nếu không có dữ liệu trong văn bản, hãy dùng kiến thức chuẩn của bạn.
- Tuyệt đối chính xác về đơn vị tính.

ĐỊNH DẠNG TRẢ LỜI:
- **Bước 1**: Phân tích bài toán, xác định các yếu tố cần tính.
- **Bước 2**: Lựa chọn công thức hoặc phương pháp giải.
- **Bước 3**: Tính toán chi tiết, thực hiện các phép toán và giải thích từng bước trung gian.
- **Bước 4**: Chọn đáp án đúng (A, B, C, D, E, F, ...) theo số lượng thực tế đáp án trong câu hỏi và giải thích lý do tại sao đó là đáp án chính xác.

Mục tiêu là **chọn đáp án đúng** và giải thích đầy đủ quá trình tính toán logic, tránh sai sót hoặc bỏ qua bước nào trong suy luận.
"""
    else:
        current_date = get_current_date_str()

        system_prompt = f"""Bạn là CHUYÊN GIA KHOA HỌC XÃ HỘI & PHÁP LÝ.
Thời điểm hiện tại: {current_date}.

NHIỆM VỤ: Trả lời câu hỏi trắc nghiệm dựa trên TÀI LIỆU được cung cấp.

QUY TRÌNH TƯ DUY (BẮT BUỘC):
1. **Đối chiếu & Xác thực**:
- Tìm từ khóa trong [DỮ LIỆU].
- Chú ý **Hiệu lực văn bản**: Nếu tài liệu quá cũ so với thời điểm hiện tại ({current_date}), hãy lưu ý khi chọn đáp án.
- So sánh chi tiết (ngày tháng, con số, chủ ngữ) với từng lựa chọn A, B, C, D.

2. **Xử lý Phủ định (QUAN TRỌNG)**: 
- Nếu câu hỏi có từ "KHÔNG", "NGOẠI TRỪ", "SAI":
- Tìm các đáp án ĐÚNG trong tài liệu -> Loại bỏ chúng.
- Đáp án còn lại (hoặc đáp án được tài liệu bảo là "không cần/không phải") là ĐÁP ÁN ĐÚNG.

3. **Xử lý Thiếu tin (Refusal)**: 
- Nếu [DỮ LIỆU] hoàn toàn không nhắc đến vấn đề: Ưu tiên chọn đáp án "Không có thông tin" / "Không được đề cập".
- Nếu không có đáp án đó, mới dùng kiến thức chuẩn xác của bạn để trả lời (nhưng phải cẩn trọng).

4. **An toàn (Safety)**: 
- Nếu câu hỏi yêu cầu hành vi vi phạm pháp luật/kích động -> Chọn đáp án mang ý nghĩa TỪ CHỐI (hoặc A).

ĐỊNH DẠNG TRẢ LỜI:
### PHÂN TÍCH:
- Tìm thấy tại [Tài liệu X]: "..."
- A: [Đúng/Sai] vì...
- B: [Đúng/Sai] vì...
### ĐÁP ÁN: [Ký tự in hoa]"""
 
    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

LỰA CHỌN:
{options_text}

YÊU CẦU: Hãy suy luận và trả lời đúng theo định dạng:
### PHÂN TÍCH: ...
### ĐÁP ÁN: ...
"""
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

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
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.bm25_data = None
        
        # Load BM25 Lightweight
        if BM25_FILE.exists():
            try:
                with open(BM25_FILE, "rb") as f: 
                    self.bm25_data = pickle.load(f)
                # Kiểm tra version để chắc chắn đúng format mới
                ver = self.bm25_data.get('version', 1)
                logger.info(f"BM25 loaded: {len(self.bm25_data.get('chunk_ids', []))} chunks (Ver: {ver})")
            except Exception as e:
                logger.error(f"Failed to load BM25: {e}")

    async def search(self, session, query, top_k=TOP_K):
        # ---------------------------------------------------------
        # 1. VECTOR SEARCH (Lấy kết quả từ Qdrant - Đã có Text)
        # ---------------------------------------------------------
        query_vec = await get_embedding_async(session, query)
        
        vec_hits_map = {} # Map: chunk_id -> Payload (chứa text, title)
        vec_scores = {}   # Map: chunk_id -> Score
        
        if query_vec:
            try:
                res = await self.client.query_points(
                    Config.COLLECTION_NAME, 
                    query=query_vec, 
                    limit=top_k, 
                    with_payload=True
                )
                for point in res.points:
                    cid = point.payload['chunk_id']
                    vec_hits_map[cid] = point.payload 
                    vec_scores[cid] = point.score
            except Exception as e:
                logger.error(f"Vector search error: {e}")

        # ---------------------------------------------------------
        # 2. BM25 SEARCH (Chỉ lấy ID và Score - KHÔNG LẤY TEXT)
        # ---------------------------------------------------------
        bm25_scores = {}
        
        if self.bm25_data:
            try:
                tokens = word_tokenize(query.lower())
                bm25_obj = self.bm25_data['bm25_obj']
                all_ids = self.bm25_data['chunk_ids']
                
                # Tính điểm
                scores = bm25_obj.get_scores(tokens)
                
                # Lấy Top 2*k candidates từ BM25
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
                
                for idx in top_idxs:
                    score = scores[idx]
                    if score > 0: 
                        chunk_id = all_ids[idx]
                        bm25_scores[chunk_id] = score
            except Exception as e:
                logger.error(f"BM25 search error: {e}")

        # ---------------------------------------------------------
        # [MỚI] 3. FETCH MISSING TEXT (Cứu những thằng BM25 tìm thấy mà Vector bỏ qua)
        # ---------------------------------------------------------
        # Tìm những ID nằm trong BM25 top nhưng chưa có trong Vector Hits
        missing_ids = [cid for cid in bm25_scores.keys() if cid not in vec_hits_map]
        
        if missing_ids:
            try:
                # Convert chunk_id sang UUID để query Qdrant (theo logic ingest cũ)
                point_ids = [generate_uuid5(cid) for cid in missing_ids]
                
                # Gọi Qdrant lấy text cho các ID này
                points = await self.client.retrieve(
                    collection_name=Config.COLLECTION_NAME,
                    ids=point_ids,
                    with_payload=True
                )
                
                # Đưa vào map chung
                for point in points:
                    if point.payload:
                        cid = point.payload['chunk_id']
                        vec_hits_map[cid] = point.payload
                        # Gán điểm vector = 0 (vì vector search không tìm thấy)
                        vec_scores[cid] = 0.0
                        
            except Exception as e:
                logger.error(f"Fetch missing text error: {e}")

        # ---------------------------------------------------------
        # 4. FUSION (Kết hợp điểm số)
        # ---------------------------------------------------------
        final_results = []
        
        # Chuẩn hóa điểm số (Normalization)
        max_vec = max(vec_scores.values()) if vec_scores else 1.0
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        
        # Tập hợp tất cả ID tìm được
        all_candidate_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        
        for cid in all_candidate_ids:
            # Nếu vẫn không có payload (do lỗi fetch hoặc ID sai) -> Bỏ qua
            if cid not in vec_hits_map:
                continue
                
            v_score = vec_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)
            
            # Công thức Hybrid
            norm_v = v_score / max_vec if max_vec > 0 else 0
            norm_b = b_score / max_bm25 if max_bm25 > 0 else 0
            
            final_score = (norm_v * ALPHA_VECTOR) + (norm_b * (1 - ALPHA_VECTOR))
            
            payload = vec_hits_map[cid]
            final_results.append({
                "chunk_id": cid,
                "text": payload.get('text', ''), # Lấy text từ payload Qdrant
                "title": payload.get('title', ''),
                "score": final_score
            })
            
        # Sắp xếp giảm dần theo điểm tổng hợp
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]

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
    """
    Chế độ KIÊN TRÌ: Retry vô hạn cho đến khi lấy được kết quả 200 OK.
    """
    # 1. Xếp hàng (Vẫn cần Limiter để không bị Ban IP vĩnh viễn)
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
        "temperature": 0.1, 
        "top_p": 0.95,      
        "max_completion_tokens": max_tokens
    }

    # Jitter
    await asyncio.sleep(random.uniform(1.0, 3.0))

    attempt = 0
    
    empty_response_count = 0 # Đếm số lần bị trả về rỗng

    # Vòng lặp vô tận (cho các lỗi mạng 429/5xx)
    while True:
        try:
            async with session.post(url, json=payload, headers=headers, timeout=120, ssl=False) as resp:
                
                # CASE A: THÀNH CÔNG HOẶC RỖNG
                if resp.status == 200:
                    try:
                        d = await resp.json()
                        if 'choices' in d and len(d['choices']) > 0: 
                            return d['choices'][0]['message']['content']
                        
                        # [FIX QUAN TRỌNG] Xử lý Empty Response
                        logger.warning(f"⚠️ Empty Response (200 OK).")
                        empty_response_count += 1
                        
                        # Nếu bị rỗng 3 lần liên tiếp -> Bỏ cuộc (Model không trả lời được câu này)
                        if empty_response_count >= 3:
                            logger.error(f"❌ Model refuses to answer (Empty 3 times). Skipping.")
                            return None
                        
                        await asyncio.sleep(2)
                        continue
                    except: 
                        await asyncio.sleep(2)
                        continue
                
                # CASE B: LỖI MẠNG/QUOTA (Vẫn retry vô hạn như cũ)
                elif resp.status in [401, 429, 500, 502, 503, 504]:
                    empty_response_count = 0 # Reset counter nếu gặp lỗi mạng
                    wait_time = 30 # Chờ cố định 30s
                    logger.warning(f"⏳ API {resp.status}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    return None

        except Exception as e:
            logger.warning(f"🔌 Net Error: {e}")
            await asyncio.sleep(5)

# ==============================================================================
# 3. CORE LOGIC (PROCESS SINGLE ROW)
# ==============================================================================

async def process_row_logic(session, retriever, row, stats=None):
    qid = row.get('qid', row.get('id', 'unknown'))
    question = row.get('question', '')
    opts = get_dynamic_options(row)
    opt_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    
    # 1. ROUTING
    route = await unified_router_v3(session, question, opts)
    
    if route["is_unsafe"]:
        ans = route["refusal_key"]
        logger.info(f"🚫 Q:{qid} {route['tag']} -> Ans:{ans}")
        return {"qid": qid, "answer": ans}

    # 2. RETRIEVAL
    top_k = 8 if route["is_stem"] else 12
    docs = await retriever.search(session, question, top_k=top_k)
    context_text = " ".join([d['text'].lower() for d in docs])
    ctx_len = len(context_text)

    # 3. MODEL & PROMPT
    SAFE_LIMIT_LARGE = 37500
    use_large = route["use_large"]
    
    if ctx_len > SAFE_LIMIT_LARGE:
        use_large = False
    
    model = Config.LLM_MODEL_LARGE if use_large else Config.LLM_MODEL_SMALL
    
    if route["is_stem"]:
        msgs = build_cot_prompt(question, opt_text, docs, is_stem=True)
    elif model == Config.LLM_MODEL_LARGE:
        msgs = build_cot_prompt(question, opt_text, docs, is_stem=False)
    else:
        msgs = build_simple_prompt(question, opt_text, docs)

    # 4. INFERENCE (Sẽ chờ đến khi thành công)
    raw = await call_llm_generic(session, msgs, model, stats)
    
    # Nếu raw là None ở đây thì chỉ có thể là lỗi 400 Bad Request (Fatal)
    # Ta thử cứu bằng Small model 1 lần
    if not raw:
        logger.warning(f"⚠️ Large Model Fatal Error. Trying Small...")
        raw = await call_llm_generic(session, msgs, Config.LLM_MODEL_SMALL, stats)

    # 5. REFUSAL HANDLING
    refusal_phrases = ["không có thông tin", "không tìm thấy", "không được đề cập", "không đủ cơ sở"]
    if raw and any(p in raw.lower() for p in refusal_phrases):
        no_info_opt = find_no_info_key(opts)
        if no_info_opt:
            return {"qid": qid, "answer": no_info_opt}
        
        # Force Knowledge
        force_msgs = [
            {"role": "system", "content": "Dùng kiến thức của bạn để chọn đáp án đúng nhất A/B/C/D. Không giải thích."},
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn:\n{opt_text}"}
        ]
        raw = await call_llm_generic(session, force_msgs, model, stats)

    # 6. EXTRACT
    ans = extract_answer_strict(raw, opts)

    # 7. ANTI-TRAP
    if ans:
        potential_trap, trap_type = find_true_refusal_key(opts)
        if ans == potential_trap and trap_type == "SAFETY":
             ans = None 

    # --- [THAY ĐỔI] BỎ HEURISTIC ---
    # Code cũ: if not ans: ans = heuristic...
    # Code mới: Nếu không extract được, ghi log lỗi và để trống (hoặc mặc định A để file không lỗi format)
    if not ans:
        logger.error(f"❌ Q:{qid} Failed to extract answer after AI call. Raw: {str(raw)[:50]}...")
        ans = "A" # Fallback cuối cùng để không gãy file CSV, nhưng không dùng thuật toán đếm từ.

    mod_name = model.split('_')[-1].upper()
    logger.info(f"Q:{qid} | Tag:{route['tag']} | Mod:{mod_name} | Ans:{ans}")

    return {"qid": qid, "answer": ans}


# ==============================================================================
# 4. MAIN LOOP WITH RESUME
# ==============================================================================
async def main():
    # 1. Load Data
    # files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    files = [Config.BASE_DIR / "data" / "test.json"]
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
            
            # Không cần vòng lặp retry ở đây nữa, vì call_llm_generic đã retry vô hạn
            try:
                # Bỏ asyncio.wait_for hoặc set timeout=None
                result = await process_row_logic(session, retriever, row, stats)
                
                df_res = pd.DataFrame([result])
                df_res[['qid', 'answer']].to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)
                
            except Exception as e:
                logger.error(f"❌ Error Q:{qid}: {e}")
                # Vẫn ghi A để bảo toàn số lượng câu
                pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)

            # Nghỉ 1 chút sau mỗi câu để chắc ăn
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

    