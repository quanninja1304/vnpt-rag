import re
import json
import uuid
import logging
from collections import defaultdict
from typing import Dict, Optional, Any, List
from underthesea import word_tokenize

logger = logging.getLogger("VNPT_BOT")

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

def generate_uuid5(unique_string):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(unique_string)))

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