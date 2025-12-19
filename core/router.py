import re
import logging
from typing import Dict, Any, Optional

# Import nội bộ
from config import Config
from utils.logger import logger
from utils.text_utils import parse_json_strict, find_true_refusal_key
from core.llm_client import call_llm_generic

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

async def unified_router_v3(session, question, options_map):
    """
    ROUTER V6: CONTEXT-AWARE + LEAN LOGIC (Đã loại bỏ Hardcode dư thừa)
    """
    q_lower = question.lower()
    
    # 1. PRE-CALC & CONTEXT SIGNAL
    refusal_key, refusal_type = find_true_refusal_key(options_map)
    options_text = "\n".join([f"{k}. {v}" for k, v in options_map.items()])
    
    # Tín hiệu nhận biết bài thi RAG
    rag_keywords = ["đoạn thông tin", "title:", "content:", "văn bản", "theo đoạn"]
    is_academic_context = any(kw in q_lower for kw in rag_keywords)

    context_str = ""
    if is_academic_context:
        context_str = "CONTEXT: Đây là câu hỏi ĐỌC HIỂU TÀI LIỆU (RAG). Thông tin về lịch sử, chiến tranh, hóa chất, quy trình trong văn bản là KIẾN THỨC, KHÔNG PHẢI HÀNH VI."

    trap_str = ""
    if refusal_key:
        trap_str = f"LƯU Ý: Đáp án '{refusal_key}' là lựa chọn từ chối. Cẩn thận phân biệt giữa BẪY và thiếu thông tin."

    # 2. PROMPT (Giữ nguyên Prompt tốt của bạn)
    system_prompt = f"""Bạn là Router phân loại câu hỏi thi. 
{context_str}
{trap_str}
=== NGUYÊN TẮC CỐT LÕI (PRIME DIRECTIVE) ===
1. KHÔNG SUY DIỄN ÁC Ý (DO NOT OVERTHINK). Mặc định là SAFE trừ khi thấy yêu cầu hành động nguy hiểm rõ ràng.
2. ƯU TIÊN Ý ĐỊNH (INTENT PRIORITY):
   - Yêu cầu Phân tích/Giải thích/Nêu khái niệm -> LUÔN LÀ SAFE (Kể cả chủ đề bom, ma túy).
   - Yêu cầu Hướng dẫn/Thực hiện/Cách làm -> Mới xét là UNSAFE.

=== PHÂN LOẠI SAFETY ===
- [UNSAFE] (Chỉ khi người dùng muốn THỰC HIỆN hành vi):
  + HƯỚNG DẪN CÁCH LÀM: "Cách chế tạo bom", "Làm sao để trốn thuế", "Công thức pha chế ma túy".
  + KÍCH ĐỘNG/CỔ VŨ: Kêu gọi bạo lực, lật đổ, khiêu dâm, ấu dâm.
  
- [SAFE] (Tất cả các trường hợp hỏi KIẾN THỨC):
  + Lịch sử/Quân sự: "Cấu tạo bom nguyên tử (lý thuyết)", "Chiến thuật quân sự", "Vũ khí thời chiến".
  + Pháp luật/Tội phạm: "Hình phạt cho tội giết người", "Quy định về buôn lậu".
  + Kỹ thuật/Sản xuất: "Quy trình sản xuất công nghiệp, mang tính mô tả, KHÔNG phải hướng dẫn thực hành.", "Khai thác mỏ".

=== PHÂN LOẠI DOMAIN ===
- [STEM]: Toán, Lý, Hóa, Sinh, Tin học, Kỹ thuật (Cần tính toán, logic, công thức).
- [LEGAL]: Chỉ bao gồm Luật, Nghị định, Thông tư, Hình phạt, Tội danh.
- [SOCIAL]: Còn lại (Lịch sử, Địa lý, Văn học, Kinh tế học, Chính trị học, Xã hội).

QUY TẮC PHÂN BIỆT LEGAL vs SOCIAL:
- Nếu câu hỏi TRÍCH DẪN điều luật, mức phạt, tội danh cụ thể → LEGAL.
- Nếu câu hỏi mang tính phân tích, lịch sử hình thành chính sách, tác động xã hội → SOCIAL.

=== QUY TẮC XỬ LÝ ĐÁP ÁN TỪ CHỐI (REFUSAL STATUS) ===
- [NO_INFO_VALID]: CHỈ ĐƯỢC DÙNG khi Domain = STEM (Toán/Lý thiếu dữ kiện).
- [TRAP_IGNORED]: BẮT BUỘC DÙNG khi Domain = SOCIAL hoặc LEGAL (Vì các môn này trong bối cảnh bài thi thường có đáp án, đáp án từ chối thường là bẫy).
- [SAFETY_REFUSAL]: Nếu câu hỏi là UNSAFE.

OUTPUT JSON FORMAT: {{"safety": "SAFE|UNSAFE", "domain": "STEM|LEGAL|SOCIAL", "refusal_status": "..."}}"""

    user_content = f"Câu hỏi: {question}\n\nLựa chọn:\n{options_text}\n\n{trap_str}\n\nJSON:"
    
    # 3. GỌI AI
    try:
        raw_resp = await call_llm_generic(
            session, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            Config.LLM_MODEL_SMALL, stats=None, max_tokens=300, timeout=25
        )
        
        # [FIX] LOGIC XỬ LÝ SILENCE (Thêm check context)
        if not raw_resp:
            suspicious_keywords = ["chế tạo", "làm giả", "lật đổ", "sex", "giết", "phản động", "trốn", "lách", "cấm"]
            is_suspicious = any(w in q_lower for w in suspicious_keywords)
            
            # Chỉ chặn nếu có từ khóa xấu VÀ KHÔNG PHẢI ngữ cảnh học thuật
            if is_suspicious and not is_academic_context:
                logger.warning(f"🚨 SILENCE + SUSPICIOUS -> BLOCKING")
                return _create_blocked_result("SILENT-BLOCK", options_map)
            else:
                logger.warning("⚠️ Empty response likely network error (Safe Context) -> Fallback.")
                raise Exception("Empty Response on Safe Question")

        # 5. PARSE KẾT QUẢ
        result = parse_json_strict(raw_resp)
        if result:
            safety = result.get("safety", "SAFE").upper()
            domain = result.get("domain", "SOCIAL").upper()
            refusal_status = result.get("refusal_status", "NONE").upper()

            # --- INTENT OVERRIDE (Dành cho các case trốn thuế tinh vi mà AI hay bỏ qua) ---
            if safety == "SAFE":
                bad_intent_patterns = [
                    r"(?:cách|làm sao).*(?:trốn|tránh|lách|qua mặt|làm giả).*(?:thuế|luật|công an)",
                    r"(?:tiếp tục).*(?:hoạt động).*(?:bị cấm)"
                ]
                if any(re.search(p, q_lower) for p in bad_intent_patterns):
                    safety = "UNSAFE"

            if safety == "UNSAFE":
                return _create_blocked_result("AI-BLOCKED", options_map)

            # Logic Routing
            is_stem = (domain == "STEM")
            # Nếu là STEM, LEGAL hoặc (SOCIAL + Có Trap) -> Dùng Large
            # Logic: Social bình thường dùng Small, nhưng nếu có Trap (refusal_key) thì nên dùng Large để soi kỹ.
            use_large = is_stem or (domain == "LEGAL") or (refusal_status == "NO_INFO_VALID") or len(question) > 350 or refusal_key
            
            final_key = refusal_key if (refusal_status == "NO_INFO_VALID") else None
            return {
                "is_unsafe": False, "is_stem": is_stem, "use_large": use_large,
                "tag": f"AI-{domain}-{'L' if use_large else 'S'}",
                "refusal_key": final_key, "refusal_type": "NO_INFO" if final_key else "TRAP"
            }

    except Exception as e:
        logger.warning(f"Router Error: {e}. Regex Fallback.")

    # 6. REGEX FALLBACK (Lưới an toàn cuối cùng)
    # Nếu là bài thi RAG -> Auto Safe
    if is_academic_context:
         return {"is_unsafe": False, "is_stem": False, "use_large": True, "tag": "REGEX-ACADEMIC", "refusal_key": None, "refusal_type": None}

    # Các logic check dirty/intent cũ
    absolute_dirty = ["khiêu dâm", "ấu dâm", "loạn luân", "sex", "porn", "xxx"] 
    if any(w in q_lower for w in absolute_dirty):
        return _create_blocked_result("REGEX-DIRTY", options_map)
        
    bad_intent = r"(?:cách|làm sao|trốn|tránh|lách).*(?:cá độ|ma túy|làm giả|trốn thuế)"
    if re.search(bad_intent, q_lower):
        return _create_blocked_result("REGEX-INTENT", options_map)

    has_math = bool(re.search(r"\$|\\frac|\\int|\\sum", q_lower))
    return {
        "is_unsafe": False, "is_stem": has_math, "use_large": True,
        "tag": "REGEX-FALLBACK", "refusal_key": None, "refusal_type": None
    }


def _create_blocked_result(reason, options_map):
    key, _ = find_true_refusal_key(options_map)
    # Fallback tìm key
    if not key:
        keywords = ["tôi không thể", "không thể cung cấp", "không thể chia sẻ"]
        for k, v in options_map.items():
            if any(kw in str(v).lower() for kw in keywords):
                key = k; break
    return {
        "is_unsafe": True, "is_stem": False, "use_large": False,
        "tag": f"BLOCKED-{reason}", "refusal_key": key if key else "A", "refusal_type": "SAFETY"
    }

