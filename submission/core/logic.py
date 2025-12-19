import logging
import asyncio
from typing import Dict, Any

import aiohttp

# Import từ các module đã tách
from config import Config
from utils.logger import logger, write_debug_log
from utils.text_utils import (
    get_dynamic_options,
    find_true_refusal_key,
    find_no_info_key,
    extract_answer_strict,
    heuristic_answer_math,
    heuristic_answer_overlap
)

from core.router import unified_router_v3
from core.retriever import HybridRetriever
from core.llm_client import call_llm_generic
from core.prompts import (
    build_cot_prompt,
    build_simple_prompt,
    build_rag_instruction_fixed
)


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
        context_text = " ".join([d['text'].lower() for d in docs])
        limit_note = f"(Trimmed context: {len(context_text)} docs)"
    
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