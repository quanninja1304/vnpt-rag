import asyncio
import json
import logging
import random
import aiohttp
from typing import List, Dict, Optional, Any

from config import Config, LIMITER_LARGE, LIMITER_SMALL, LIMITER_EMBED

logger = logging.getLogger("VNPT_BOT")

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