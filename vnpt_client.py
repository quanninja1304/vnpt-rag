import requests
import time
import traceback
from config import Config

def get_vnpt_embedding(text, max_retries=3):
    """
    Gọi API Embedding VNPT.
    """
    model = Config.MODEL_EMBEDDING_API
    # Theo PDF Trang 11: Endpoint là .../vnptai-hackathon-embedding
    # Config.VNPT_EMBEDDING_URL đã được set chính xác trong config.py
    url = Config.VNPT_EMBEDDING_URL 
    
    creds = Config.VNPT_CREDENTIALS.get(model)
    if not creds:
        print(f"❌ Config Error: Không tìm thấy credentials cho {model}")
        return None
    
    headers = {
        'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
        'Token-id': creds['token_id'],
        'Token-key': creds['token_key'],
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": model,
        "input": text,
        "encoding_format": "float"
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data['data'][0]['embedding']
                else:
                    print(f"⚠️ API trả về 200 nhưng không có data: {data}")
                    return None
                    
            elif response.status_code == 429: # Rate Limit
                wait_time = 2 * (attempt + 1)
                print(f"⏳ Embed Rate Limit (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            elif response.status_code >= 500: # Server Error
                print(f"⚠️ Server Error {response.status_code}. Retrying...")
                time.sleep(1)
                continue
                
            else: # 400, 401, 404... -> Lỗi Config, không Retry
                print(f"❌ Embed Error {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network Error (Embed): {e}")
            time.sleep(1)
            
    return None

def call_vnpt_llm(messages, model=Config.LLM_MODEL_LARGE):
    """
    Gọi LLM sinh câu trả lời.
    Có cơ chế Fallback thông minh chỉ khi gặp lỗi Server/Mạng.
    """
    # [QUAN TRỌNG] Theo PDF Trang 7: Endpoint chứa tên model
    url_model_name = model.replace("_", "-")
    url = f"{Config.VNPT_API_URL}/{url_model_name}"
    
    creds = Config.VNPT_CREDENTIALS.get(model)
    if not creds:
        return f"Lỗi Config: Không có creds cho {model}"
    
    headers = {
        'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
        'Token-id': creds['token_id'],
        'Token-key': creds['token_key'],
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 20,
        "max_completion_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        
        # 1. Thành công
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        
        # 2. Lỗi có thể Fallback (429: Hết quota, 5xx: Server sập)
        elif response.status_code == 429 or response.status_code >= 500:
            print(f"⚠️ API {model} lỗi {response.status_code}. Đang Fallback...")
            if model == Config.LLM_MODEL_LARGE:
                return call_vnpt_llm(messages, model=Config.LLM_MODEL_SMALL)
            return "Xin lỗi, hệ thống đang quá tải."
            
        # 3. Lỗi Config (400, 401) -> KHÔNG Fallback để biết mà sửa
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            return "Lỗi cấu hình hệ thống."

    # 4. Lỗi Mạng (Timeout, Connection Refused) -> Fallback
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network Error ({model}): {e}")
        if model == Config.LLM_MODEL_LARGE:
            print("🔄 Switching to Small Model...")
            return call_vnpt_llm(messages, model=Config.LLM_MODEL_SMALL)
        return "Lỗi kết nối mạng."
    
    # 5. Lỗi Code Python (KeyError, ValueError...) -> Crash để debug, KHÔNG Fallback
    except Exception as e:
        print(f"❌ Code Error in call_vnpt_llm: {e}")
        traceback.print_exc() # In chi tiết lỗi dòng nào
        return "Lỗi xử lý nội bộ."

# --- TEST ---
if __name__ == "__main__":
    # Test Embed
    print("Testing Embedding...")
    vec = get_vnpt_embedding("Test")
    print(f"Vector dim: {len(vec) if vec else 'None'}")
    
    # Test LLM
    print("\nTesting LLM...")
    msg = [{"role": "user", "content": "Xin chào"}]
    print(call_vnpt_llm(msg))