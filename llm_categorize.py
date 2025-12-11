import google.generativeai as genai
import pandas as pd
import json
import time
import os
from tqdm import tqdm
from config import Config

# --- CẤU HÌNH GEMINI ---
GEMINI_API_KEY = "AIzaSyC5kbuXLInHNLX4S6OWCkGGeZh4NPHtIyA" 

# Cấu hình Model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest') # Dùng Flash cho nhanh và quota cao

# Danh sách danh mục chuẩn (để Gemini chọn)
TARGET_CATEGORIES = [
    "Lịch sử Việt Nam", "Địa lý & Hành chính", "Pháp luật & Nhà nước", 
    "Văn hóa & Xã hội", "Kinh tế & Doanh nghiệp", "Quân sự & Quốc phòng",
    "Nhân vật lịch sử", "Giáo dục & Y tế", "Khoa học & Kỹ thuật"
]

def clean_json_string(text):
    """Làm sạch string trả về từ Gemini để parse JSON"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def classify_batch(titles_batch):
    """Gửi 1 lô tiêu đề cho Gemini phân loại"""
    prompt = f"""
    Bạn là một chuyên gia phân loại dữ liệu RAG. 
    Hãy phân loại danh sách các Tiêu đề bài viết Wikipedia sau vào 1 trong các nhóm: {json.dumps(TARGET_CATEGORIES, ensure_ascii=False)}.
    
    Quy tắc:
    1. Trả về định dạng JSON: {{"Tiêu đề 1": "Nhóm 1", "Tiêu đề 2": "Nhóm 2"}}
    2. Nếu không chắc chắn, hãy chọn nhóm phù hợp nhất hoặc "Tổng hợp".
    3. KHÔNG giải thích, chỉ trả về JSON thuần.

    Danh sách tiêu đề:
    {json.dumps(titles_batch, ensure_ascii=False)}
    """
    
    try:
        # Gọi Gemini
        response = model.generate_content(prompt)
        json_str = clean_json_string(response.text)
        return json.loads(json_str)
    except Exception as e:
        print(f"⚠️ Lỗi Batch Gemini: {e}")
        return {}


def run_gemini_categorization():
    input_file = Config.LATEST_CHUNKS_FILE
    
    if not input_file.exists():
        print("❌ Chưa có file chunks. Chạy chunking.py trước!")
        return

    print(f"📂 Đang đọc: {input_file}")
    df = pd.read_parquet(input_file)
    
    # 1. Lấy danh sách tiêu đề duy nhất
    unique_titles = df['doc_title'].unique().tolist()
    print(f"🔍 Tìm thấy {len(unique_titles)} bài viết duy nhất.")
    
    # Checkpoint (để lỡ mạng lag không mất công chạy lại)
    cache_file = "gemini_categories_cache.json"
    title_to_cat = {}
    
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            title_to_cat = json.load(f)
        print(f"🔄 Đã load {len(title_to_cat)} bài từ cache.")
        
    # Lọc những bài chưa làm
    titles_to_process = [t for t in unique_titles if t not in title_to_cat]
    print(f"🔥 Cần phân loại: {len(titles_to_process)} bài.")

    # 2. Chạy Batching
    BATCH_SIZE = 40 # Gửi 40 tiêu đề 1 lần (Flash chịu tốt)
    
    # Thanh tiến trình
    pbar = tqdm(total=len(titles_to_process))
    
    for i in range(0, len(titles_to_process), BATCH_SIZE):
        batch = titles_to_process[i : i + BATCH_SIZE]
        
        # Gọi Gemini
        results = classify_batch(batch)
        
        # Lưu kết quả
        title_to_cat.update(results)
        
        # Cập nhật tiến trình
        pbar.update(len(batch))
        
        # Lưu Cache mỗi 5 batch (an toàn)
        if i % (BATCH_SIZE * 5) == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(title_to_cat, f, ensure_ascii=False)
        
        # Rate Limit Sleep (Gemini Free Tier giới hạn 15 req/phút -> 4s/req)
        # Batch 40 bài x 15 req = 600 bài/phút -> 50k bài mất ~1.5 tiếng
        time.sleep(4) 

    pbar.close()
    
    # Lưu cache lần cuối
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(title_to_cat, f, ensure_ascii=False)

    # 3. Map ngược lại vào DataFrame và cập nhật vector_text
    print("🔄 Đang cập nhật dữ liệu gốc...")
    
    def apply_update(row):
        title = row['doc_title']
        # Lấy category từ Gemini, nếu lỗi/thiếu thì dùng cái cũ
        new_cat = title_to_cat.get(title, row['doc_category'])
        
        # Cập nhật vector_text
        # Format cũ trong chunking.py: "Tiêu đề: ...\nLĩnh vực: ...\nNội dung: ..."
        old_vec_text = row['vector_text']
        
        # Thay thế dòng Lĩnh vực cũ bằng cái mới
        lines = old_vec_text.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith("Lĩnh vực:"):
                new_lines.append(f"Lĩnh vực: {new_cat}")
            else:
                new_lines.append(line)
        
        return pd.Series([new_cat, '\n'.join(new_lines)])

    tqdm.pandas(desc="Applying updates")
    df[['doc_category', 'vector_text']] = df.apply(apply_update, axis=1)

    # 4. Lưu Parquet
    df.to_parquet(input_file, index=False)
    print(f"✅ HOÀN TẤT! Đã cập nhật category xịn từ Gemini vào {input_file}")
    print("👉 Giờ bạn hãy chạy indexing.py")

if __name__ == "__main__":
    run_gemini_categorization()