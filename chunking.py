import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import re
import os
import json
from pathlib import Path
from config import Config

# ===========================
# 1. STATE MANAGEMENT (NEW)
# ===========================
def load_processed_state():
    """Đọc danh sách các bài đã chunk trước đó"""
    if Config.CHUNKING_STATE_FILE.exists():
        with open(Config.CHUNKING_STATE_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_state(processed_titles):
    """Lưu lại danh sách các bài đã chunk"""
    with open(Config.CHUNKING_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_titles), f, ensure_ascii=False)

# ===========================
# 2. CLEAN TEXT (CORE LOGIC)
# ===========================
def clean_wiki_text(text: str) -> str:
    """
    Làm sạch văn bản Wikipedia (Fix triệt để lỗi chunk cuối bị dính footer)
    """
    if not isinstance(text, str) or not text: return ""
    
    # --- 1. CẮT BỎ FOOTER (Logic dòng đơn) ---
    # Thay vì tìm regex phức tạp, ta duyệt từng dòng.
    # Nếu gặp dòng nào ngắn (< 50 ký tự) mà chứa từ khóa dừng -> CẮT HẾT từ đó về sau.
    
    stop_phrases = [
        'tham khảo', 'thao khảo', 'liên kết ngoài', 'chú thích', 'xem thêm',
        'tài liệu tham khảo', 'đọc thêm', 'nguồn', 'ghi chú'
    ]
    
    lines = text.split('\n')
    cut_index = len(lines)
    
    for i, line in enumerate(lines):
        # Chuẩn hóa dòng để kiểm tra
        line_clean = line.strip().lower()
        
        # Bỏ decorators
        line_clean = re.sub(r'[=:\-\.]', '', line_clean).strip()
        
        # Nếu dòng ngắn (là tiêu đề) và khớp từ khóa dừng
        if len(line_clean) < 40 and line_clean in stop_phrases:
            cut_index = i
            break
            
    # Cắt bỏ phần rác
    text = '\n'.join(lines[:cut_index])

    # --- 2. XÓA RÁC ARTIFACTS ---
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[a-zà-ỹ\s]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[.*?\]\]', '', text)
    
    # --- 3. GỘP DÒNG TIÊU ĐỀ (Fix lỗi cụt lủn) ---
    # Biến các dòng tiêu đề cô lập thành câu để dính vào đoạn sau
    text = text.replace('\r\n', '\n')
    # Regex: Tìm dấu xuống dòng đơn (\n) không đi kèm \n khác
    text = re.sub(r'(?<!\n)\n(?!\n)', '. ', text)
    text = re.sub(r'\.\.', '.', text) # Sửa lỗi 2 dấu chấm
    text = re.sub(r'\. \.', '.', text)
    text = re.sub(r' +', ' ', text) # Xóa khoảng trắng thừa
    
    return text.strip()

# ===========================
# 3. CHUNKING PROCESS
# ===========================
def process_chunking():
    # --- A. LOAD DỮ LIỆU ---
    print(f"File input: {Config.CHUNKING_INPUT_FILE}")
    if not Config.CHUNKING_INPUT_FILE.exists():
        print(f"Lỗi: Không tìm thấy file {Config.CHUNKING_INPUT_FILE}")
        return

    try:
        if Config.CHUNKING_INPUT_FILE.suffix == '.parquet':
            df = pd.read_parquet(Config.CHUNKING_INPUT_FILE)
        else:
            df = pd.read_csv(Config.CHUNKING_INPUT_FILE)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return
        
    print(f"Số lượng bài viết gốc: {len(df)}")
    
    # --- LOGIC INCREMENTAL: LỌC BÀI MỚI ---
    processed_titles = load_processed_state()
    print(f"📦 Tổng bài viết trong kho: {len(df)}")
    print(f"🔄 Đã xử lý trước đó: {len(processed_titles)}")
    
    # Lọc ra các bài chưa có trong state
    df_new = df[~df['title'].isin(processed_titles)]
    
    if len(df_new) == 0:
        print("✅ Không có bài viết mới. Pipeline nghỉ ngơi!")
        # Xóa file delta cũ để tránh Indexing nạp lại thừa
        if Config.LATEST_CHUNKS_FILE.exists():
            os.remove(Config.LATEST_CHUNKS_FILE)
        return

    print(f"⚡ Phát hiện {len(df_new)} bài viết mới. Bắt đầu chunking...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", "!", "?", ";", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    new_chunks = []
    
    # --- VÒNG LẶP XỬ LÝ (Chỉ chạy trên df_new) ---
    for idx, row in tqdm(df_new.iterrows(), total=len(df_new)):
        original_text = row.get('text', '')
        title = row.get('title', 'Không tiêu đề')
        url = row.get('url', '')
        categories = row.get('categories', [])
        cat_str = str(categories) if categories else ""

        # Cleaning
        clean_text = clean_wiki_text(original_text)
        if len(clean_text) < 50: 
            processed_titles.add(title) # Đánh dấu đã xử lý (dù là rác)
            continue

        # Chunking
        chunks = splitter.create_documents([clean_text])
        
        for i, chunk in enumerate(chunks):
            content = re.sub(r'^[.,;\s]+', '', chunk.page_content).strip()
            
            # --- FILTERS ---
            if len(content) < 60: continue
            if content.endswith(':'): continue
            
            bad_keywords = ["Niên biểu", "Mục lục", "Danh sách", "Các vua", "Tiểu sử"]
            if len(content) < 100 and any(kw in content for kw in bad_keywords):
                if content.count('.') > 2: continue
            
            if len(content) < 150 and content[-1] not in ['.', '!', '?', '"', "'", ')']:
                continue
            if not any(char in content for char in ['.', '?', '!', ';']):
                if len(content) < 100: continue
            if content.count("ISBN") > 0 or content.count("Xuất bản") > 1:
                continue

            # Context Injection
            if content[-1] not in ['.', '!', '?', ';', '"', "'", ')']:
                content += "."
            
            vector_text = f"Chủ đề: {title}\nNội dung: {content}"
            
            new_chunks.append({
                "chunk_id": f"{idx}_{i}", # Lưu ý: idx này là của df_new
                "doc_title": title,
                "doc_url": url,
                "doc_category": cat_str,
                "vector_text": vector_text,
                "display_text": content,
                "char_len": len(vector_text)
            })
            
        # Đánh dấu bài này đã xong
        processed_titles.add(title)

    # --- LƯU KẾT QUẢ ---
    if not new_chunks:
        print("⚠️ Các bài mới không tạo được chunk nào.")
        save_processed_state(processed_titles) # Vẫn lưu state để lần sau không check lại
        return

    df_delta = pd.DataFrame(new_chunks)
    
    # 1. Lưu file DELTA (Chỉ chứa cái mới để Indexing dùng)
    Config.setup_dirs()
    df_delta.to_parquet(Config.LATEST_CHUNKS_FILE, index=False, compression='snappy')
    print(f"💾 [Delta] Đã lưu {len(df_delta)} chunks mới vào: {Config.LATEST_CHUNKS_FILE}")
    
    # 2. Append vào file MASTER (Để backup toàn bộ)
    if Config.MASTER_CHUNKS_FILE.exists():
        try:
            df_master = pd.read_parquet(Config.MASTER_CHUNKS_FILE)
            df_combined = pd.concat([df_master, df_delta], ignore_index=True)
            df_combined.to_parquet(Config.MASTER_CHUNKS_FILE, index=False, compression='snappy')
        except:
            df_delta.to_parquet(Config.MASTER_CHUNKS_FILE, index=False)
    else:
        df_delta.to_parquet(Config.MASTER_CHUNKS_FILE, index=False, compression='snappy')
    print(f"💾 [Master] Đã cập nhật file tổng: {Config.MASTER_CHUNKS_FILE}")

    # 3. Lưu trạng thái
    save_processed_state(processed_titles)
    print("✅ Đã cập nhật trạng thái xử lý.")

if __name__ == "__main__":
    process_chunking()