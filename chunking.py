import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import re
import os
from pathlib import Path
from config import Config

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
    
    # --- B. KHỞI TẠO SPLITTER ---
    # Ưu tiên cắt theo đoạn văn (\n\n) trước, sau đó đến câu (. )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=[
        "\n\n",      # Ưu tiên 1: Ngắt đoạn
        "\n",        # Ưu tiên 2: Xuống dòng
        ". ",        # Ưu tiên 3: Hết câu (có dấu cách)
        ".",         # Ưu tiên 4: Hết câu (dính liền - trường hợp lỗi typo)
        "!", "?",    # Câu cảm thán/hỏi
        ";",         # Dấu chấm phẩy
        " ",         # Dấu cách (Fallback cuối cùng)
        ""           # Cắt ký tự (Bất đắc dĩ mới dùng)
    ],
        length_function=len,
        is_separator_regex=False
    )
    
    final_chunks = []
    
    # --- C. VÒNG LẶP XỬ LÝ ---
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_text = row.get('text', '')
        title = row.get('title', 'Không tiêu đề')
        url = row.get('url', '')
        categories = row.get('categories', [])
        
        # Convert categories list to string
        cat_str = str(categories) if categories else ""

        # 1. CLEANING
        clean_text = clean_wiki_text(original_text)
        
        # Lọc: Bài quá ngắn sau khi clean -> Bỏ
        if len(clean_text) < 50: continue

        # 2. CHUNKING
        chunks = splitter.create_documents([clean_text])
        
        for i, chunk in enumerate(chunks):
            content = re.sub(r'^[.,;\s]+', '', chunk.page_content).strip()
            
            # Lọc 1: Chunk quá ngắn
            if len(content) < 60: continue

            # 2. Lọc chunk "Treo"
            # Nếu kết thúc bằng dấu hai chấm, nghĩa là nó chưa nói hết câu -> BỎ
            if content.endswith(':'): continue

            # 3. Lọc chunk "Danh sách rác"
            # Nếu chunk chứa từ "Niên biểu", "Xem thêm", "Mục lục" và ngắn dưới 100 ký tự -> BỎ
            bad_keywords = ["Niên biểu", "Mục lục", "Danh sách", "Các vua", "Tiểu sử"]
            if len(content) < 100 and any(kw in content for kw in bad_keywords):
                if content.count('.') > 2: # Nếu có nhiều dấu chấm (do gộp dòng tiêu đề)
                    continue

            # 4. Lọc chunk không có dấu kết thúc câu nếu chunk ngắn (< 150 ký tự)
            if len(content) < 150 and content[-1] not in ['.', '!', '?', '"', "'", ')']:
                continue
            
            # Lọc 2: Chunk không có dấu câu kết thúc -> thường là list rác
            if not any(char in content for char in ['.', '?', '!', ';']):
                # Cho phép ngoại lệ nếu chunk rất dài -> có thể là đoạn văn thiếu dấu chấm
                if len(content) < 100: continue
            
            # Lọc 3: Chunk chứa quá nhiều từ khóa sách vở
            if content.count("ISBN") > 0 or content.count("Xuất bản") > 1:
                continue

            # --- E. CONTEXT INJECTION
            # Format: "Chủ đề: {title}\nNội dung: {content}"
            if content[-1] not in ['.', '!', '?', ';', '"', "'", ')']:
                content += "."
                
            vector_text = f"Chủ đề: {title}\nNội dung: {content}"
            
            final_chunks.append({
                "chunk_id": f"{idx}_{i}",
                "doc_title": title,
                "doc_url": url,
                "doc_category": cat_str,
                "vector_text": vector_text,    # Text dùng để Embed (context ịnected)
                "display_text": content,       # Text gốc
                "char_len": len(vector_text)
            })

    # --- F. LƯU KẾT QUẢ ---
    if not final_chunks:
        print("Cảnh báo: Không tạo ra được chunk nào!")
        return

    result_df = pd.DataFrame(final_chunks)
    
    print(f"\nXử lý hoàn tất!")
    print(f"   - Đầu vào: {len(df)} bài viết")
    print(f"   - Đầu ra : {len(result_df)} chunks sạch")
    print(f"   - Tỷ lệ  : {len(result_df)/len(df):.1f} chunks/bài")
    
    # Tạo thư mục output
    os.makedirs(Config.CHUNKING_OUTPUT_FILE.parent, exist_ok=True)
    
    result_df.to_parquet(Config.CHUNKING_OUTPUT_FILE, index=False, compression='snappy')
    print(f"File đã lưu tại: {Config.CHUNKING_OUTPUT_FILE}")
    
    # --- G. KIỂM TRA MẪU (SANITY CHECK) ---
    print("\n" + "="*60)
    print("KIỂM TRA 1 CHUNK NGẪU NHIÊN")
    print("="*60)
    if len(result_df) > 0:
        sample = result_df.sample(1).iloc[0]
        print(f"[Title]: {sample['doc_title']}")
        print(f"[Vector Text]:\n{sample['vector_text']}")
        print("-"*60)

if __name__ == "__main__":
    process_chunking()