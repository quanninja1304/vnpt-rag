import pandas as pd
import re
import os
import glob
from datetime import datetime

# --- CẤU HÌNH ---
INPUT_DIR = "phap_luat_txt"
OUTPUT_FILE = "output/1_manual_law_strict.parquet"

FILENAME_MAP = {
    "hien_phap_2025.txt": "Hiến pháp nước CHXHCN Việt Nam 2013", # Nhớ sửa tên file nếu bạn đã đổi lại thành 2013
    "hien_phap_2013.txt": "Hiến pháp nước CHXHCN Việt Nam 2013",
    "luat_an_ninh_mang.txt": "Luật An ninh mạng 2018",
    "luat_dan_su.txt": "Bộ luật Dân sự 2015",
    "luat_dat_dai.txt": "Luật Đất đai 2024",
    "luat_giao_duc.txt": "Luật Giáo dục 2019",
    "luat_hinh_su.txt": "Bộ luật Hình sự 2015",
    "luat_giao_thong_duong_bo.txt": "Luật Giao thông đường bộ 2008",
    "luat_hon_nhan_va_gia_dinh.txt": "Luật Hôn nhân và Gia đình 2014",
    "luat_lao_dong.txt": "Bộ luật Lao động 2019"
}

def clean_line(line):
    """Xóa ký tự rác đầu/cuối dòng"""
    # Thay thế non-breaking space bằng space thường
    line = line.replace('\xa0', ' ').replace('\u200b', '')
    return line.strip()

def parse_strict(content, doc_title):
    lines = content.split('\n')
    chunks = []
    
    # State variables
    current_context = [] # ["CHƯƠNG I", "CHẾ ĐỘ CHÍNH TRỊ"]
    current_article_header = "" # "Điều 1."
    current_body = [] # ["Nước CHXHCN VN...", "là nước độc lập..."]
    
    # Regex neo chặt đầu dòng (^): Chỉ bắt khi "Điều" đứng đầu
    re_article_start = re.compile(r'^Điều\s+\d+', re.IGNORECASE)
    re_context_start = re.compile(r'^(CHƯƠNG|MỤC|PHẦN)\s+', re.IGNORECASE)

    for line in lines:
        line = clean_line(line)
        if not line: continue
        
        # Bỏ qua rác
        if any(x in line for x in ["Tải về", "Mục lục", "Về đầu trang"]): continue

        # --- CASE 1: BẮT GẶP ĐIỀU LUẬT MỚI ---
        if re_article_start.match(line):
            # 1. Lưu Điều luật CŨ (nếu đang gom dở)
            if current_article_header:
                context_str = " - ".join(current_context)
                full_text = f"Văn bản: {doc_title}\n{context_str}\n{current_article_header}\n" + "\n".join(current_body)
                chunks.append(full_text.strip())
            
            # 2. Reset để bắt đầu Điều luật MỚI
            current_article_header = line
            current_body = []
            
        # --- CASE 2: BẮT GẶP NGỮ CẢNH (CHƯƠNG/MỤC) ---
        elif re_context_start.match(line) or (line.isupper() and len(line) < 100 and "ĐIỀU" not in line and "CỘNG HÒA" not in line):
            # Nếu gặp Chương mới -> Cũng phải lưu Điều luật cũ lại (vì hết chương rồi)
            if current_article_header:
                context_str = " - ".join(current_context)
                full_text = f"Văn bản: {doc_title}\n{context_str}\n{current_article_header}\n" + "\n".join(current_body)
                chunks.append(full_text.strip())
                current_article_header = ""
                current_body = []

            # Cập nhật Context
            if re_context_start.match(line):
                # Gặp "CHƯƠNG..." -> Reset context cũ
                current_context = [line]
            else:
                # Gặp tiêu đề viết hoa "CHẾ ĐỘ CHÍNH TRỊ" -> Nối thêm vào
                if line not in current_context:
                    current_context.append(line)

        # --- CASE 3: NỘI DUNG ---
        else:
            if current_article_header:
                current_body.append(line)
            else:
                # Nội dung chưa thuộc điều nào (Lời nói đầu, Căn cứ pháp lý...)
                pass

    # --- LƯU CHUNK CUỐI CÙNG ---
    if current_article_header:
        context_str = " - ".join(current_context)
        full_text = f"Văn bản: {doc_title}\n{context_str}\n{current_article_header}\n" + "\n".join(current_body)
        chunks.append(full_text.strip())
        
    return chunks

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Thư mục '{INPUT_DIR}' không tồn tại.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_data = []
    txt_files = glob.glob(f"{INPUT_DIR}/*.txt")

    print(f"📂 Tìm thấy {len(txt_files)} file.")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        doc_title = FILENAME_MAP.get(filename, filename.replace(".txt", "").title())
        
        print(f"\n🔨 Processing: {filename} -> {doc_title}")
        
        try:
            # Dùng utf-8-sig để tránh ký tự BOM (\ufeff) đầu file
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            chunks = parse_strict(content, doc_title)
            
            if not chunks:
                print(f"   ⚠️ 0 CHUNKS! Kiểm tra lại xem file có chữ 'Điều' ở đầu dòng không.")
            else:
                # In ra 3 tiêu đề đầu tiên bắt được để kiểm tra
                titles = [c.split('\n')[2] for c in chunks[:3] if len(c.split('\n')) > 2]
                print(f"   ✅ OK: {len(chunks)} điều luật.")
                print(f"   👀 Sample: {titles}...")

                for i, chunk_text in enumerate(chunks):
                    # Bỏ qua chunk quá ngắn
                    if len(chunk_text) < 30: continue
                    
                    all_data.append({
                        # [QUAN TRỌNG] ID duy nhất để không bị trùng đè trong Qdrant
                        # Kết hợp tên file và số thứ tự chunk
                        "chunk_id": f"law_{filename}_{i}",
                        
                        # [QUAN TRỌNG] Các trường khớp với indexing.py
                        "doc_title": doc_title,
                        "doc_category": "Pháp luật",        # Để string, không để list
                        "doc_url": f"local/{filename}",
                        
                        # Text dùng để Embed (Gửi lên API)
                        "vector_text": chunk_text,
                        
                        # Text lưu vào Payload (Để LLM đọc sau này)
                        "display_text": chunk_text
                    })
                
        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    if all_data:
        df = pd.DataFrame(all_data)
        df['crawled_at'] = datetime.now().isoformat()
        
        print("\n📊 THỐNG KÊ:")
        print(df['doc_title'].value_counts())
        
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"\n💾 Đã lưu: {OUTPUT_FILE}")
    else:
        print("\n❌ Không có dữ liệu.")

if __name__ == "__main__":
    main()