import requests
from bs4 import BeautifulSoup
import trafilatura
import pandas as pd
import time
import random
import re
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

# --- CẤU HÌNH ---
OUTPUT_FILE = "output/vietjack_history_geo.parquet"

# Danh sách trang Mục lục (Index) các môn
# Lưu ý: Lớp 10, 11 chương trình mới thường chia 3 bộ sách (KNTT, CD, CTST). 
# Ở đây tôi chọn bộ "Kết Nối Tri Thức" (phổ biến nhất) làm mẫu. Bạn có thể thêm link bộ khác nếu cần.
SUBJECT_INDEXES = [
    # --- LỚP 12 (Chương trình cũ - Vẫn thi THPTQG 2025 theo form này nhiều) ---
    {"url": "https://vietjack.com/lich-su-12/index.jsp", "category": "Lịch sử 12", "match": "lich-su-12"},
    {"url": "https://vietjack.com/dia-li-12/index.jsp", "category": "Địa lý 12", "match": "dia-li-12"},
    
    # --- LỚP 11 (Kết nối tri thức) ---
    {"url": "https://vietjack.com/lich-su-11-kn/index.jsp", "category": "Lịch sử 11", "match": "lich-su-11"},
    {"url": "https://vietjack.com/dia-li-11-kn/index.jsp", "category": "Địa lý 11", "match": "dia-li-11"},

    # --- LỚP 10 (Kết nối tri thức) ---
    {"url": "https://vietjack.com/lich-su-10-kn/index.jsp", "category": "Lịch sử 10", "match": "lich-su-10"},
    {"url": "https://vietjack.com/dia-li-10-kn/index.jsp", "category": "Địa lý 10", "match": "dia-li-10"},
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_lesson_links(index_info):
    """Bước 1: Lấy danh sách link bài học từ trang mục lục"""
    url = index_info['url']
    category = index_info['category']
    match_pattern = index_info['match']
    
    print(f"🔍 Scanning Index: {category} ({url})...")
    links = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm tất cả thẻ a
        # Vietjack thường để bài học trong thẻ <a> có href chứa tên môn
        all_links = soup.find_all('a', href=True)
        
        seen_links = set()
        
        for a in all_links:
            href = a['href']
            # 1. Lọc link chứa pattern môn (ví dụ 'lich-su-12')
            # 2. Lọc bỏ link bài tập/trắc nghiệm (chỉ lấy bài học/lý thuyết/giải bài tập sgk có nội dung)
            # 3. Tránh link trùng
            if match_pattern in href and href not in seen_links:
                # Loại bỏ các link không phải bài học chính (ví dụ link về tác giả, quảng cáo)
                if any(x in href for x in ['facebook', 'youtube', '#']): continue
                
                full_url = urljoin("https://vietjack.com/", href)
                title = a.get_text().strip()
                
                # Ưu tiên các link có tiêu đề bắt đầu bằng "Bài", "Chương", "Lý thuyết"
                if len(title) > 5: 
                    links.append({"url": full_url, "title": title, "category": category})
                    seen_links.add(href)
                    
        print(f"   -> Found {len(links)} lessons for {category}.")
        return links
        
    except Exception as e:
        print(f"❌ Error scanning index {url}: {e}")
        return []

def clean_text(text):
    if not text: return ""
    # Xóa rác Vietjack
    garbage = ["Quảng cáo", "Xem thêm", "Tải về", "Mục lục", "Bản in", "Trang chủ", "VietJack", "Bình luận", "Theo dõi chúng tôi"]
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_lines = [line for line in lines if not any(g.lower() in line.lower() for g in garbage)]
    # Bỏ dòng quá ngắn (thường là menu)
    clean_lines = [line for line in clean_lines if len(line) > 5]
    return "\n".join(clean_lines)

def scrape_content(target):
    """Bước 2: Cào nội dung chi tiết"""
    url = target['url']
    title = target['title']
    category = target['category']
    
    # print(f"🕷️ Crawling: {title[:30]}...") # Uncomment nếu muốn log chi tiết
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        # 1. Trafilatura Extract (Nhanh & Sạch)
        content = ""
        try:
            content = trafilatura.extract(response.text, include_comments=False, include_tables=True)
        except: pass
        
        # 2. Fallback BS4 (Nếu Trafilatura fail)
        if not content or len(content) < 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Vietjack content hay nằm trong div class 'content' hoặc 'middle-col'
            main_div = soup.find('div', class_='content') or soup.find('div', class_='middle-col') or soup.body
            if main_div:
                content = main_div.get_text(separator='\n')
        
        content = clean_text(content)
        
        if len(content) < 100: # Nội dung quá ngắn -> Bỏ qua (có thể là trang lỗi)
            return None

        # Format chuẩn RAG
        # Inject Title và Category vào đầu text
        full_vector_text = f"Sách giáo khoa: {category}. Bài: {title}.\n{content}"
        
        return {
            "title": title,
            "url": url,
            "text": full_vector_text, # Dùng text này để embed
            "display_text": content, # Dùng text này để hiển thị (nếu cần tách)
            "categories": [category],
            "doc_type": "textbook_lesson"
        }

    except Exception as e:
        # print(f"❌ Error {url}: {e}")
        return None

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # BƯỚC 1: LẤY DANH SÁCH LINK
    all_targets = []
    print("🚀 Bắt đầu quét Mục lục...")
    for index_info in SUBJECT_INDEXES:
        links = get_lesson_links(index_info)
        all_targets.extend(links)
    
    print(f"\n🔥 Tổng cộng tìm thấy {len(all_targets)} bài học. Bắt đầu cào nội dung...")
    
    # BƯỚC 2: CÀO NỘI DUNG (ĐA LUỒNG)
    final_data = []
    # Dùng 10 workers cho nhanh
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scrape_content, target) for target in all_targets]
        
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                final_data.append(result)
            
            # Log tiến độ mỗi 50 bài
            if (i + 1) % 50 == 0:
                print(f"   ✅ Progress: {i + 1}/{len(all_targets)}...")

    print(f"\n✅ Hoàn tất! Thu thập được {len(final_data)} bài học.")
    
    if final_data:
        df = pd.DataFrame(final_data)
        df['crawled_at'] = datetime.now().isoformat()
        
        # Mapping cột cho chuẩn pipeline cũ
        # Nếu pipeline cũ của bạn dùng cột 'text' để embed thì code này đã chuẩn.
        # Nếu pipeline cũ dùng 'vector_text', hãy rename:
        # df = df.rename(columns={'text': 'vector_text'}) 
        
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"💾 Đã lưu file Parquet tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()