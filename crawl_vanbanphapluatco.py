"""
PRODUCTION CRAWLER - VANBANPHAPLUAT.CO
Output Schema khớp với quy trình Chunking/RAG
"""
import requests
from bs4 import BeautifulSoup
import trafilatura
import pandas as pd
import time
import random
import hashlib
import os
import re
import sqlite3
import logging
from urllib.parse import urljoin
from tqdm import tqdm
from datetime import datetime
import urllib3
from config import Config

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CẤU HÌNH CHẠY THẬT (PRODUCTION) ---
DB_FILE = "vbpl_production.db"
OUTPUT_FILE = Config.OUTPUT_DIR / "vbpl_full_dataset.parquet"
TEMP_BATCH_DIR = "vbpl_batches_prod"

# Tăng kích thước batch để giảm số lượng file nhỏ (ghi đĩa mỗi 100 bài)
CHECKPOINT_SIZE = 100       

# Bộ lọc nội dung rác (Văn bản luật thường dài, <800 ký tự thường là lỗi hoặc mục lục)
MIN_CONTENT_LENGTH = 800  

# Logging
logging.basicConfig(
    filename="vbpl_production.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

os.makedirs(TEMP_BATCH_DIR, exist_ok=True)

# --- DANH MỤC CRAWL ĐẦY ĐỦ (FULL SCOPE) ---
SEED_CATEGORIES = {
    # === LOẠI VĂN BẢN ===
    "Loại văn bản": [
        "/loai-van-ban/luat", "/loai-van-ban/nghi-dinh", "/loai-van-ban/thong-tu",
        "/loai-van-ban/quyet-dinh", "/loai-van-ban/chi-thi", "/loai-van-ban/nghi-quyet",
        "/loai-van-ban/phap-lenh", "/loai-van-ban/lenh", "/loai-van-ban/thong-tu-lien-tich",
        "/loai-van-ban/cong-dien", "/loai-van-ban/cong-van", "/loai-van-ban/quy-che",
        "/loai-van-ban/quy-dinh", "/loai-van-ban/huong-dan", "/loai-van-ban/tieu-chuan-viet-nam",
        "/loai-van-ban/quy-chuan"
    ],
    
    # === LĨNH VỰC ===
    "Lĩnh vực": [
        "/linh-vuc/doanh-nghiep", "/linh-vuc/lao-dong-tien-luong", "/linh-vuc/thue-phi-le-phi",
        "/linh-vuc/bao-hiem", "/linh-vuc/giao-thong-van-tai", "/linh-vuc/xay-dung-do-thi",
        "/linh-vuc/tai-chinh-nha-nuoc", "/linh-vuc/nong-nghiep", "/linh-vuc/the-thao-y-te",
        "/linh-vuc/giao-duc", "/linh-vuc/van-hoa-xa-hoi", "/linh-vuc/tai-nguyen-moi-truong",
        "/linh-vuc/bat-dong-san", "/linh-vuc/thuong-mai", "/linh-vuc/dau-tu",
        "/linh-vuc/chung-khoan", "/linh-vuc/tien-te-ngan-hang", "/linh-vuc/so-huu-tri-tue",
        "/linh-vuc/cong-nghe-thong-tin", "/linh-vuc/quyen-dan-su", "/linh-vuc/trach-nhiem-hinh-su",
        "/linh-vuc/vi-pham-hanh-chinh", "/linh-vuc/thu-tuc-to-tung", "/linh-vuc/bo-may-hanh-chinh",
        "/linh-vuc/ke-toan-kiem-toan", "/linh-vuc/cong-nghiep", "/linh-vuc/dien-dien-tu",
        "/linh-vuc/hoa-chat", "/linh-vuc/xuat-nhap-khau"
    ],
    
    # === VĂN BẢN MỚI ===
    "Văn bản mới": ["/van-ban-moi"]
}

# --- DATABASE MANAGER ---
class HistoryDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS visited_urls (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    category TEXT,
                    status TEXT DEFAULT 'pending',
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        except Exception as e:
            logging.critical(f"❌ Cannot connect to DB: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()

    def exists(self, url):
        try:
            h = hashlib.md5(url.encode()).hexdigest()
            self.cursor.execute("SELECT 1 FROM visited_urls WHERE url_hash = ?", (h,))
            return self.cursor.fetchone() is not None
        except:
            return False

    def add(self, url, category, status='success'):
        h = hashlib.md5(url.encode()).hexdigest()
        try:
            self.cursor.execute(
                "INSERT OR IGNORE INTO visited_urls (url_hash, url, category, status) VALUES (?, ?, ?, ?)", 
                (h, url, category, status)
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"DB insert error: {e}")

# --- NETWORK & EXTRACTION ---
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'vi-VN,vi;q=0.9'
    })
    return session

def extract_legal_document(session, url):
    try:
        resp = session.get(url, timeout=15, verify=False) 
        if len(resp.text) < 1000: return None

        # Dùng trafilatura để lấy text sạch
        data = trafilatura.extract(
            resp.text,
            output_format="json",
            include_comments=False,
            include_tables=True, # Giữ bảng biểu vì luật hay có bảng
            favor_precision=True
        )
        
        if data:
            import json
            j = json.loads(data)
            text = j.get('text', '').strip()
            title = j.get('title', '').strip()
            
            if len(text) < MIN_CONTENT_LENGTH: return None
            
            return {
                "title": title,
                "text": text,
                "url": url
            }
    except Exception as e:
        logging.error(f"Extract error {url}: {e}")
    return None

def find_document_links(session, url):
    links = set()
    base_url = "https://vanbanphapluat.co"
    try:
        resp = session.get(url, timeout=15, verify=False)
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            
            # Logic lọc link bài viết (loại bỏ link danh mục/quảng cáo)
            if (base_url in full_url and 
                '/loai-van-ban/' not in full_url and
                '/linh-vuc/' not in full_url and
                '/van-ban-moi' not in full_url and
                len(full_url) > len(base_url) + 15):
                links.add(full_url)
    except Exception as e:
        logging.error(f"List page error {url}: {e}")
    return list(links)

def save_batch(batch_data, batch_id):
    if not batch_data: return
    try:
        df = pd.DataFrame(batch_data)
        batch_file = os.path.join(TEMP_BATCH_DIR, f"prod_batch_{batch_id:04d}.parquet")
        df.to_parquet(batch_file, index=False)
        logging.info(f"💾 Saved batch {batch_id}: {len(df)} docs")
    except Exception as e:
        logging.error(f"Save batch error: {e}")

# --- CORE LOGIC (INFINITE SCROLL) ---
def crawl_category_full(session, db, category_group, seed_urls, batch_buffer, batch_counter):
    total_collected = 0
    
    for seed_url in seed_urls:
        # Xác định tên category cụ thể từ URL để lưu vào cột doc_category
        # Vd: /linh-vuc/thue-phi -> "Thuế phí"
        specific_cat_name = "Văn bản pháp luật"
        if '/linh-vuc/' in seed_url:
            specific_cat_name = seed_url.split('/linh-vuc/')[-1].replace('-', ' ').title()
        elif '/loai-van-ban/' in seed_url:
            specific_cat_name = seed_url.split('/loai-van-ban/')[-1].replace('-', ' ').title()
        
        page = 1
        empty_pages_count = 0 
        
        # Vòng lặp vô tận, chỉ dừng khi không còn bài
        while True:
            # Dừng nếu 3 trang liên tiếp không tìm thấy bài mới
            if empty_pages_count >= 3:
                logging.info(f"Dừng quét {specific_cat_name} tại trang {page} (Hết dữ liệu)")
                break

            # URL phân trang
            current_url = seed_url if page == 1 else f"{seed_url}?page={page}"
            
            # Lấy links
            doc_links = find_document_links(session, current_url)
            
            if not doc_links:
                logging.warning(f"{specific_cat_name} - Trang {page}: Không có link nào.")
                empty_pages_count += 1
                page += 1
                continue

            # Duyệt bài
            new_in_page = 0
            # Dùng tqdm nhưng ẩn bớt để đỡ spam console khi chạy lâu
            for doc_url in doc_links:
                if db.exists(doc_url):
                    continue
                
                doc = extract_legal_document(session, doc_url)
                if doc:
                    # --- [QUAN TRỌNG] ĐỔI TÊN CỘT CHO KHỚP HEADER ---
                    batch_buffer.append({
                        "doc_title": doc['title'],       # Khớp với yêu cầu
                        "doc_category": specific_cat_name, # Khớp với yêu cầu
                        "doc_url": doc['url'],           # Khớp với yêu cầu
                        "doc_content": doc['text'],      # Dữ liệu gốc để chunking
                        "crawled_at": datetime.now().isoformat()
                    })
                    
                    db.add(doc_url, specific_cat_name, 'success')
                    new_in_page += 1
                    total_collected += 1
                    
                    if len(batch_buffer) >= CHECKPOINT_SIZE:
                        save_batch(batch_buffer, batch_counter[0])
                        batch_counter[0] += 1
                        batch_buffer.clear()
                
                # Delay ngẫu nhiên để không bị block IP
                time.sleep(random.uniform(0.5, 1.2))

            if new_in_page > 0:
                print(f"[{specific_cat_name}] Page {page}: +{new_in_page} bài mới.")
                empty_pages_count = 0 # Reset biến đếm nếu tìm thấy bài
            else:
                empty_pages_count += 1
                
            page += 1
            
    return total_collected

# --- MAIN RUN ---
def main():
    print("🚀 BẮT ĐẦU CRAWL PRODUCTION (FULL DỮ LIỆU)...")
    print(f"📦 Output sẽ lưu tại: {TEMP_BATCH_DIR}")
    print("⚠️  Lưu ý: Quá trình này có thể kéo dài nhiều giờ.")
    
    session = get_session()
    batch_buffer = []
    batch_counter = [0]
    
    with HistoryDB(DB_FILE) as db:
        for group_name, seeds in SEED_CATEGORIES.items():
            print(f"\n📂 Đang xử lý nhóm: {group_name.upper()}")
            # Tạo full URLs
            full_seeds = [urljoin("https://vanbanphapluat.co", s) for s in seeds]
            
            crawl_category_full(session, db, group_name, full_seeds, batch_buffer, batch_counter)
    
    # Save nốt batch cuối
    if batch_buffer:
        save_batch(batch_buffer, batch_counter[0])

    # Merge file cuối cùng
    print("\n📦 Đang gộp toàn bộ dữ liệu...")
    all_files = [os.path.join(TEMP_BATCH_DIR, f) for f in os.listdir(TEMP_BATCH_DIR) if f.endswith('.parquet')]
    
    if all_files:
        combined_df = pd.concat([pd.read_parquet(f) for f in tqdm(all_files, desc="Merging")], ignore_index=True)
        
        # Deduplicate lần cuối (tránh trùng lặp do 1 bài thuộc nhiều danh mục)
        combined_df.drop_duplicates(subset=['doc_url'], keep='first', inplace=True)
        
        combined_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✅ HOÀN TẤT! Tổng số văn bản: {len(combined_df):,}")
        print(f"📁 File kết quả: {OUTPUT_FILE}")
        
        # (Tuỳ chọn) Xóa file tạm để giải phóng ổ cứng
        for f in all_files: os.remove(f)
        os.rmdir(TEMP_BATCH_DIR)
    else:
        print("⚠️ Không có dữ liệu nào được thu thập.")

if __name__ == "__main__":
    main()