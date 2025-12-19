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
from concurrent.futures import ThreadPoolExecutor, as_completed


# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CẤU HÌNH CHẠY THẬT (PRODUCTION) ---
DB_FILE = str(Config.BASE_DIR / "database" / "vbpl_production.db")
OUTPUT_FILE = Config.CRAWL_OUTPUT_VANBANPHAPLUATCO
TEMP_BATCH_DIR = str(Config.BASE_DIR / "vbpl_batches_prod")

# Tăng kích thước batch để giảm số lượng file nhỏ (ghi đĩa mỗi 100 bài)
CHECKPOINT_SIZE = 100       

# Bộ lọc nội dung rác (Văn bản luật thường dài, <800 ký tự thường là lỗi hoặc mục lục)
MIN_CONTENT_LENGTH = 800  

MAX_WORKERS = 10  # Số luồng chạy song song (Tăng tốc độ)

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
        # "/loai-van-ban/luat", "/loai-van-ban/nghi-dinh", "/loai-van-ban/thong-tu",
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

        # --- BƯỚC 1: DÙNG BEAUTIFULSOUP ĐỂ LỌC RÁC TRƯỚC ---
        soup = BeautifulSoup(resp.content, 'html.parser')

        # 1. Xóa các thẻ HTML rác chắc chắn không phải nội dung
        for tag in soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'iframe', 'form']):
            tag.decompose()

        # 2. Xóa các khối DIV thường là Sidebar/Quảng cáo/Tin liên quan
        # Tìm các div có class hoặc id chứa từ khóa nhạy cảm
        noise_keywords = re.compile(r"sidebar|widget|comment|relate|news|tin-moi|quang-cao|right-col", re.IGNORECASE)
        for div in soup.find_all("div", attrs={"class": noise_keywords}):
            div.decompose()
        for div in soup.find_all("div", attrs={"id": noise_keywords}):
            div.decompose()

        # 3. (Mẹo) Nếu trang web dùng Bootstrap, nội dung chính thường ở col-md-8 hoặc col-md-9
        # Sidebar thường ở col-md-3 hoặc col-md-4. Ta xóa cột nhỏ đi.
        for div in soup.find_all("div", class_=re.compile(r"col-md-[34]")):
             div.decompose()

        # --- BƯỚC 2: ĐƯA HTML ĐÃ SẠCH VÀO TRAFILATURA ---
        # Chuyển soup ngược lại thành string
        clean_html = str(soup)

        data = trafilatura.extract(
            clean_html,
            output_format="json",
            include_comments=False,
            include_tables=True, 
            favor_precision=True
        )
        
        if data:
            import json
            j = json.loads(data)
            text = j.get('text', '').strip()
            title = j.get('title', '').strip()
            
            # Nếu title bị rỗng do trafilatura, lấy fallback từ thẻ h1 của soup gốc
            if not title and soup.find('h1'):
                title = soup.find('h1').get_text().strip()

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
    MAX_PAGES_PER_CATEGORY = 70
    total_collected = 0
    for seed_url in seed_urls:
        specific_cat_name = "Văn bản pháp luật"
        if '/linh-vuc/' in seed_url: specific_cat_name = seed_url.split('/linh-vuc/')[-1].replace('-', ' ').title()
        elif '/loai-van-ban/' in seed_url: specific_cat_name = seed_url.split('/loai-van-ban/')[-1].replace('-', ' ').title()
        
        page = 1
        empty_pages_count = 0 
        
        while True:
            if page > MAX_PAGES_PER_CATEGORY:
                print(f"🛑 Đã đạt giới hạn {MAX_PAGES_PER_CATEGORY} trang cho mục '{specific_cat_name}'. Chuyển mục tiếp theo >>")
                break

            if empty_pages_count >= 3:
                logging.info(f"Dừng quét {specific_cat_name} tại trang {page}")
                break

            current_url = seed_url if page == 1 else f"{seed_url}?page={page}"
            doc_links = find_document_links(session, current_url)
            
            if not doc_links:
                empty_pages_count += 1
                page += 1
                continue

            # --- ĐA LUỒNG (MULTI-THREADING) ---
            new_in_page = 0
            links_to_crawl = [u for u in doc_links if not db.exists(u)]
            
            if links_to_crawl:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_url = {executor.submit(extract_legal_document, session, url): url for url in links_to_crawl}
                    
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            doc = future.result()
                            if doc:
                                batch_buffer.append({
                                    "doc_title": doc['title'],
                                    "doc_category": specific_cat_name,
                                    "doc_url": doc['url'],
                                    "doc_content": doc['text'],
                                    "crawled_at": datetime.now().isoformat()
                                })
                                db.add(url, specific_cat_name, 'success')
                                new_in_page += 1
                                total_collected += 1
                                
                                if len(batch_buffer) >= CHECKPOINT_SIZE:
                                    save_batch(batch_buffer, batch_counter[0])
                                    batch_counter[0] += 1
                                    batch_buffer.clear()
                        except Exception:
                            pass
            
            if new_in_page > 0:
                print(f"[{specific_cat_name}] Page {page}: +{new_in_page} bài mới.")
                empty_pages_count = 0
            else:
                # Nếu trang có link nhưng toàn link cũ -> Vẫn tính là empty để sớm nhảy sang danh mục khác
                empty_pages_count += 1
                print(f"[{specific_cat_name}] Page {page}: Không có bài mới.")
                
            page += 1
    return total_collected

# --- MAIN RUN ---
def main():
    print("🚀 BẮT ĐẦU CRAWL RESUME (ĐA LUỒNG)...")
    session = get_session()
    batch_buffer = []
    batch_counter = [300] # Bắt đầu từ số lớn để không trùng file cũ
    
    with HistoryDB(DB_FILE) as db:
        for group_name, seeds in SEED_CATEGORIES.items():
            print(f"\n📂 Đang xử lý nhóm: {group_name.upper()}")
            full_seeds = [urljoin("https://vanbanphapluat.co", s) for s in seeds]
            crawl_category_full(session, db, group_name, full_seeds, batch_buffer, batch_counter)
    
    if batch_buffer: save_batch(batch_buffer, batch_counter[0])
    print("\n📦 Đang gộp toàn bộ dữ liệu...")
    all_files = [os.path.join(TEMP_BATCH_DIR, f) for f in os.listdir(TEMP_BATCH_DIR) if f.endswith('.parquet')]
    
    if all_files:
        # Đọc tất cả file nhỏ
        combined_df = pd.concat([pd.read_parquet(f) for f in tqdm(all_files, desc="Merging")], ignore_index=True)
        
        # Xóa các dòng trùng lặp (Deduplicate)
        combined_df.drop_duplicates(subset=['doc_url'], keep='first', inplace=True)
        
        # Lưu ra file cuối cùng
        combined_df.to_parquet(OUTPUT_FILE, index=False)
        
        print(f"✅ HOÀN TẤT! Tổng số văn bản: {len(combined_df):,}")
        print(f"📁 File kết quả cuối cùng: {OUTPUT_FILE}")
    else:
        print("⚠️ Không tìm thấy dữ liệu để gộp.")
    print("✅ HOÀN TẤT!")

if _name_ == "_main_":
    main()