import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import trafilatura
import pandas as pd
import time
import random
import re
import sqlite3
import logging
import hashlib
import os
import shutil
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import urllib3
from datetime import datetime

# Tắt warning SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CẤU HÌNH ---
DB_FILE = "crawler_history.db"
OUTPUT_FILE = "knowledge_base_v5.parquet"
TEMP_BATCH_DIR = "batches_temp"
MAX_EMPTY_PAGES = 5
MAX_PAGES_SAFE_LIMIT = 500
CHECKPOINT_SIZE = 20

# Logging
logging.basicConfig(
    filename="crawler_v5.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger().addHandler(console)

# --- CHIẾN DỊCH ---
CAMPAIGNS = [
    {
        "name": "Nguoi_Ke_Su",
        "base_url": "https://nguoikesu.com",
        "seed_template": "https://nguoikesu.com/nhan-vat?start={}",
        "step": 5,
        "start_idx": 0,
        "allow_regex": r"nguoikesu\.com/(nhan-vat|dong-lich-su|di-tich-lich-su)/[\w-]{5,}$", 
        "deny_regex": r"(print|format|download|#|\.jpg|\.png|\.pdf|feed)"
    },
    {
        "name": "Tuyen_Giao",
        "base_url": "https://tuyengiao.vn",
        "seed_template": "https://tuyengiao.vn/nghien-cuu?page={}",
        "step": 1,
        "start_idx": 1,
        "allow_regex": r"tuyengiao\.vn/[\w-]{3,}/[\w-]{3,}$", 
        "deny_regex": r"(tag/|category/|author/|tim-kiem|login|dang-nhap|rss|feed)"
    },
    {
        "name": "Van_Ban_Chinh_Phu",
        "base_url": "https://vanban.chinhphu.vn",
        "seed_template": "https://vanban.chinhphu.vn/?pageid=27160&page={}",
        "step": 1,
        "start_idx": 1,
        "allow_regex": r"(chi-tiet-van-ban|ItemID=)", 
        "deny_regex": r"(javascript|mailto|print|download)"
    }
]

# --- DATABASE ---
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
                    source TEXT,
                    status TEXT DEFAULT 'pending',
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON visited_urls(source)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON visited_urls(status)')
            self.conn.commit()
            logging.info(f"✅ Database initialized: {db_path}")
        except Exception as e:
            logging.critical(f"❌ Cannot connect to DB: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Always close connection"""
        self.close()
        if exc_type is not None:
            logging.error(f"Database context exited with error: {exc_val}")
        return False  # Don't suppress exceptions

    def exists(self, url):
        try:
            h = hashlib.md5(url.encode()).hexdigest()
            self.cursor.execute("SELECT 1 FROM visited_urls WHERE url_hash = ?", (h,))
            return self.cursor.fetchone() is not None
        except Exception as e:
            logging.error(f"DB query error: {e}")
            return False

    def add(self, url, source, status='success'):
        h = hashlib.md5(url.encode()).hexdigest()
        try:
            self.cursor.execute(
                "INSERT OR IGNORE INTO visited_urls (url_hash, url, source, status) VALUES (?, ?, ?, ?)", 
                (h, url, source, status)
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"DB insert error: {e}")

    def get_stats(self):
        try:
            self.cursor.execute("""
                SELECT source, 
                       COUNT(*) as total,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success,
                       SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
                FROM visited_urls 
                GROUP BY source
            """)
            return self.cursor.fetchall()
        except Exception as e:
            logging.error(f"DB stats error: {e}")
            return []
    
    def get_total_stats(self):
        """Tổng hợp toàn bộ"""
        try:
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
                FROM visited_urls
            """)
            return self.cursor.fetchone()
        except:
            return (0, 0, 0)

    def close(self):
        """Safely close database connection"""
        try:
            if self.conn:
                self.conn.commit()  # Commit any pending transactions
                self.conn.close()
                logging.info("🔒 Database connection closed")
        except Exception as e:
            logging.error(f"Error closing database: {e}")
        finally:
            self.conn = None
            self.cursor = None

# --- NETWORK ---
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3, 
        read=3, 
        connect=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9'
    })
    return session

# --- CRAWL CONTENT ---
def crawl_content(session, url, category):
    try:
        resp = session.get(url, timeout=15, verify=False)
        
        # Quick validation
        if len(resp.text) < 500: return None
        if "window.location" in resp.text[:1000]: return None
        if any(x in resp.text.lower()[:500] for x in ['404', 'not found', 'captcha']): return None

        # Extract
        data = trafilatura.extract(
            resp.text, 
            output_format="json", 
            include_comments=False, 
            include_tables=True,
            no_fallback=False
        )
        
        if data:
            import json
            j = json.loads(data)
            title = j.get('title', '').strip() or "Untitled"
            text = j.get('text', '').strip()
            
            # Content quality check
            if len(text) < 300: return None
            if text.count('\n') < 3: return None
            
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
            return {
                "chunk_id": f"{category}_{doc_id}",
                "doc_title": title,
                "doc_url": url,
                "doc_category": category,
                "vector_text": f"Nguồn: {category}\nTiêu đề: {title}\n\n{text}",
                "display_text": f"**{title}**\n*(Nguồn: {url})*\n\n{text}",
                "crawl_date": datetime.now().isoformat(),
                "content_length": len(text)
            }
            
    except Exception as e:
        logging.error(f"Crawl error {url}: {str(e)}")
    
    return None

# --- BATCH MANAGEMENT ---
def save_batch_incremental(batch_data, batch_id):
    if not batch_data: return
    try:
        df = pd.DataFrame(batch_data)
        batch_file = os.path.join(TEMP_BATCH_DIR, f"batch_{batch_id:04d}.parquet")
        df.to_parquet(batch_file, index=False)
        logging.info(f"💾 Batch {batch_id} saved: {len(df)} docs")
    except Exception as e:
        logging.error(f"❌ Failed to save batch {batch_id}: {e}")

def merge_all_batches():
    batch_files = sorted([f for f in os.listdir(TEMP_BATCH_DIR) if f.endswith('.parquet')])
    if not batch_files:
        logging.warning("⚠️ No batch files to merge")
        return 0
    
    try:
        print(f"   📦 Merging {len(batch_files)} batch files...")
        dfs = []
        
        for bf in tqdm(batch_files, desc="Loading batches", leave=False):
            df = pd.read_parquet(os.path.join(TEMP_BATCH_DIR, bf))
            dfs.append(df)
        
        if not dfs: return 0

        combined = pd.concat(dfs, ignore_index=True)
        combined.drop_duplicates(subset=['doc_url'], keep='last', inplace=True)
        
        # Merge với file cũ
        if os.path.exists(OUTPUT_FILE):
            try:
                old_df = pd.read_parquet(OUTPUT_FILE)
                print(f"   🔄 Merging with existing KB ({len(old_df)} docs)...")
                combined = pd.concat([old_df, combined], ignore_index=True)
                combined.drop_duplicates(subset=['doc_url'], keep='last', inplace=True)
            except Exception as e:
                logging.error(f"⚠️ Could not merge with old file: {e}")
        
        combined.to_parquet(OUTPUT_FILE, index=False)
        logging.info(f"✅ Final KB: {len(combined)} total docs")
        
        # Cleanup
        for bf in batch_files:
            os.remove(os.path.join(TEMP_BATCH_DIR, bf))
        
        return len(combined)
        
    except Exception as e:
        logging.error(f"❌ Merge failed: {e}")
        return 0

# --- PRETTY PRINT HELPERS ---
def format_title(title, max_len=35):
    """Format title cho progress bar"""
    if len(title) > max_len:
        return title[:max_len-3] + "..."
    return title

def format_time(seconds):
    """Format thời gian đẹp"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# --- MAIN ---
def main():
    # Clean temp directory
    if os.path.exists(TEMP_BATCH_DIR):
        try:
            shutil.rmtree(TEMP_BATCH_DIR)
        except OSError as e:
            logging.warning(f"⚠️ Cannot clean temp dir: {e}")
    os.makedirs(TEMP_BATCH_DIR, exist_ok=True)

    start_time = datetime.now()
    session = get_session()
    batch_buffer = []
    batch_counter = 0
    total_new_docs = 0
    
    print("🚀 BẮT ĐẦU CHIẾN DỊCH CRAWL (V5 PRODUCTION)")
    print(f"⏰ Thời gian bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Use context manager for database
    with HistoryDB(DB_FILE) as db:
        try:
            for camp_idx, camp in enumerate(CAMPAIGNS, 1):
                print(f"\n{'='*70}")
                print(f"📂 Nguồn {camp_idx}/{len(CAMPAIGNS)}: {camp['name']}")
                print(f"{'='*70}")
                
                consecutive_empty = 0
                page = 0
                campaign_success = 0
                campaign_failed = 0
                
                pbar = tqdm(
                    range(MAX_PAGES_SAFE_LIMIT), 
                    desc=f"Scanning {camp['name'][:20]}",
                    leave=True
                )
                
                for _ in pbar:
                    idx = camp['start_idx'] + (page * camp['step'])
                    list_url = camp['seed_template'].format(idx)
                    
                    # Crawl listing page
                    new_links = []
                    try:
                        r = session.get(list_url, verify=False, timeout=15)
                        soup = BeautifulSoup(r.content, 'html.parser')
                        
                        for a in soup.find_all('a', href=True):
                            full_url = urljoin(camp['base_url'], a['href'])
                            
                            if (re.search(camp['allow_regex'], full_url) and 
                                not re.search(camp['deny_regex'], full_url) and
                                not db.exists(full_url)):
                                new_links.append(full_url)
                                
                    except Exception as e:
                        logging.error(f"List page error {list_url}: {e}")
                    
                    new_links = list(set(new_links))
                    
                    # Check if page is empty
                    if not new_links:
                        consecutive_empty += 1
                        pbar.set_postfix(
                            empty=consecutive_empty,
                            success=campaign_success,
                            failed=campaign_failed
                        )
                        
                        if consecutive_empty >= MAX_EMPTY_PAGES:
                            logging.info(f"✋ Stop {camp['name']}: {MAX_EMPTY_PAGES} empty pages")
                            break
                    else:
                        consecutive_empty = 0
                        
                        # Crawl articles
                        for url in new_links:
                            row = crawl_content(session, url, camp['name'])
                            
                            if row:
                                batch_buffer.append(row)
                                db.add(url, camp['name'], 'success')
                                campaign_success += 1
                                total_new_docs += 1
                                
                                pbar.set_postfix(
                                    success=campaign_success,
                                    failed=campaign_failed,
                                    title=format_title(row['doc_title'], 30)
                                )
                            else:
                                db.add(url, camp['name'], 'failed')
                                campaign_failed += 1
                            
                            # Save checkpoint
                            if len(batch_buffer) >= CHECKPOINT_SIZE:
                                save_batch_incremental(batch_buffer, batch_counter)
                                batch_counter += 1
                                batch_buffer = []
                            
                            time.sleep(random.uniform(0.5, 1.5))
                    
                    page += 1
                
                pbar.close()
                success_rate = (campaign_success/(campaign_success+campaign_failed)*100) if (campaign_success+campaign_failed) > 0 else 0
                print(f"✅ {camp['name']}: {campaign_success} docs | Failed: {campaign_failed} | Rate: {success_rate:.1f}%")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user! Saving progress...")
        
        finally:
            # Save remaining buffer
            if batch_buffer:
                save_batch_incremental(batch_buffer, batch_counter)
            
            # Merge all batches
            print("\n" + "="*70)
            print("📦 FINALIZING...")
            print("="*70)
            total_docs = merge_all_batches()
            
            # Calculate stats
            elapsed = (datetime.now() - start_time).total_seconds()
            total_crawled, total_success, total_failed = db.get_total_stats()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"🎉 CHIẾN DỊCH HOÀN TẤT!")
            print(f"{'='*70}")
            print(f"⏱️  Thời gian      : {format_time(elapsed)}")
            print(f"📁 File output    : {OUTPUT_FILE}")
            print(f"📊 Tổng documents : {total_docs:,} docs")
            print(f"🆕 Docs mới crawl : {total_new_docs:,} docs")
            
            if elapsed > 0:
                print(f"⚡ Tốc độ         : {total_new_docs/elapsed:.2f} docs/sec")
            
            print(f"\n📈 CHI TIẾT THEO NGUỒN:")
            print(f"{'-'*70}")
            print(f"{'Source':<25} {'Total':>8} {'Success':>8} {'Failed':>8} {'Rate':>8}")
            print(f"{'-'*70}")
            
            stats = db.get_stats()
            if stats:
                for source, total, success, failed in stats:
                    rate = (success/total*100) if total > 0 else 0
                    print(f"{source:<25} {total:>8,} {success:>8,} {failed:>8,} {rate:>7.1f}%")
            else:
                print("(Không có dữ liệu)")
            
            print(f"{'-'*70}")
            print(f"{'TỔNG CỘNG':<25} {total_crawled:>8,} {total_success:>8,} {total_failed:>8,} {(total_success/total_crawled*100) if total_crawled > 0 else 0:>7.1f}%")
            print(f"{'='*70}")
            
            print(f"\n💾 History DB: {DB_FILE}")
            print(f"📝 Log file: crawler_v5.log")
            print(f"\n💡 Tip: Chạy lại script để thu thập thêm bài viết mới (incremental mode)")
    
    # Database connection is automatically closed here by context manager
    print("\n✅ All resources cleaned up successfully")

if __name__ == "__main__":
    main()