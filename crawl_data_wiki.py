import wikipediaapi
import pandas as pd
import time
import json
import logging
from threading import Lock, Thread
from queue import Queue, Empty
from datetime import datetime
from functools import wraps
from config import Config, COMPREHENSIVE_CATEGORIES

# Setup Logging
Config.setup_dirs()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'crawl_{datetime.now():%Y%m%d}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='VNPT_Hackathon_Bot/1.0 (RTX3090_Optimized)',
    language='vi',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    timeout=30
)

def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1: raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

class DataWriter:
    """Ghi dữ liệu streaming ra đĩa để tiết kiệm RAM"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = Lock()
        # Tạo file mới (hoặc clear file cũ nếu chạy mới)
        with open(filepath, 'w', encoding='utf-8'): pass

    def append_article(self, article_data: dict):
        with self.lock:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

class ProductionWikiCrawler:
    def __init__(self):
        self.writer = DataWriter(Config.CRAWL_TEMP_JSONL)
        self.queue = Queue()
        
        # State Tracking
        self.visited = set()
        self.visited_lock = Lock()
        
        # Stats
        self.stats = {"crawled": 0, "errors": 0}
        self.stats_lock = Lock()

    def is_visited(self, name: str) -> bool:
        """Kiểm tra thread-safe xem đã duyệt chưa"""
        with self.visited_lock:
            if name in self.visited:
                return True
            self.visited.add(name)
            return False

    @retry_on_failure(max_retries=Config.RETRY_ATTEMPTS)
    def fetch_page(self, name, is_category=False):
        prefix = "Category:" if is_category else ""
        page = wiki_wiki.page(f"{prefix}{name}")
        if not page.exists():
            raise ValueError(f"Page not found: {name}")
        return page

    def process_article(self, member_obj, parent_category):
        try:
            title = member_obj.title
            if self.is_visited(title): return

            article_data = {
                "title": title,
                "url": member_obj.fullurl,
                "text": member_obj.text[:Config.TEXT_LIMIT],
                "summary": member_obj.summary,
                "categories": [parent_category],
                "crawled_at": datetime.now().isoformat()
            }
            
            self.writer.append_article(article_data)
            
            with self.stats_lock:
                self.stats["crawled"] += 1
                if self.stats["crawled"] % 100 == 0:
                    logger.info(f"⚡ Progress: {self.stats['crawled']} articles saved.")

        except Exception as e:
            logger.error(f"Error article {member_obj.title}: {e}")
            with self.stats_lock: self.stats["errors"] += 1

    def worker_loop(self):
        """Worker Thread: Lấy task từ Queue và xử lý"""
        while True:
            try:
                # Timeout 5s: Nếu queue rỗng quá 5s -> coi như xong việc
                task = self.queue.get(timeout=5)
            except Empty:
                return 

            try:
                cat_name, level = task

                # Check điều kiện dừng
                if level > Config.MAX_LEVEL:
                    continue
                
                # Check visited cho Category (tránh vòng lặp vô hạn)
                if self.is_visited(f"CAT:{cat_name}"):
                    continue

                logger.info(f"📂 Processing [{cat_name}] - Level {level}")
                cat_page = self.fetch_page(cat_name, is_category=True)
                
                # Duyệt members
                for member_name, member_obj in cat_page.categorymembers.items():
                    if member_obj.ns == wikipediaapi.Namespace.MAIN:
                        self.process_article(member_obj, cat_name)
                    
                    elif member_obj.ns == wikipediaapi.Namespace.CATEGORY:
                        if level < Config.MAX_LEVEL:
                            clean_name = member_obj.title.replace("Category:", "").replace("Thể loại:", "").strip()
                            # Đẩy sub-category vào Queue cho worker khác
                            self.queue.put((clean_name, level + 1))
                
                time.sleep(Config.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Failed category {cat_name}: {e}")
            finally:
                # QUAN TRỌNG: Luôn báo task_done để queue.join() không bị treo
                self.queue.task_done()

    def run(self, root_categories):
        start_time = time.time()
        
        # 1. Init Queue
        logger.info(f"Initializing with {len(root_categories)} root categories...")
        for cat in root_categories:
            self.queue.put((cat, 0))

        # 2. Start Workers
        threads = []
        logger.info(f"Starting {Config.MAX_WORKERS} worker threads...")
        for _ in range(Config.MAX_WORKERS):
            t = Thread(target=self.worker_loop)
            t.start()
            threads.append(t)

        # 3. Wait
        self.queue.join()     # Đợi xử lý hết task trong queue
        for t in threads:     # Đợi các thread kết thúc hẳn
            t.join()

        # 4. Finalize
        self.finalize_data()
        
        elapsed = time.time() - start_time
        logger.info(f"🏁 Done! Total: {self.stats['crawled']} articles. Time: {elapsed/60:.2f} mins.")

    def finalize_data(self):
        """Convert JSONL to Parquet"""
        if not Config.CRAWL_TEMP_JSONL.exists(): return
        
        logger.info("Converting JSONL to Parquet...")
        try:
            # Đọc JSONL -> Deduplicate -> Save Parquet
            df = pd.read_json(Config.CRAWL_TEMP_JSONL, lines=True)
            df.drop_duplicates(subset=['title'], inplace=True)
            
            df.to_parquet(Config.CRAWL_OUTPUT_PARQUET, index=False, compression='snappy')
            logger.info(f"✓ Saved Parquet: {Config.CRAWL_OUTPUT_PARQUET} ({len(df)} records)")
            
            # Xóa file temp (Optional)
            # os.remove(Config.CRAWL_TEMP_JSONL)
        except Exception as e:
            logger.error(f"Error finalizing data: {e}")

if __name__ == "__main__":
    crawler = ProductionWikiCrawler()
    crawler.run(COMPREHENSIVE_CATEGORIES)