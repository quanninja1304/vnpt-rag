import wikipediaapi
import pandas as pd
import time
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import logging
from typing import Dict, List, Set, Tuple, Optional
from functools import wraps
import warnings

from config import Config, COMPREHENSIVE_CATEGORIES


warnings.filterwarnings('ignore')

# Setup logging
Config.setup_dirs()
log_file = Config.LOGS_DIR / f'crawl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries=3, delay=2):
    """Decorator để retry khi gặp lỗi"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

# Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='VNPT_Hackathon_Pro/5.0 (Production_RAG_Crawler)',
    language='vi',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    timeout=30
)

# ================================
# 3. CACHE MANAGER - Incremental
# ================================

class IncrementalCache:
    """Cache thông minh với incremental update"""
    
    def __init__(self):
        self.articles: Dict = {}
        self.category_tree: Dict = {}
        self.metadata: Dict = {
            "last_update": None,
            "total_articles": 0,
            "total_categories": 0,
            "version": "2.0"
        }
        
        self.lock = Lock()
        self._load_cache()
    
    def _load_cache(self):
        """Load cache từ disk"""
        # Load articles
        if Config.ARTICLES_CACHE.exists():
            try:
                with open(Config.ARTICLES_CACHE, 'r', encoding='utf-8') as f:
                    self.articles = json.load(f)
                logger.info(f"✓ Loaded {len(self.articles)} articles from cache")
            except Exception as e:
                logger.warning(f"Failed to load articles cache: {e}")
        
        # Load category tree
        if Config.CATEGORY_TREE.exists():
            try:
                with open(Config.CATEGORY_TREE, 'r', encoding='utf-8') as f:
                    self.category_tree = json.load(f)
                logger.info(f"✓ Loaded category tree from cache")
            except Exception as e:
                logger.warning(f"Failed to load category tree: {e}")
        
        # Load metadata
        if Config.METADATA.exists():
            try:
                with open(Config.METADATA, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Last update: {self.metadata.get('last_update', 'N/A')}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def save_incremental(self):
        """Lưu cache (incremental)"""
        with self.lock:
            # Save articles
            with open(Config.ARTICLES_CACHE, 'w', encoding='utf-8') as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
            
            # Save category tree
            with open(Config.CATEGORY_TREE, 'w', encoding='utf-8') as f:
                json.dump(self.category_tree, f, ensure_ascii=False, indent=2)
            
            # Update metadata
            self.metadata.update({
                "last_update": datetime.now().isoformat(),
                "total_articles": len(self.articles),
                "total_categories": len(self.category_tree)
            })
            
            with open(Config.METADATA, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Cache saved: {len(self.articles)} articles, {len(self.category_tree)} categories")
    
    def has_article(self, title: str) -> bool:
        return title in self.articles
    
    def add_article(self, title: str, data: Dict):
        with self.lock:
            self.articles[title] = data
    
    def get_article(self, title: str) -> Optional[Dict]:
        return self.articles.get(title)
    
    def add_to_tree(self, parent: str, child: str, level: int):
        with self.lock:
            if parent not in self.category_tree:
                self.category_tree[parent] = {"level": level, "children": []}
            if child not in self.category_tree[parent]["children"]:
                self.category_tree[parent]["children"].append(child)


# ================================
# 4. CATEGORY TRACKER
# ================================

class CategoryTracker:
    """Theo dõi và phát hiện vòng lặp category"""
    
    def __init__(self):
        self.visited: Set[str] = set()
        self.in_progress: Set[str] = set()
        self.lock = Lock()
        
        # Statistics
        self.stats = {
            "visited": 0,
            "cycle_detected": 0,
            "max_level_reached": 0
        }
    
    def should_visit(self, cat_name: str, level: int, max_level: int) -> Tuple[bool, str]:
        """Kiểm tra xem có nên crawl category này không"""
        with self.lock:
            if level > max_level:
                self.stats["max_level_reached"] += 1
                return False, "max_level"
            
            if cat_name in self.visited:
                return False, "already_visited"
            
            if cat_name in self.in_progress:
                self.stats["cycle_detected"] += 1
                return False, "cycle_detected"
            
            return True, "ok"
    
    def mark_visiting(self, cat_name: str):
        with self.lock:
            self.in_progress.add(cat_name)
    
    def mark_visited(self, cat_name: str):
        with self.lock:
            self.visited.add(cat_name)
            self.stats["visited"] += 1
            if cat_name in self.in_progress:
                self.in_progress.remove(cat_name)


# ================================
# 5. MAIN CRAWLER CLASS
# ================================

class ProductionWikiCrawler:
    """Production-ready Wikipedia crawler"""
    
    def __init__(self, max_level=5, max_workers=25):
        self.max_level = max_level
        self.max_workers = max_workers
        
        self.cache = IncrementalCache()
        self.tracker = CategoryTracker()
        self.lock = Lock()
        
        self.stats = {
            "articles_crawled": 0,
            "articles_from_cache": 0,
            "categories_processed": 0,
            "errors": 0,
            "retries": 0
        }
        
        logger.info(f"Crawler initialized: max_level={max_level}, workers={max_workers}")
    
    @retry_on_failure(max_retries=Config.RETRY_ATTEMPTS)
    def _fetch_wikipedia_page(self, page_name: str, is_category: bool = False):
        """Fetch page với retry mechanism"""
        if is_category:
            page = wiki_wiki.page(f"Category:{page_name}")
        else:
            page = wiki_wiki.page(page_name)
        
        if not page.exists():
            raise ValueError(f"Page not found: {page_name}")
        
        return page
    
    def crawl_article(self, member_name: str, member_obj, parent_category: str) -> Optional[Dict]:
        """Crawl một bài viết"""
        
        # Check cache first
        if self.cache.has_article(member_name):
            with self.lock:
                self.stats["articles_from_cache"] += 1
            return self.cache.get_article(member_name)
        
        try:
            # Retry mechanism tự động từ decorator
            article_data = {
                "title": member_obj.title,
                "url": member_obj.fullurl,
                "text": member_obj.text[:Config.TEXT_LIMIT],
                "summary": member_obj.summary,
                "categories": [parent_category],  # List để có thể merge
                "crawled_at": datetime.now().isoformat(),
                "length": len(member_obj.text)
            }
            
            # Add to cache
            self.cache.add_article(member_name, article_data)
            
            with self.lock:
                self.stats["articles_crawled"] += 1
                current_count = self.stats["articles_crawled"]

            # --- THÊM ĐOẠN NÀY: Auto-save mỗi 1000 bài ---
            if current_count % 1000 == 0:
                logger.info(f"💾 Auto-saving progress at {current_count} articles...")
                self.cache.save_incremental() 
            # ---------------------------------------------
            return article_data
            
        except Exception as e:
            logger.warning(f"Failed to crawl article {member_name}: {e}")
            with self.lock:
                self.stats["errors"] += 1
            return None
    
    def crawl_category(self, cat_name: str, level: int = 0):
        """Crawl một category (đệ quy)"""
        
        # Check điều kiện
        should_visit, reason = self.tracker.should_visit(cat_name, level, self.max_level)
        
        if not should_visit:
            logger.debug(f"{'  ' * level}⊘ Skip [{cat_name}] - {reason}")
            return
        
        # Mark visiting
        self.tracker.mark_visiting(cat_name)
        
        try:
            # Fetch với retry
            cat_page = self._fetch_wikipedia_page(cat_name, is_category=True)
            
            logger.info(f"{'  ' * level}→ [{cat_name}] (level {level})")
            
            members = cat_page.categorymembers
            articles = []
            subcats = []
            
            # Phân loại members
            for member_name, member_obj in members.items():
                try:
                    if member_obj.ns == wikipediaapi.Namespace.MAIN:
                        articles.append((member_name, member_obj))
                    
                    elif member_obj.ns == wikipediaapi.Namespace.CATEGORY:
                        clean_name = member_obj.title.replace("Thể loại:", "").replace("Category:", "").strip()
                        subcats.append(clean_name)
                        self.cache.add_to_tree(cat_name, clean_name, level + 1)
                
                except Exception as e:
                    continue
            
            # Crawl articles
            for member_name, member_obj in articles:
                self.crawl_article(member_name, member_obj, cat_name)
            
            # Mark visited
            self.tracker.mark_visited(cat_name)
            with self.lock:
                self.stats["categories_processed"] += 1
            
            # Crawl subcategories (đệ quy)
            for subcat in subcats:
                self.crawl_category(subcat, level + 1)
            
            # Rate limiting
            time.sleep(Config.RATE_LIMIT_DELAY)
            
        except Exception as e:
            logger.error(f"Error crawling category {cat_name}: {e}")
            self.tracker.mark_visited(cat_name)  # Mark để không retry vô hạn
            with self.lock:
                self.stats["errors"] += 1
    
    def crawl_parallel(self, root_categories: List[str]):
        """Crawl song song các root categories"""
        logger.info(f"Starting parallel crawl: {len(root_categories)} root categories")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.crawl_category, cat, 0): cat
                for cat in root_categories
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Root Categories"):
                cat = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed root category {cat}: {e}")
        
        logger.info("✓ All root categories processed")
    
    def save_all(self):
        """Lưu tất cả dữ liệu"""
        logger.info("Saving data...")
        
        # Save cache
        self.cache.save_incremental()
        
        # Prepare DataFrame
        df = pd.DataFrame(self.cache.articles.values())
        
        if len(df) == 0:
            logger.warning("No data to export!")
            return
        
        # Deduplicate
        df = df.drop_duplicates(subset=['title'], keep='first')
        logger.info(f"Total unique articles: {len(df)}")
        
        # Export CSV
        if Config.EXPORT_CSV:
            csv_path = Config.CRAWL_OUTPUT_CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ CSV saved: {csv_path}")
        
        # Export JSON
        if Config.EXPORT_JSON:
            json_path = Config.CRAWL_OUTPUT_JSON
            df.to_json(json_path, orient='records', force_ascii=False, indent=2)
            logger.info(f"✓ JSON saved: {json_path}")
        
        # Export Parquet
        if Config.EXPORT_PARQUET:
            try:
                parquet_path = Config.CRAWL_OUTPUT_PARQUET
                df.to_parquet(parquet_path, index=False, compression='snappy')
                logger.info(f"✓ Parquet saved: {parquet_path}")
            except Exception as e:
                logger.warning(f"Failed to save Parquet: {e}")
        
        # Export category tree
        tree_path = Config.OUTPUT_DIR / "category_tree_full.json"
        with open(tree_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache.category_tree, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Category tree saved: {tree_path}")
    
    def print_stats(self):
        """In thống kê chi tiết"""
        logger.info("=" * 70)
        logger.info("CRAWL STATISTICS")
        logger.info("=" * 70)
        
        # Crawler stats
        logger.info("Articles:")
        logger.info(f"  - Crawled new: {self.stats['articles_crawled']:,}")
        logger.info(f"  - From cache: {self.stats['articles_from_cache']:,}")
        logger.info(f"  - Total: {self.stats['articles_crawled'] + self.stats['articles_from_cache']:,}")
        
        logger.info("\nCategories:")
        logger.info(f"  - Processed: {self.stats['categories_processed']:,}")
        logger.info(f"  - Visited: {self.tracker.stats['visited']:,}")
        logger.info(f"  - Cycle detected: {self.tracker.stats['cycle_detected']:,}")
        logger.info(f"  - Max level reached: {self.tracker.stats['max_level_reached']:,}")
        
        logger.info("\nErrors & Retries:")
        logger.info(f"  - Errors: {self.stats['errors']:,}")
        logger.info(f"  - Retries: {self.stats['retries']:,}")
        
        logger.info("=" * 70)


# ================================
# 6. MAIN EXECUTION
# ================================

def main():
    """Main function"""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("WIKIPEDIA VIETNAM CRAWLER - PRODUCTION VERSION")
    logger.info("=" * 70)
    logger.info(f"Total root categories: {len(COMPREHENSIVE_CATEGORIES)}")
    logger.info(f"Max level: {Config.MAX_LEVEL}")
    logger.info(f"Max workers: {Config.MAX_WORKERS}")
    logger.info("=" * 70)
    
    # Initialize crawler
    crawler = ProductionWikiCrawler(
        max_level=Config.MAX_LEVEL,
        max_workers=Config.MAX_WORKERS
    )
    
    # Crawl
    try:
        crawler.crawl_parallel(COMPREHENSIVE_CATEGORIES)
    except KeyboardInterrupt:
        logger.warning("\nCrawl interrupted by user")
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
    
    # Save everything
    crawler.save_all()
    
    # Print stats
    crawler.print_stats()
    
    # Time
    elapsed = time.time() - start_time
    logger.info(f"\n⏱ Total time: {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
    
    total_articles = crawler.stats['articles_crawled'] + crawler.stats['articles_from_cache']
    if total_articles > 0:
        logger.info(f"Speed: {total_articles/elapsed:.1f} articles/second")
    
    logger.info("\n✓ Crawl completed successfully!")


if __name__ == "__main__":
    main()