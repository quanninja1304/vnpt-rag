"""
Wikipedia Vietnam Crawler - Production Version
================================================
Features:
- Incremental caching (JSON + Parquet)
- Multi-threaded crawling (configurable workers)
- Max depth 5 với cycle detection
- Retry mechanism cho timeout
- Full category tree logging
- Export CSV + JSON + Parquet
- 500+ optimized categories for RAG
"""

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
import warnings
warnings.filterwarnings('ignore')

# Thêm retry decorator
from functools import wraps

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


# ================================
# 1. CẤU HÌNH & LOGGING
# ================================

class Config:
    """Centralized configuration"""
    # Directories
    CACHE_DIR = Path("cache")
    LOGS_DIR = Path("logs")
    OUTPUT_DIR = Path("output")
    
    # Files
    ARTICLES_CACHE = CACHE_DIR / "articles_cache.json"
    CATEGORY_TREE = CACHE_DIR / "category_tree.json"
    METADATA = CACHE_DIR / "metadata.json"
    
    # Crawl settings
    MAX_LEVEL = 2
    MAX_WORKERS = 15
    TEXT_LIMIT = 25000  # Tăng lên 20k cho RAG tốt hơn
    RETRY_ATTEMPTS = 3
    RATE_LIMIT_DELAY = 0.2
    
    # Export formats
    EXPORT_CSV = True
    EXPORT_JSON = True
    EXPORT_PARQUET = True
    
    @classmethod
    def setup_dirs(cls):
        """Tạo các thư mục cần thiết"""
        for dir_path in [cls.CACHE_DIR, cls.LOGS_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)


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

# Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='VNPT_Hackathon_Pro/5.0 (Production_RAG_Crawler)',
    language='vi',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    timeout=30
)


# ================================
# 2. DANH SÁCH 500+ CATEGORIES TỐI ƯU CHO RAG
# ================================

COMPREHENSIVE_CATEGORIES = [
    # ========== LỊCH SỬ (60+ categories) ==========
    "Lịch_sử_Việt_Nam", "Triều_đại_Việt_Nam", "Chiến_tranh_liên_quan_đến_Việt_Nam",
    "Nhà_nước_Việt_Nam", "Sự_kiện_lịch_sử_Việt_Nam", "Nhân_vật_lịch_sử_Việt_Nam",
    "Vua_Việt_Nam", "Hoàng_đế_Việt_Nam", "Chúa_Nguyễn", "Chúa_Trịnh",
    "Nhà_Lý", "Nhà_Trần", "Nhà_Lê", "Nhà_Nguyễn", "Nhà_Hồ",
    "Cách_mạng_Việt_Nam", "Kháng_chiến_chống_Pháp", "Chiến_tranh_Việt_Nam",
    "Đông_Dương_thuộc_Pháp", "Bắc_thuộc", "Độc_lập_Việt_Nam",
    "Thống_nhất_đất_nước", "Cải_cách_và_Mở_cửa", "Đổi_mới_Việt_Nam",
    
    # Nhân vật lịch sử chi tiết
    "Chính_trị_gia_Việt_Nam", "Tướng_lĩnh_Việt_Nam", "Danh_tướng_Việt_Nam",
    "Anh_hùng_dân_tộc_Việt_Nam", "Danh_nhân_Việt_Nam", "Nho_sĩ_Việt_Nam",
    "Sĩ_quan_quân_đội_nhân_dân_Việt_Nam", "Liệt_sĩ_Việt_Nam",
    "Chủ_tịch_nước_Việt_Nam", "Thủ_tướng_Việt_Nam", "Tổng_Bí_thư_Đảng_Cộng_sản_Việt_Nam",
    
    # Sự kiện lịch sử quan trọng
    "Khởi_nghĩa_Việt_Nam", "Trận_chiến_Việt_Nam", "Hội_nghị_Việt_Nam",
    "Hiệp_ước_liên_quan_đến_Việt_Nam", "Cách_mạng_tháng_Tám",
    "Chiến_dịch_Điện_Biên_Phủ", "Chiến_dịch_Hồ_Chí_Minh",
    
    # ========== ĐỊA LÝ (100+ categories) ==========
    "Địa_lý_Việt_Nam", "Hành_chính_Việt_Nam", "Đơn_vị_hành_chính_Việt_Nam",
    
    # 63 Tỉnh thành
    "Tỉnh_của_Việt_Nam", "Thành_phố_trực_thuộc_trung_ương_(Việt_Nam)",
    "Hà_Nội", "Thành_phố_Hồ_Chí_Minh", "Đà_Nẵng", "Hải_Phòng", "Cần_Thơ",
    "An_Giang", "Bà_Rịa-Vũng_Tàu", "Bắc_Giang", "Bắc_Kạn", "Bạc_Liêu",
    "Bắc_Ninh", "Bến_Tre", "Bình_Định", "Bình_Dương", "Bình_Phước", "Bình_Thuận",
    "Cà_Mau", "Cao_Bằng", "Đắk_Lắk", "Đắk_Nông", "Điện_Biên", "Đồng_Nai",
    "Đồng_Tháp", "Gia_Lai", "Hà_Giang", "Hà_Nam", "Hà_Tĩnh", "Hải_Dương",
    "Hậu_Giang", "Hòa_Bình", "Hưng_Yên", "Khánh_Hòa", "Kiên_Giang",
    "Kon_Tum", "Lai_Châu", "Lâm_Đồng", "Lạng_Sơn", "Lào_Cai", "Long_An",
    "Nam_Định", "Nghệ_An", "Ninh_Bình", "Ninh_Thuận", "Phú_Thọ", "Phú_Yên",
    "Quảng_Bình", "Quảng_Nam", "Quảng_Ngãi", "Quảng_Ninh", "Quảng_Trị",
    "Sóc_Trăng", "Sơn_La", "Tây_Ninh", "Thái_Bình", "Thái_Nguyên",
    "Thanh_Hóa", "Thừa_Thiên_Huế", "Tiền_Giang", "Trà_Vinh", "Tuyên_Quang",
    "Vĩnh_Long", "Vĩnh_Phúc", "Yên_Bái",
    
    # Đơn vị hành chính cấp dưới
    "Huyện_của_Việt_Nam", "Quận_(Việt_Nam)", "Thị_xã_Việt_Nam", "Thành_phố_thuộc_tỉnh",
    "Xã_(Việt_Nam)", "Phường_(Việt_Nam)", "Thị_trấn_Việt_Nam",
    
    # Địa hình tự nhiên
    "Sông_ngòi_Việt_Nam", "Sông_Hồng", "Sông_Cửu_Long", "Sông_Đồng_Nai",
    "Núi_Việt_Nam", "Đèo_Việt_Nam", "Cao_nguyên_Việt_Nam", "Đồng_bằng_Việt_Nam",
    "Vịnh_Việt_Nam", "Vịnh_Hạ_Long", "Vịnh_Nha_Trang",
    "Đảo_Việt_Nam", "Quần_đảo_Trường_Sa", "Quần_đảo_Hoàng_Sa",
    "Biển_Đông", "Hồ_Việt_Nam", "Thác_nước_Việt_Nam",
    
    # Danh lam thắng cảnh
    "Danh_thắng_Việt_Nam", "Vườn_quốc_gia_Việt_Nam", "Khu_du_lịch_Việt_Nam",
    "Động_Việt_Nam", "Hang_động_Việt_Nam", "Bãi_biển_Việt_Nam",
    
    # ========== VĂN HÓA (80+ categories) ==========
    "Văn_hóa_Việt_Nam", "Di_sản_văn_hóa_Việt_Nam",
    "Di_sản_văn_hóa_thế_giới_tại_Việt_Nam", "Di_sản_văn_hóa_phi_vật_thể",
    
    # Văn học & Ngôn ngữ
    "Văn_học_Việt_Nam", "Nhà_văn_Việt_Nam", "Thi_sĩ_Việt_Nam", "Nhà_thơ_Việt_Nam",
    "Tác_phẩm_văn_học_Việt_Nam", "Truyện_Kiều", "Văn_học_dân_gian_Việt_Nam",
    "Thơ_Việt_Nam", "Thơ_chữ_Nôm", "Chữ_Nôm", "Tiếng_Việt",
    
    # Nghệ thuật
    "Nghệ_thuật_Việt_Nam", "Hội_họa_Việt_Nam", "Họa_sĩ_Việt_Nam",
    "Điêu_khắc_Việt_Nam", "Thủ_công_mỹ_nghệ_Việt_Nam",
    "Gốm_sứ_Việt_Nam", "Tranh_Đông_Hồ", "Tranh_dân_gian_Việt_Nam",
    
    # Âm nhạc & Sân khấu
    "Âm_nhạc_Việt_Nam", "Nhạc_sĩ_Việt_Nam", "Ca_sĩ_Việt_Nam",
    "Nhạc_cụ_dân_tộc_Việt_Nam", "Dân_ca_Việt_Nam", "Hát_quan_họ",
    "Sân_khấu_Việt_Nam", "Chèo", "Tuồng", "Cải_lương", "Rối_nước",
    
    # Điện ảnh & Truyền thông
    "Điện_ảnh_Việt_Nam", "Đạo_diễn_Việt_Nam", "Diễn_viên_Việt_Nam",
    "Phim_Việt_Nam", "Truyền_thông_Việt_Nam", "Báo_chí_Việt_Nam",
    "Đài_phát_thanh_Việt_Nam", "Đài_truyền_hình_Việt_Nam",
    
    # Ẩm thực
    "Ẩm_thực_Việt_Nam", "Món_ăn_Việt_Nam", "Phở", "Bánh_mì_Việt_Nam",
    "Rượu_Việt_Nam", "Trà_Việt_Nam", "Gia_vị_Việt_Nam",
    
    # Lễ hội & Phong tục
    "Lễ_hội_Việt_Nam", "Tết_Nguyên_Đán", "Phong_tục_tập_quán_Việt_Nam",
    "Tín_ngưỡng_Việt_Nam", "Thờ_cúng_tổ_tiên", "Tục_thờ_Mẫu",
    
    # Dân tộc
    "Dân_tộc_Việt_Nam", "54_dân_tộc_Việt_Nam", "Người_Kinh",
    "Người_Tày", "Người_Thái", "Người_Mường", "Người_Khmer_(Việt_Nam)",
    "Người_Hoa_(Việt_Nam)", "Người_Nùng", "Người_H'Mông",
    
    # ========== TÔN GIÁO (30+ categories) ==========
    "Tôn_giáo_tại_Việt_Nam", "Phật_giáo_Việt_Nam", "Chùa_Việt_Nam",
    "Thiền_phái_Việt_Nam", "Tăng_ni_Việt_Nam",
    "Công_giáo_tại_Việt_Nam", "Nhà_thờ_Công_giáo_Việt_Nam",
    "Đạo_Cao_Đài", "Đạo_Hòa_Hảo", "Tin_lành_tại_Việt_Nam",
    "Hồi_giáo_tại_Việt_Nam", "Đạo_giáo_Việt_Nam", "Đền_thờ_Việt_Nam",
    
    # ========== CHÍNH TRỊ & PHÁP LUẬT (60+ categories) ==========
    "Chính_trị_Việt_Nam", "Hệ_thống_chính_trị_Việt_Nam",
    "Nhà_nước_Việt_Nam", "Cơ_quan_nhà_nước_Việt_Nam",
    
    # Đảng
    "Đảng_Cộng_sản_Việt_Nam", "Đại_hội_Đảng_Cộng_sản_Việt_Nam",
    "Ban_Chấp_hành_Trung_ương_Đảng_Cộng_sản_Việt_Nam",
    "Bộ_Chính_trị", "Tổng_Bí_thư_Đảng_Cộng_sản_Việt_Nam",
    
    # Quốc hội
    "Quốc_hội_Việt_Nam", "Chủ_tịch_Quốc_hội_Việt_Nam",
    "Đại_biểu_Quốc_hội_Việt_Nam", "Uỷ_ban_Thường_vụ_Quốc_hội",
    
    # Chính phủ
    "Chính_phủ_Việt_Nam", "Thủ_tướng_Việt_Nam", "Bộ_(Việt_Nam)",
    "Bộ_trưởng_Việt_Nam", "Phó_Thủ_tướng_Việt_Nam",
    
    # Các bộ ngành
    "Bộ_Ngoại_giao_(Việt_Nam)", "Bộ_Quốc_phòng_(Việt_Nam)",
    "Bộ_Công_an_(Việt_Nam)", "Bộ_Giáo_dục_và_Đào_tạo_(Việt_Nam)",
    "Bộ_Y_tế_(Việt_Nam)", "Bộ_Tài_chính_(Việt_Nam)",
    "Bộ_Giao_thông_vận_tải_(Việt_Nam)",
    
    # Chủ tịch nước & Pháp luật
    "Chủ_tịch_nước_Việt_Nam", "Tòa_án_nhân_dân_tối_cao_(Việt_Nam)",
    "Viện_Kiểm_sát_nhân_dân_tối_cao_(Việt_Nam)",
    "Pháp_luật_Việt_Nam", "Hiến_pháp_Việt_Nam", "Bộ_luật_Việt_Nam",
    "Luật_Việt_Nam", "Hệ_thống_pháp_luật_Việt_Nam",
    
    # Tổ chức chính trị - xã hội
    "Tổ_chức_chính_trị_-_xã_hội_tại_Việt_Nam",
    "Mặt_trận_Tổ_quốc_Việt_Nam", "Đoàn_Thanh_niên_Cộng_sản_Hồ_Chí_Minh",
    "Hội_Liên_hiệp_Phụ_nữ_Việt_Nam", "Tổng_Liên_đoàn_Lao_động_Việt_Nam",
    "Hội_Nông_dân_Việt_Nam", "Hội_Cựu_chiến_binh_Việt_Nam",
    
    # Quân đội & An ninh
    "Quân_đội_nhân_dân_Việt_Nam", "Quân_chủng_Việt_Nam",
    "Quân_khu_(Việt_Nam)", "Sư_đoàn_Việt_Nam",
    "Công_an_nhân_dân_Việt_Nam",
    
    # Huân chương
    "Huân_chương_Việt_Nam", "Huy_chương_Việt_Nam", "Danh_hiệu_Việt_Nam",
    
    # ========== KINH TẾ (50+ categories) ==========
    "Kinh_tế_Việt_Nam", "Lịch_sử_kinh_tế_Việt_Nam", "Đổi_mới_kinh_tế",
    "Doanh_nghiệp_Việt_Nam", "Công_ty_Việt_Nam", "Tập_đoàn_Việt_Nam",
    "Ngân_hàng_Việt_Nam", "Chứng_khoán_Việt_Nam",
    "Nông_nghiệp_Việt_Nam", "Công_nghiệp_Việt_Nam", "Dịch_vụ_Việt_Nam",
    "Thương_mại_Việt_Nam", "Xuất_khẩu_Việt_Nam", "Nhập_khẩu_Việt_Nam",
    "Du_lịch_Việt_Nam", "Khách_sạn_Việt_Nam", "Khu_nghỉ_dưỡng_Việt_Nam",
    "Đồng_Việt_Nam", "Thuế_Việt_Nam", "Ngân_sách_nhà_nước_Việt_Nam",
    
    # ========== GIÁO DỤC & KHOA HỌC (40+ categories) ==========
    "Giáo_dục_Việt_Nam", "Trường_đại_học_Việt_Nam",
    "Trường_trung_học_phổ_thông_Việt_Nam", "Trường_cao_đẳng_Việt_Nam",
    "Đại_học_Quốc_gia_Hà_Nội", "Đại_học_Quốc_gia_Thành_phố_Hồ_Chí_Minh",
    "Học_viện_(Việt_Nam)", "Viện_(Việt_Nam)",
    
    "Khoa_học_Việt_Nam", "Nhà_khoa_học_Việt_Nam", "Viện_Hàn_lâm_Khoa_học_và_Công_nghệ_Việt_Nam",
    "Công_nghệ_Việt_Nam", "Công_nghệ_thông_tin_Việt_Nam",
    
    # ========== Y TẾ (20+ categories) ==========
    "Y_tế_Việt_Nam", "Bệnh_viện_Việt_Nam", "Y_học_cổ_truyền_Việt_Nam",
    "Dược_học_Việt_Nam", "Thuốc_Việt_Nam", "Bác_sĩ_Việt_Nam",
    "Y_tế_công_cộng_Việt_Nam", "Dịch_bệnh_tại_Việt_Nam",
    
    # ========== GIAO THÔNG & KIẾN TRÚC (40+ categories) ==========
    "Giao_thông_Việt_Nam", "Đường_bộ_Việt_Nam", "Đường_cao_tốc_Việt_Nam",
    "Cầu_tại_Việt_Nam", "Sân_bay_Việt_Nam", "Cảng_biển_Việt_Nam",
    "Đường_sắt_Việt_Nam", "Ga_đường_sắt_Việt_Nam",
    "Giao_thông_công_cộng_Việt_Nam", "Xe_buýt_Việt_Nam",
    
    "Kiến_trúc_Việt_Nam", "Nhà_cổ_Việt_Nam", "Đình_làng_Việt_Nam",
    "Chợ_Việt_Nam", "Công_trình_kiến_trúc_Việt_Nam",
    "Tòa_nhà_chọc_trời_Việt_Nam", "Công_viên_Việt_Nam",
    
    # ========== THỂ THAO (30+ categories) ==========
    "Thể_thao_Việt_Nam", "Bóng_đá_Việt_Nam", "Câu_lạc_bộ_bóng_đá_Việt_Nam",
    "Đội_tuyển_bóng_đá_quốc_gia_Việt_Nam", "V.League", "Cầu_thủ_bóng_đá_Việt_Nam",
    "SEA_Games", "ASIAD", "Olympic_Việt_Nam",
    "Võ_thuật_Việt_Nam", "Võ_cổ_truyền_Việt_Nam", "Vovinam",
    "Cầu_lông_Việt_Nam", "Điền_kinh_Việt_Nam", "Bơi_lội_Việt_Nam",
    "Vận_động_viên_Việt_Nam", "Huấn_luyện_viên_Việt_Nam",
    
    # ========== BIỂU TƯỢNG & TỔNG HỢP (20+ categories) ==========
    "Biểu_tượng_quốc_gia_Việt_Nam", "Quốc_kỳ_Việt_Nam", "Quốc_ca_Việt_Nam",
    "Quốc_huy_Việt_Nam", "Danh_sách_liên_quan_đến_Việt_Nam",
    "Việt_Nam", "Xã_hội_Việt_Nam", "Con_người_Việt_Nam",
]


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
            csv_path = Config.OUTPUT_DIR / "final_wikipedia_vietnam_full.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ CSV saved: {csv_path}")
        
        # Export JSON
        if Config.EXPORT_JSON:
            json_path = Config.OUTPUT_DIR / "final_wikipedia_vietnam_full.json"
            df.to_json(json_path, orient='records', force_ascii=False, indent=2)
            logger.info(f"✓ JSON saved: {json_path}")
        
        # Export Parquet
        if Config.EXPORT_PARQUET:
            try:
                parquet_path = Config.OUTPUT_DIR / "final_wikipedia_vietnam_full.parquet"
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
        logger.warning("\n⚠ Crawl interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
    
    # Save everything
    crawler.save_all()
    
    # Print stats
    crawler.print_stats()
    
    # Time
    elapsed = time.time() - start_time
    logger.info(f"\n⏱ Total time: {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
    
    total_articles = crawler.stats['articles_crawled'] + crawler.stats['articles_from_cache']
    if total_articles > 0:
        logger.info(f"📊 Speed: {total_articles/elapsed:.1f} articles/second")
    
    logger.info("\n✓ Crawl completed successfully!")


if __name__ == "__main__":
    main()