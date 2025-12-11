# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration"""
    # --- 1. DIRECTORIES & FILES ---
    BASE_DIR = Path(__file__).parent
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Crawler Files
    ARTICLES_CACHE = CACHE_DIR / "articles_cache.json"
    CATEGORY_TREE = CACHE_DIR / "category_tree.json"
    METADATA = CACHE_DIR / "metadata.json"
    
    # Output của Crawler (Luôn là file chứa TOÀN BỘ dữ liệu thô)
    CRAWL_OUTPUT_JSON = OUTPUT_DIR / "final_wikipedia_vietnam_full.json"
    CRAWL_OUTPUT_CSV = OUTPUT_DIR / "final_wikipedia_vietnam_full.csv"
    CRAWL_OUTPUT_PARQUET = OUTPUT_DIR / "final_wikipedia_vietnam_full.parquet"
    CRAWL_TEMP_JSONL = OUTPUT_DIR / "temp_raw_data.jsonl"
    # --- FILES QUẢN LÝ TRẠNG THÁI (NEW) ---
    # File lưu danh sách tiêu đề các bài đã được chunking xong
    CHUNKING_STATE_FILE = OUTPUT_DIR / "chunking_state.json"
    
    # File chứa các chunk MỚI NHẤT vừa sinh ra (Delta) - Dùng để Indexing
    LATEST_CHUNKS_FILE = OUTPUT_DIR / "delta_chunks_to_index.parquet"
    LAW_CHUNKS_FILE = OUTPUT_DIR / "1_manual_law_strict.parquet"

    # File chứa TOÀN BỘ chunks (Master) - Để lưu trữ lâu dài
    MASTER_CHUNKS_FILE = OUTPUT_DIR / "wiki_vn_chunks_master.parquet"

    # Pipeline Flow
    CHUNKING_INPUT_FILE = CRAWL_OUTPUT_PARQUET 
    
    # Indexing sẽ đọc file DELTA thay vì file full
    INDEXING_INPUT_FILE = LATEST_CHUNKS_FILE

    # checkpoint
    CHECKPOINT_FILE = OUTPUT_DIR / "indexing_checkpoint.txt"

    # Crawl settings
    MAX_LEVEL = 2
    MAX_WORKERS = 20 # Tăng worker vì cơ chế Queue xử lý rất nhanh
    TEXT_LIMIT = 30000
    RETRY_ATTEMPTS = 3
    RATE_LIMIT_DELAY = 0.05
    
    # --- 3. CHUNKING SETTINGS ---
    TOKENIZER_NAME = "bkai-foundation-models/vietnamese-bi-encoder" 
    CHUNK_SIZE_TOKENS = 2048
    CHUNK_OVERLAP_TOKENS = 256

    # --- 4. INDEXING & QDRANT SETTINGS ---
    COLLECTION_NAME = "vnpt_hackathon_rag"
    MODEL_PATH = "vnptai_hackathon_embedding"
    DUMMY_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder" 

    VNPT_API_URL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions"
    VNPT_EMBEDDING_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"

    VNPT_ACCESS_TOKEN = os.getenv("VNPT_ACCESS_TOKEN", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6ImM1NTJlMjY1LTE1YmItNGMxOS1iM2M4LTI5ZjMxZTEzOTY2MSIsInN1YiI6IjQyZWZmOGVkLWQxMmEtMTFmMC05M2Q1LTMxMzk4NzZkNGIyMyIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ3V5ZW5kYWlxdWFuMDZAZ21haWwuY29tIiwic2NvcGUiOlsicmVhZCJdLCJpc3MiOiJodHRwczovL2xvY2FsaG9zdCIsIm5hbWUiOiJuZ3V5ZW5kYWlxdWFuMDZAZ21haWwuY29tIiwidXVpZF9hY2NvdW50IjoiNDJlZmY4ZWQtZDEyYS0xMWYwLTkzZDUtMzEzOTg3NmQ0YjIzIiwiYXV0aG9yaXRpZXMiOlsiVVNFUiIsIlRSQUNLXzIiXSwianRpIjoiMWRmNTQ1ZWEtMmQ2ZC00NzU1LWJjYjctNDFkYjRmNDVhMTU1IiwiY2xpZW50X2lkIjoiYWRtaW5hcHAifQ.jwo4XbiZfAsWE14F2v5S2KO6YNLvES_AwVPJt8wpWNODkUCvA0YDb34BtfTZNXCPZBIPZbkFL25xKd4zPxHy3ZuQsOXsMDd98xD5v1qZtCT1_BqrewRig1btzXrhocU2TMJH_76VIv4KKyHmUWFJwftBd2wy2ixo8k3ojOwgUMxp4X8rfSYTruXs1M7mGNlkDYYTtUfevV_2YFwvcB8pmRjrfklMtBIrz2sTKnDDKn_ML8jJ-ipUuz0kvA8Tyn79PFyp4bfV76NwsDcxP-IeVZAiVS8c47dOQ3nQHXmsNbjI6dB34Aa1b7cEJ_fArCE261PosISSfhdPU1hlgg5p8w")

    # Tên Model
    LLM_MODEL_LARGE = "vnptai_hackathon_large"
    LLM_MODEL_SMALL = "vnptai_hackathon_small"
    MODEL_EMBEDDING_API = "vnptai_hackathon_embedding"

    VNPT_CREDENTIALS = {
        # Config cho Large
        LLM_MODEL_LARGE: {
            "token_id": os.getenv("VNPT_LARGE_ID", "4525a842-6caa-553c-e063-62199f0a1086"),
            "token_key": os.getenv("VNPT_LARGE_KEY", "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAOfzHEYrHzU8anUg0pAukEdooGkBSSduyuHJj+hK0NukYmQCAfaxoQ5jUD9ekVFEGwlzg6BOAxpGo+2Cj2RY5UMCAwEAAQ==")
        },
        # Config cho Small
        LLM_MODEL_SMALL: {
            "token_id": os.getenv("VNPT_SMALL_ID", "4525a842-6cab-553c-e063-62199f0a1086"),
            "token_key": os.getenv("VNPT_SMALL_KEY", "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJIZcz/zVyTEiNX6eICBMdBacp7LD5swBGZkv7Y9wQItZlN+JfzrFqwEd7ywPJpHcnwt78CAWnnvLl/NmSU9v70CAwEAAQ==")
        },
        MODEL_EMBEDDING_API: {
            "token_id": os.getenv("VNPT_EMBED_ID", "4525a84b-0034-2031-e063-62199f0af9db"),
            "token_key": os.getenv("VNPT_EMBED_KEY", "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBANNtXXdFdXwzYMNhQwPNVObxjmwyzrw3dZitGvYw39Ofph/7qLus3P55x4Pf/zt+ohGCBK2XAhhtaw2W1ez8HuECAwEAAQ==")
        }
    }
    
    # Batch Size
    BATCH_SIZE = 256
    
    # [QUAN TRỌNG] Chuyển thành False để hỗ trợ nạp nối tiếp (Incremental)
    FORCE_RECREATE = False

    # Qdrant Connection
    USE_QDRANT_CLOUD = os.getenv("USE_CLOUD", "False").lower() == "true"
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    SPARSE_AVAILABLE = False

    # [NEW] Tham số tối ưu HNSW & Vector (Tuning ở đây dễ hơn)
    HNSW_M = 32
    HNSW_EF_CONSTRUCT = 200
    QUANTIZATION_QUANTILE = 0.99

    # Export formats
    EXPORT_CSV = True
    EXPORT_JSON = True
    EXPORT_PARQUET = True
    
    @classmethod
    def setup_dirs(cls):
        """Tạo các thư mục cần thiết"""
        for dir_path in [cls.CACHE_DIR, cls.LOGS_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(exist_ok=True)

# Danh sách Categories (Dữ liệu tĩnh)
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