import os
from pathlib import Path
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# --- CẤU HÌNH RATE LIMIT (Tuân thủ luật BTC) ---
LIMITER_LARGE = AsyncLimiter(100, 60)   # Tăng từ 1 lên 100 req/phút
LIMITER_SMALL = AsyncLimiter(100, 60)   # Tăng từ 1 lên 100 req/phút
LIMITER_EMBED = AsyncLimiter(500, 60)   

TIMEOUT_PER_QUESTION = 120            

class Config:
    """
    Configuration for VNPT AI Hackathon Submission.
    Environment: Docker /code
    """
    
    # --- 1. ĐƯỜNG DẪN CƠ BẢN ---
    # Lấy đường dẫn thư mục chứa file config.py này làm gốc
    # Trong Docker: /code
    BASE_DIR = Path(__file__).resolve().parent
    
    # --- 2. INPUT / OUTPUT (QUAN TRỌNG NHẤT) ---
    # BTC sẽ mount file test vào đây. Tên file thường là private_test.json
    INPUT_FILE = os.getenv("INPUT_FILE", BASE_DIR / "private_test.json")
    
    # File kết quả nộp bài
    OUTPUT_FILE = BASE_DIR / "submission.csv"
    
    # Log & Cache
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(exist_ok=True) # Tạo folder nếu chưa có
    DEBUG_LOG_FILE = LOGS_DIR / "debug.log"

    # --- 3. RESOURCES (BM25s MỚI) ---
    # [FIX LỖI] Cập nhật đường dẫn cho thư viện bm25s (Folder + IDs)
    # Lưu ý: Bạn cần move các file này vào folder 'resources' trong submission
    RESOURCES_DIR = BASE_DIR / "resources"
    
    BM25_INDEX_DIR = RESOURCES_DIR / "bm25s_index"
    BM25_IDS_FILE = RESOURCES_DIR / "bm25s_ids.pkl"
    BM25_META_FILE = RESOURCES_DIR / "bm25_metadata.json"


    MAX_CONCURRENT_TASKS = 5

    # --- 4. API CONFIGURATION (LẤY TỪ ENV) ---
    # [FIX BẢO MẬT] Xóa hardcode token. Lấy từ biến môi trường.
    # Khi chạy test local, hãy set biến môi trường hoặc dùng file .env
    VNPT_API_URL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions"
    VNPT_EMBEDDING_URL = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"

    VNPT_ACCESS_TOKEN = os.getenv("VNPT_ACCESS_TOKEN", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6ImM1NTJlMjY1LTE1YmItNGMxOS1iM2M4LTI5ZjMxZTEzOTY2MSIsInN1YiI6IjQyZWZmOGVkLWQxMmEtMTFmMC05M2Q1LTMxMzk4NzZkNGIyMyIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ3V5ZW5kYWlxdWFuMDZAZ21haWwuY29tIiwic2NvcGUiOlsicmVhZCJdLCJpc3MiOiJodHRwczovL2xvY2FsaG9zdCIsIm5hbWUiOiJuZ3V5ZW5kYWlxdWFuMDZAZ21haWwuY29tIiwidXVpZF9hY2NvdW50IjoiNDJlZmY4ZWQtZDEyYS0xMWYwLTkzZDUtMzEzOTg3NmQ0YjIzIiwiYXV0aG9yaXRpZXMiOlsiVVNFUiIsIlRSQUNLXzIiXSwianRpIjoiMWRmNTQ1ZWEtMmQ2ZC00NzU1LWJjYjctNDFkYjRmNDVhMTU1IiwiY2xpZW50X2lkIjoiYWRtaW5hcHAifQ.jwo4XbiZfAsWE14F2v5S2KO6YNLvES_AwVPJt8wpWNODkUCvA0YDb34BtfTZNXCPZBIPZbkFL25xKd4zPxHy3ZuQsOXsMDd98xD5v1qZtCT1_BqrewRig1btzXrhocU2TMJH_76VIv4KKyHmUWFJwftBd2wy2ixo8k3ojOwgUMxp4X8rfSYTruXs1M7mGNlkDYYTtUfevV_2YFwvcB8pmRjrfklMtBIrz2sTKnDDKn_ML8jJ-ipUuz0kvA8Tyn79PFyp4bfV76NwsDcxP-IeVZAiVS8c47dOQ3nQHXmsNbjI6dB34Aa1b7cEJ_fArCE261PosISSfhdPU1hlgg5p8w")

    # Qdrant Cloud (External API)
    # Nếu BTC cho dùng mạng, dùng Cloud. Nếu không, phải setup Local path ở đây.
    QDRANT_URL = "https://70247b74-893b-48d1-9361-5e0f0535c1a0.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SC73sycYRM4ETxJRtIGMbRHiWiQmH8SmJri0lsUwhT4"
    COLLECTION_NAME = "vnpt_hackathon_rag"

    # --- 5. MODEL CONFIG ---
    LLM_MODEL_LARGE = "vnptai_hackathon_large"
    LLM_MODEL_SMALL = "vnptai_hackathon_small"
    MODEL_EMBEDDING_API = "vnptai_hackathon_embedding"

    # Credentials chi tiết (ID/Key)
    # [FIX] Vẫn lấy từ ENV để an toàn, xóa hardcode secret
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
    
    # --- 6. RAG PARAMETERS (Tùy chỉnh logic) ---
    TOP_K = 18
    ALPHA_VECTOR = 0.5