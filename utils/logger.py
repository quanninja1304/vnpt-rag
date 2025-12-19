import logging
from datetime import datetime
from config import Config


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        # Dùng str() bao quanh path để an toàn tuyệt đối
        logging.FileHandler(str(Config.LOGS_DIR / 'inference_resume.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VNPT_BOT")

def write_debug_log(qid, question, route_tag, model_used, answer, true_label=None, note=""):
    """Hàm ghi log chi tiết vào file txt"""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Kiểm tra đúng sai nếu có đáp án mẫu
        result_status = ""
        if true_label:
            result_status = "✅ ĐÚNG" if str(answer).strip() == str(true_label).strip() else f"❌ SAI (Đúng là {true_label})"
        
        log_content = f"""
--------------------------------------------------------------------------------
[{timestamp}] QID: {qid}
❓ Question: {question}
🏷️ Route: {route_tag} | 🤖 Model: {model_used}
📝 Answer: {answer} {result_status}
ℹ️ Note: {note}
--------------------------------------------------------------------------------
"""
        # Mở file mode 'a' (append) để ghi nối tiếp
        with open(Config.DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_content)
            
    except Exception as e:
        print(f"Lỗi ghi log: {e}")