# build_bm25.py
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from tqdm import tqdm
from config import Config
import os

def build_bm25():
    # 1. Load dữ liệu text gốc
    # File này bạn đã có sau khi chạy chunking.py
    input_file = Config.LATEST_CHUNKS_FILE # hoặc file master nếu bạn gộp nhiều lần
    
    if not input_file.exists():
        print("❌ Không tìm thấy file dữ liệu chunks.")
        return

    print(f"📂 Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_parquet(input_file)
    
    # Chỉ lấy cột text và id
    documents = df['vector_text'].tolist() # Text dùng để search
    chunk_ids = df['chunk_id'].tolist()
    
    print(f"⚡ Đang tách từ (Tokenizing) cho {len(documents)} văn bản...")
    # Tokenize tiếng Việt: "Hà Nội" -> ["Hà Nội"] thay vì ["Hà", "Nội"]
    # Quá trình này có thể mất 10-15 phút cho 400k dòng, hãy kiên nhẫn
    tokenized_corpus = []
    for doc in tqdm(documents):
        # word_tokenize giúp BM25 hiểu cụm từ tiếng Việt
        tokens = word_tokenize(doc.lower()) 
        tokenized_corpus.append(tokens)

    print("🏗️ Đang xây dựng BM25 Index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Lưu metadata để map lại từ kết quả BM25 -> Chunk ID
    # Ta lưu cả object BM25 và danh sách ID tương ứng
    output_data = {
        "bm25_obj": bm25,
        "chunk_ids": chunk_ids,
        "texts": df['display_text'].tolist(), # Lưu text gốc để hiển thị luôn (đỡ phải query lại)
        "titles": df['doc_title'].tolist()
    }

    output_path = Config.OUTPUT_DIR / "bm25_index.pkl"
    print(f"💾 Đang lưu file index vào: {output_path}")
    
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
        
    print("✅ HOÀN TẤT! File này nặng khoảng vài trăm MB -> 1GB. Nhớ copy vào Docker.")

if __name__ == "__main__":
    build_bm25()