#build_bm25.py:

import bm25s
import pandas as pd
import pickle
import string
import os
import sys
import json
import gc
import pyarrow.parquet as pq
from underthesea import word_tokenize
from tqdm import tqdm
from config import Config
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
INDEX_VERSION = "4.1"  # Major.Minor (Minor change: Optimization)
TOKENIZER_NAME = "underthesea_word_tokenize"

# --- 1. PREPROCESSOR (WORKER FUNCTION) ---
# Hàm này phải đặt ở top-level để Multiprocessing pickle được
translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def preprocess_text_worker(text):
    """
    Hàm xử lý đơn lẻ cho từng text.
    Tối ưu hóa string manipulation trước khi gọi tokenizer nặng.
    """
    if not isinstance(text, str) or not text:
        return []
    
    # 1. Lowercase & Remove Punctuation (Nhanh, Python C-optimized)
    # Dùng translate nhanh hơn replace/regex nhiều lần
    clean_text = text.lower().translate(translator)
    
    # 2. Tokenize (Chậm, CPU bound)
    # Chỉ gọi underthesea khi text đã sạch
    tokens = word_tokenize(clean_text)
    
    # 3. Filter empty (Nhanh)
    return [t for t in tokens if len(t.strip()) > 0]

# --- 2. BUILDER CLASS ---
class BM25Builder:
    def __init__(self):
        self.output_dir = Config.BASE_DIR / "bm25s_index"
        self.id_map_file = Config.BASE_DIR / "bm25s_ids.pkl"
        self.metadata_file = Config.BASE_DIR / "bm25_metadata.json"
        
        self.all_ids = []
        self.all_tokens = [] # List[List[str]] - Nhẹ hơn raw text nhiều
        
        Config.BASE_DIR.mkdir(parents=True, exist_ok=True)

    def process_files_streaming(self):
        """
        Đọc và xử lý từng file một (Streaming) để tiết kiệm RAM.
        """
        files = list((Config.BASE_DIR / "output_batch_chunking").glob("*.parquet"))
        files.sort()
        
        print(f"🚀 Bắt đầu xử lý {len(files)} file theo cơ chế Streaming...")
        
        # Tận dụng tối đa CPU (trừ 1 core cho OS)
        num_cores = max(1, cpu_count() - 1)
        
        # Tạo Pool một lần dùng cho toàn bộ quá trình
        with Pool(processes=num_cores) as pool:
            
            for file_path in tqdm(files, desc="Processing Files"):
                # 1. Đọc 1 file vào RAM (Pyarrow nhanh hơn Pandas)
                try:
                    table = pq.read_table(file_path, columns=['chunk_id', 'vector_text'])
                    
                    # Convert sang list python (nhanh hơn xử lý vector pandas cho text)
                    batch_ids = table['chunk_id'].to_pylist()
                    batch_texts = table['vector_text'].to_pylist()
                    
                    # Giải phóng table pyarrow ngay
                    del table
                except Exception as e:
                    print(f"❌ Lỗi đọc file {file_path.name}: {e}")
                    continue

                # 2. Tokenize song song cho batch này
                # chunksize lớn để giảm overhead IPC (Inter-Process Communication)
                batch_tokens = pool.map(preprocess_text_worker, batch_texts, chunksize=2000)
                
                # 3. Lưu kết quả vào list tổng
                self.all_ids.extend([str(i) for i in batch_ids])
                self.all_tokens.extend(batch_tokens)
                
                # 4. DỌN DẸP RAM NGAY LẬP TỨC
                del batch_ids
                del batch_texts
                del batch_tokens
                gc.collect() # Ép Python trả RAM cho OS
        
        print(f"✅ Tokenization hoàn tất. Tổng chunks: {len(self.all_ids)}")

    def build_and_save(self):
        if not self.all_tokens:
            print("❌ Không có dữ liệu để build index.")
            return

        print(f"\n🏗️ Đang Build Index BM25s (Method: Luceta)...")
        # Khởi tạo không truyền corpus để tiết kiệm RAM lúc init
        retriever = bm25s.BM25(method='lucene')
        
        # Indexing
        retriever.index(self.all_tokens)
        
        # Giải phóng list tokens khổng lồ ngay sau khi index xong
        print("🧹 Giải phóng RAM token list...")
        del self.all_tokens
        gc.collect()

        print(f"\n💾 Đang lưu xuống đĩa...")
        
        # 1. Lưu Index
        retriever.save(self.output_dir)
        
        # 2. Lưu ID Map
        with open(self.id_map_file, "wb") as f:
            pickle.dump(self.all_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # 3. Lưu Metadata (Version Control)
        metadata = {
            "version": INDEX_VERSION,
            "created_at": datetime.now().isoformat(),
            "num_chunks": len(self.all_ids),
            "tokenizer": TOKENIZER_NAME,
            "library": "bm25s"
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        print("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")
        
        # Tính kích thước thư mục index
        total_size = sum(f.stat().st_size for f in self.output_dir.glob('**/*') if f.is_file())
        print(f"📦 Index Size: {total_size / (1024*1024):.2f} MB")
        print(f"🔖 Metadata: {self.metadata_file}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
        
    builder = BM25Builder()
    builder.process_files_streaming()
    builder.build_and_save()