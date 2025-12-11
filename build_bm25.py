import pandas as pd
import pickle
import string
import os
import shutil
import tempfile
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from tqdm import tqdm
from config import Config
from multiprocessing import Pool, cpu_count

# --- VERSION CONTROL ---
# Tăng số này lên nếu bạn thay đổi cấu trúc dữ liệu lưu trong pickle
INDEX_VERSION = 2 

# --- HÀM XỬ LÝ TEXT ---
def preprocess_text(text):
    if not text: return []
    # Chuyển về string để tránh lỗi nếu dữ liệu là số/None
    text = str(text).lower()
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = word_tokenize(text)
    return [t for t in tokens if len(t.strip()) > 0]

def save_atomic(data, filepath):
    """Ghi file an toàn: Ghi vào temp -> Move đè lên file cũ"""
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)
    
    # Tạo file tạm
    tmp_f = tempfile.NamedTemporaryFile(delete=False, dir=dirname, suffix=".tmp")
    try:
        pickle.dump(data, tmp_f)
        tmp_f.close() # Đóng file để flush buffer
        # Move atomic (Ghi đè an toàn)
        shutil.move(tmp_f.name, filepath)
        print(f"✅ Saved atomically to: {filepath}")
    except Exception as e:
        os.unlink(tmp_f.name) # Xóa file tạm nếu lỗi
        raise e

def build_bm25_incremental():
    output_path = Config.OUTPUT_DIR / "bm25_index.pkl"
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. State Variables
    master_corpus = []
    master_ids = []
    master_texts = []
    master_titles = []
    existing_ids = set() # Set để lookup O(1)

    # 2. Load Old Index (Backward Compatibility check)
    if output_path.exists():
        print(f"🔄 Kiểm tra file index cũ: {output_path}")
        try:
            with open(output_path, "rb") as f:
                data = pickle.load(f)
            
            # Check Version
            if data.get("version", 0) < INDEX_VERSION:
                print(f"⚠️ Index cũ (v{data.get('version')}) không tương thích v{INDEX_VERSION}. Rebuild toàn bộ.")
            elif "tokenized_corpus" in data:
                print("✅ Load thành công dữ liệu cũ.")
                master_corpus = data["tokenized_corpus"]
                master_ids = data["chunk_ids"]
                master_texts = data["texts"]
                master_titles = data["titles"]
                existing_ids = set(map(str, master_ids)) # Đảm bảo ID là string để so sánh chuẩn
                print(f"📊 Dữ liệu hiện có: {len(master_ids)} chunks.")
            else:
                print("⚠️ File cũ thiếu dữ liệu corpus. Rebuild toàn bộ.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file cũ ({e}). Sẽ build mới.")

    # 3. Files to Process
    files_to_index = [
        Config.LATEST_CHUNKS_FILE,
        Config.BASE_DIR / "data" / "law_chunks_ready.parquet"
    ]

    new_documents = [] # Buffer chứa dữ liệu mới cần tokenize

    print("\n🔍 Đang quét dữ liệu mới (Batch Processing)...")
    
    for file_path in files_to_index:
        if not file_path.exists():
            continue
            
        print(f"   📂 Đang quét: {file_path.name}")
        
        try:
            # [MEMORY SAFETY] Đọc file Parquet theo từng batch (tránh tràn RAM với file lớn)
            parquet_file = pq.ParquetFile(file_path)
            
            # Kiểm tra Schema
            required_cols = {'chunk_id', 'vector_text'}
            file_schema = set(parquet_file.schema.names)
            if not required_cols.issubset(file_schema):
                print(f"   ⚠️ Bỏ qua {file_path.name}: Thiếu cột {required_cols - file_schema}")
                continue

            # Batch size 10k dòng để cân bằng tốc độ/RAM
            for batch in parquet_file.iter_batches(batch_size=10000, columns=['chunk_id', 'vector_text', 'display_text', 'doc_title']):
                df_batch = batch.to_pandas()
                
                # [PERFORMANCE] Dùng itertuples nhanh gấp nhiều lần iterrows
                for row in df_batch.itertuples(index=False):
                    cid = str(row.chunk_id)
                    
                    # [CORRECT LOGIC] Check duplicate TRƯỚC khi xử lý
                    if cid in existing_ids:
                        continue
                    
                    # Add ngay vào set để chặn các dòng trùng lặp tiếp theo ngay trong vòng lặp này
                    existing_ids.add(cid)
                    
                    new_documents.append({
                        "text_to_process": row.vector_text,
                        "chunk_id": cid,
                        # Fallback an toàn nếu display_text null
                        "display_text": getattr(row, 'display_text', row.vector_text),
                        "doc_title": getattr(row, 'doc_title', '')
                    })
                    
        except Exception as e:
            print(f"   ❌ Lỗi đọc file {file_path.name}: {e}")

    # 4. Check if update needed
    count_new = len(new_documents)
    print(f"   => Tổng cộng tìm thấy {count_new} chunks MỚI.")
    
    if count_new == 0:
        print("\n🎉 Index đã cập nhật nhất. Không cần làm gì thêm.")
        return

    # 5. Tokenize (Parallel)
    print(f"\n⚡ Đang tách từ (Multiprocessing) cho {count_new} chunks...")
    texts_to_process = [d['text_to_process'] for d in new_documents]
    
    num_processes = max(1, cpu_count() - 1)
    with Pool(processes=num_processes) as pool:
        new_tokenized = list(tqdm(
            pool.imap(preprocess_text, texts_to_process, chunksize=100),
            total=count_new,
            desc="Tokenizing"
        ))

    # 6. Merge Data
    print("📥 Đang gộp dữ liệu...")
    master_corpus.extend(new_tokenized)
    master_ids.extend([d['chunk_id'] for d in new_documents])
    master_texts.extend([d['display_text'] for d in new_documents])
    master_titles.extend([d['doc_title'] for d in new_documents])

    # 7. Re-calculate BM25 (Fast)
    print(f"🏗️ Đang tính toán lại trọng số BM25 cho {len(master_corpus)} chunks...")
    bm25 = BM25Okapi(master_corpus)

    # 8. Save (Atomic)
    output_data = {
        "version": INDEX_VERSION, # Đánh dấu version
        "bm25_obj": bm25,
        "tokenized_corpus": master_corpus,
        "chunk_ids": master_ids,
        "texts": master_texts,
        "titles": master_titles
    }

    print(f"💾 Đang lưu file (Atomic)...")
    save_atomic(output_data, output_path)

    # Stats
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ HOÀN TẤT! Tổng DB: {len(master_ids)} chunks. Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    build_bm25_incremental()