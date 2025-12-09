import pandas as pd
from qdrant_client import QdrantClient, models
import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from config import Config
from vnpt_client import get_vnpt_embedding

# --- 1. RATE LIMITER (NON-BLOCKING) ---
class RateLimiter:
    def __init__(self, max_calls_per_minute=480):
        # 480 req/phút -> an toàn so với 500
        self.delay = 60.0 / max_calls_per_minute
        self.next_allowed_time = 0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            target_time = max(now, self.next_allowed_time)
            self.next_allowed_time = target_time + self.delay
        
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

def generate_uuid5(unique_string):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def main():
    # 1. Load Data
    if not Config.LATEST_CHUNKS_FILE.exists():
        print("❌ Không có dữ liệu chunks.")
        return
    
    df = pd.read_parquet(Config.LATEST_CHUNKS_FILE)
    total_records = len(df)
    print(f"🔥 Chuẩn bị index {total_records} chunks qua API...")

    # 2. Setup Qdrant
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    
    print("🧪 Đang test API Embedding...")
    sample_vec = None
    for i in range(3):
        sample_vec = get_vnpt_embedding("Kiểm tra vector")
        if sample_vec: break
        print(f"   Retry test {i+1}...")
        time.sleep(2)

    if not sample_vec:
        print("❌ API Embedding lỗi. Kiểm tra mạng hoặc Key.")
        return
    
    vector_size = len(sample_vec)
    print(f"✅ Vector Size chuẩn: {vector_size}")

    if Config.FORCE_RECREATE and client.collection_exists(Config.COLLECTION_NAME):
        client.delete_collection(Config.COLLECTION_NAME)

    if not client.collection_exists(Config.COLLECTION_NAME):
        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            )
        )

    # 3. Indexing Loop
    limiter = RateLimiter(max_calls_per_minute=480)
    records = df.to_dict('records')
    failed_ids = []
    
    BATCH_SIZE = 50 
    
    def process_item(item):
        try:
            limiter.wait()
            vec = get_vnpt_embedding(item['vector_text'])
            if vec:
                return models.PointStruct(
                    id=generate_uuid5(str(item['chunk_id'])),
                    vector=vec,
                    payload={
                        "title": item['doc_title'],
                        "text": item['display_text'],
                        "category": item['doc_category'],
                        "url": item['doc_url'],
                        "chunk_id": item['chunk_id'] # [FIX] Thêm chunk_id vào payload
                    }
                )
        except Exception as e:
            print(f"\n⚠️ Error processing {item.get('chunk_id')}: {e}")
        return None

    print("🚀 Bắt đầu Indexing (Treo máy khoảng 6-8 tiếng)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        for i in range(0, total_records, BATCH_SIZE):
            batch_records = records[i : i + BATCH_SIZE]
            
            future_to_item = {
                executor.submit(process_item, item): item 
                for item in batch_records
            }
            
            valid_points = []
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result:
                        valid_points.append(result)
                    else:
                        failed_ids.append(item['chunk_id'])
                except Exception:
                    failed_ids.append(item['chunk_id'])
            
            # Upsert
            if valid_points:
                try:
                    client.upsert(
                        collection_name=Config.COLLECTION_NAME,
                        points=valid_points
                    )
                except Exception as e:
                    print(f"❌ Lỗi Upsert Qdrant: {e}")
                    # Bây giờ dòng này sẽ chạy đúng vì payload đã có chunk_id
                    failed_ids.extend([p.payload['chunk_id'] for p in valid_points])
            
            # Progress Log
            processed_count = i + len(batch_records)
            percent = (processed_count / total_records) * 100
            print(f"✅ Progress: {processed_count}/{total_records} ({percent:.2f}%) | Fail: {len(failed_ids)}", end='\r')

    print("\n✅ HOÀN TẤT INDEXING!")
    
    if failed_ids:
        print(f"⚠️ Có {len(failed_ids)} chunks bị lỗi. Lưu vào log.")
        with open(Config.FAILED_BATCHES_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_ids, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

