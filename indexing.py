import asyncio
import aiohttp
import pandas as pd
import json
import uuid
import sys
import os
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm
from qdrant_client import QdrantClient, models
from config import Config

# --- 1. CẤU HÌNH TỐI ƯU ---
# Rate Limit an toàn: 480 req/phút (tối đa 500)
RATE_LIMITER = AsyncLimiter(480, 60)

# Số lượng Concurrent Workers (Async nhẹ nên có thể để 20-30)
NUM_WORKERS = 30

# Số lượng vector gom lại trước khi Upsert vào Qdrant
UPSERT_BATCH_SIZE = 50

# --- 2. HÀM HỖ TRỢ ---
def save_checkpoint(ids_list):
    """Lưu danh sách ID đã xong vào file (append mode)"""
    if not ids_list: return
    try:
        with open(Config.CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            for chunk_id in ids_list:
                f.write(f"{chunk_id}\n")
    except Exception as e:
        print(f"⚠️ Lỗi ghi checkpoint: {e}")

def generate_uuid5(unique_string):
    """Tạo UUID cố định dựa trên chuỗi nhập vào (Idempotent)"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(unique_string)))

async def get_embedding_async(session, text, retry_attempts=3):
    """
    Phiên bản Async của get_vnpt_embedding.
    Sử dụng aiohttp để không chặn luồng khi chờ server phản hồi.
    """
    model_name = Config.MODEL_EMBEDDING_API
    url = Config.VNPT_EMBEDDING_URL
    
    # Lấy Credential từ Config
    creds = Config.VNPT_CREDENTIALS.get(model_name)
    if not creds:
        print(f"❌ Config Error: Missing credentials for {model_name}")
        return None

    headers = {
        'Authorization': f'Bearer {Config.VNPT_ACCESS_TOKEN}',
        'Token-id': creds['token_id'],
        'Token-key': creds['token_key'],
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": model_name,
        "input": text,
        "encoding_format": "float"
    }

    for attempt in range(retry_attempts):
        async with RATE_LIMITER: # Đợi slot (không block thread)
            try:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        # Xử lý các format trả về có thể có
                        if 'data' in data and len(data['data']) > 0:
                            return data['data'][0]['embedding']
                        else:
                            return None
                    
                    elif response.status == 429: # Too Many Requests
                        # Backoff nhẹ
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    
                    elif response.status >= 500: # Server Error
                        await asyncio.sleep(1)
                        continue
                    
                    else:
                        # Lỗi 400, 401... (Lỗi client/auth) -> Không retry
                        print(f"❌ API Error {response.status}: {await response.text()}")
                        return None

            except Exception as e:
                # Lỗi mạng, timeout...
                await asyncio.sleep(1)
                
    return None

# --- 3. WORKER LOGIC ---
async def worker(queue, session, client, pbar, failed_log):
    """
    Worker nhận job từ queue -> Gọi API -> Gom Batch -> Upsert Qdrant
    """
    buffer_points = []
    
    while True:
        item = await queue.get()
        if item is None: # Tín hiệu dừng
            break

        # 1. Gọi Embedding API
        try:
            # text để embed: kết hợp title + content
            vector = await get_embedding_async(session, item['vector_text'])
            
            if vector:
                # 2. Tạo Point Struct
                point = models.PointStruct(
                    id=generate_uuid5(item['chunk_id']),
                    vector=vector,
                    payload={
                        "title": item.get('doc_title', ''),
                        "text": item.get('display_text', ''), # Text hiển thị
                        "category": item.get('doc_category', ''),
                        "url": item.get('doc_url', ''),
                        "chunk_id": item['chunk_id']
                    }
                )
                buffer_points.append(point)
            else:
                failed_log.append(item['chunk_id'])
                
        except Exception as e:
            print(f"⚠️ Worker Error item {item.get('chunk_id')}: {e}")
            failed_log.append(item['chunk_id'])
        
        # Update progress bar
        pbar.update(1)

        # 3. Upsert Batch nếu buffer đầy
        if len(buffer_points) >= UPSERT_BATCH_SIZE:
            try:
                # Chạy upsert trong thread khác để không block event loop
                await asyncio.to_thread(
                    client.upsert,
                    collection_name=Config.COLLECTION_NAME,
                    points=buffer_points
                )
                # UPSERT THÀNH CÔNG -> GHI CHECKPOINT NGAY
                # Lấy danh sách chunk_id từ payload để lưu
                processed_ids = [p.payload['chunk_id'] for p in buffer_points]
                save_checkpoint(processed_ids)

            except Exception as e:
                print(f"❌ Qdrant Upsert Error: {e}")
                # Lưu lại id bị lỗi upsert
                failed_log.extend([p.payload['chunk_id'] for p in buffer_points])
            finally:
                buffer_points = [] # Clear buffer

        queue.task_done()

    # 4. Vét nốt buffer còn lại trước khi nghỉ
    if buffer_points:
        try:
            await asyncio.to_thread(
                client.upsert,
                collection_name=Config.COLLECTION_NAME,
                points=buffer_points
            )
            # Ghi nốt checkpoint cuối
            processed_ids = [p.payload['chunk_id'] for p in buffer_points]
            save_checkpoint(processed_ids)

        except Exception as e:
             print(f"❌ Final Upsert Error: {e}")
             failed_log.extend([p.payload['chunk_id'] for p in buffer_points])

# --- 4. MAIN PROCESS ---
async def main():
    # Setup Directories
    Config.setup_dirs()

    # 1. Load Data
    input_file = Config.INDEXING_INPUT_FILE # File delta chunks
    if not input_file.exists():
        print(f"❌ Không tìm thấy file input: {input_file}")
        print("💡 Hãy chạy chunking.py trước.")
        return

    print(f"📂 Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_parquet(input_file)
    
    df['chunk_id'] = df['chunk_id'].astype(str)
    
    # 2. Load Checkpoint
    completed_ids = set()
    if Config.CHECKPOINT_FILE.exists():
        print("🔄 Checking checkpoint file...")
        with open(Config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            # [FIX 1] Ép kiểu dòng đọc được sang String và strip()
            completed_ids = set(str(line.strip()) for line in f if line.strip())
        print(f"📊 Found {len(completed_ids)} completed chunks.")

    # 3. Filter Data
    df_to_process = df[~df['chunk_id'].isin(completed_ids)]
    total_records = len(df_to_process)
    
    if total_records == 0:
        print("🎉 ALL DONE! Everything is indexed.")
        return

    print(f"🔥 Remaining to index: {total_records} chunks")

    # 2. Setup Qdrant
    print(f"🔌 Connecting to Qdrant at {Config.QDRANT_URL}...")
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

    is_resuming = len(completed_ids) > 0
    
    if is_resuming:
        print("⚠️ DETECTED RESUME MODE: Ignoring Config.FORCE_RECREATE.")
        print("   -> Will NOT delete existing collection.")
    else:
        # Chỉ cho phép xóa nếu KHÔNG phải là resume (Start Fresh)
        if Config.FORCE_RECREATE and client.collection_exists(Config.COLLECTION_NAME):
            print(f"🗑️ FRESH START: Deleting collection '{Config.COLLECTION_NAME}'...")
            client.delete_collection(Config.COLLECTION_NAME)
            # Xóa luôn file checkpoint (nếu có rác) để sạch sẽ
            if Config.CHECKPOINT_FILE.exists():
                os.remove(Config.CHECKPOINT_FILE)

    if not client.collection_exists(Config.COLLECTION_NAME):
        print(f"🆕 Creating collection '{Config.COLLECTION_NAME}'...")
        
        # Lấy thử 1 vector để check dimension (hoặc hardcode 1024 nếu biết chắc)
        # Ở đây ta gọi thử 1 request thật để lấy size chuẩn
        async with aiohttp.ClientSession() as temp_session:
            print("🧪 Testing API to get vector size...")
            sample_vec = await get_embedding_async(temp_session, "test dimension")
            if not sample_vec:
                print("❌ Fatal: Không gọi được API Embedding để lấy size. Dừng chương trình.")
                return
            vec_size = len(sample_vec)
            print(f"✅ Vector Size detected: {vec_size}")

        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vec_size, 
                distance=models.Distance.COSINE
            ),
            # Tối ưu HNSW config từ Config
            hnsw_config=models.HnswConfigDiff(
                m=Config.HNSW_M,
                ef_construct=Config.HNSW_EF_CONSTRUCT
            )
        )

    # 3. Setup Queue & Workers
    queue = asyncio.Queue()
    
    # Nạp data vào queue
    for record in records:
        queue.put_nowait(record)
    
    failed_log = []
    
    # Khởi tạo Session
    # Set limit connection pool cao hơn số worker
    conn = aiohttp.TCPConnector(limit=NUM_WORKERS + 10) 
    async with aiohttp.ClientSession(connector=conn) as session:
        
        # Progress Bar
        pbar = tqdm(total=total_records, desc="Indexing", unit="chunk")
        
        # Tạo Workers
        workers = []
        for _ in range(NUM_WORKERS):
            w = asyncio.create_task(worker(queue, session, client, pbar, failed_log))
            workers.append(w)

        # Chờ queue được xử lý hết
        await queue.join()

        # Gửi tín hiệu dừng (None) cho từng worker
        for _ in range(NUM_WORKERS):
            await queue.put(None)
        
        # Chờ tất cả worker tắt hẳn
        await asyncio.gather(*workers)
        
        pbar.close()

    # 4. Kết thúc
    print("\n✅ INDEXING COMPLETED!")
    
    if failed_log:
        print(f"⚠️ Có {len(failed_log)} chunks bị lỗi. Đang lưu log...")
        failed_file = Config.LOGS_DIR / "indexing_failed_ids.json"
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_log, f, ensure_ascii=False, indent=2)
        print(f"📄 Saved failed IDs to {failed_file}")

if __name__ == "__main__":
    # Fix lỗi Event Loop trên Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())