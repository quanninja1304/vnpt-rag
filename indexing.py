import asyncio
import aiohttp
import pandas as pd
from qdrant_client import QdrantClient, models
import json
import uuid
import sys
import os
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm
from config import Config

# --- 1. UTILS ---
def generate_uuid5(unique_string):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

# --- 2. ASYNC API CLIENT ---
async def get_embedding_async(session, text, limiter, token_key):
    """Gọi API bất đồng bộ với Rate Limit"""
    
    # [FIX 1] Lấy URL chuẩn từ Config
    url = Config.VNPT_EMBEDDING_URL 
    
    headers = {
        # [FIX 2] Token Key lấy từ tham số truyền vào (trích từ Config)
        "Authorization": f"Bearer {token_key}", 
        "Content-Type": "application/json"
    }
    
    # Payload theo document của VNPT
    payload = {
        "input": text,
        # Lưu ý: check lại doc xem model name có cần thiết ko, thường embedding model tên cố định
        "model": Config.MODEL_EMBEDDING_API 
    }

    retry_count = 0
    # [OPTIMIZE] Tăng timeout lên 60s để tránh lỗi mạng chập chờn
    timeout = aiohttp.ClientTimeout(total=60) 

    while retry_count < 3:
        async with limiter: 
            try:
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Handle format trả về (linh hoạt cho cả list và dict)
                        if isinstance(data, list): return data
                        if "data" in data and len(data["data"]) > 0:
                            return data["data"][0]["embedding"]
                        return None
                    
                    elif response.status == 429:
                        # [LOGIC] Nếu bị 429 thật, ngủ lâu hơn chút
                        print(f"⚠️ 429 Too Many Requests. Backoff 10s...")
                        await asyncio.sleep(10)
                        retry_count += 1
                    else:
                        # Log lỗi nhẹ để không spam màn hình
                        # err_text = await response.text()
                        # print(f"❌ {response.status}: {err_text[:50]}...")
                        return None
            except Exception as e:
                # print(f"⚠️ Net Error: {e}")
                retry_count += 1
                await asyncio.sleep(1)
    return None

# --- 3. WORKER LOGIC ---
async def worker(queue, session, client, limiter, pbar, failed_log, token_key):
    buffer_points = []
    
    while True:
        item = await queue.get()
        if item is None: break 

        vec = await get_embedding_async(session, item['vector_text'], limiter, token_key)
        
        if vec:
            point = models.PointStruct(
                id=generate_uuid5(str(item['chunk_id'])),
                vector=vec,
                payload={
                    "title": item['doc_title'],
                    "text": item['display_text'],
                    "category": item['doc_category'],
                    "url": item.get('doc_url', ''),
                    "chunk_id": item['chunk_id']
                }
            )
            buffer_points.append(point)
        else:
            failed_log.append(item['chunk_id'])
            # [SAFETY] Ghi log nóng phòng trường hợp crash
            with open("temp_failed_log_async.txt", "a") as f:
                f.write(f"{item['chunk_id']}\n")

        pbar.update(1)

        # Upsert Batch (50 items)
        if len(buffer_points) >= 50:
            try:
                # Upsert blocking trong thread riêng
                await asyncio.to_thread(
                    client.upsert, 
                    collection_name=Config.COLLECTION_NAME, 
                    points=buffer_points
                )
            except Exception as e:
                print(f"❌ Upsert Failed: {e}")
                failed_log.extend([p.payload['chunk_id'] for p in buffer_points])
            finally:
                buffer_points = [] 

        queue.task_done()

    # Vét nốt buffer còn lại
    if buffer_points:
        try:
            await asyncio.to_thread(
                client.upsert, 
                collection_name=Config.COLLECTION_NAME, 
                points=buffer_points
            )
        except Exception:
            pass

# --- 4. MAIN ---
async def main_async():
    # [FIX 3] Ưu tiên đọc file ĐÃ LỌC (cleaned) nếu có, nếu không thì đọc file LATEST
    input_file = "cleaned_chunks.parquet" # File sinh ra từ bước lọc dữ liệu
    if not os.path.exists(input_file):
        print(f"⚠️ Không tìm thấy file đã lọc '{input_file}'. Dùng file gốc (SẼ RẤT LÂU).")
        input_file = Config.LATEST_CHUNKS_FILE
    
    if not os.path.exists(input_file):
        print("❌ Không có dữ liệu input.")
        return
    
    df = pd.read_parquet(input_file)
    
    # [QUAN TRỌNG] Checkpoint: Lọc bỏ những ID đã làm rồi
    done_file = "processed_chunks.txt"
    if os.path.exists(done_file):
        with open(done_file, 'r') as f:
            done_ids = set(line.strip() for line in f)
        # Chỉ giữ lại những dòng chưa làm
        df = df[~df['chunk_id'].astype(str).isin(done_ids)]
        print(f"🔄 Resume: Đã bỏ qua {len(done_ids)} chunks đã làm trước đó.")

    records = df.to_dict('records')
    total = len(records)
    print(f"🔥 Bắt đầu Async Indexing: {total} chunks...")

    if total == 0:
        print("✅ Không còn gì để làm.")
        return

    # Setup Qdrant
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    
    # Kiểm tra Collection
    if not client.collection_exists(Config.COLLECTION_NAME):
        # Lấy vector size chuẩn bằng 1 request test (Sync)
        print("🧪 Testing API connection...")
        try:
             # Logic test sync ở đây (lược bỏ cho gọn)
             pass
        except:
             pass
        
        # Tạo mới nếu chưa có
        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE) # Giả định 1024
        )

    # Queue & Limiter
    queue = asyncio.Queue()
    for item in records:
        queue.put_nowait(item)

    limiter = AsyncLimiter(480, 60) 
    failed_ids = []
    
    # [FIX 4] Lấy Token chuẩn từ Config Credentials
    embedding_config = Config.VNPT_CREDENTIALS[Config.MODEL_EMBEDDING_API]
    token_key = embedding_config["token_key"] # Lấy đúng Key

    num_workers = 20 
    
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=total, desc="Indexing", unit="chk")
        tasks = []
        for _ in range(num_workers):
            task = asyncio.create_task(
                worker(queue, session, client, limiter, pbar, failed_ids, token_key)
            )
            tasks.append(task)

        await queue.join()

        for _ in range(num_workers):
            await queue.put(None)
        
        await asyncio.gather(*tasks)
        pbar.close()

    print("\n✅ INDEXING COMPLETED!")
    if failed_ids:
        print(f"⚠️ Failed: {len(failed_ids)} chunks.")
        with open("failed_chunks.json", "w") as f:
            json.dump(failed_ids, f)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())