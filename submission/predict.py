import asyncio
import sys
import json
import logging
import pandas as pd
import aiohttp
from pathlib import Path
import random

# Qdrant
from qdrant_client import AsyncQdrantClient

# modules
from config import Config, TIMEOUT_PER_QUESTION, LIMITER_EMBED, LIMITER_LARGE, LIMITER_SMALL
from utils.logger import logger
from core.retriever import HybridRetriever
from core.logic import process_row_logic


Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File lưu kết quả (Dùng để Resume)
OUTPUT_FILE = Config.OUTPUT_FILE
DEBUG_LOG_FILE = Config.DEBUG_LOG_FILE

# Constants
BM25_INDEX_DIR = Config.BM25_INDEX_DIR
BM25_IDS_FILE = Config.BM25_IDS_FILE
BM25_META_FILE = Config.BM25_META_FILE

async def main():
    # 1. Load Data
    # files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    # files = [Config.BASE_DIR / "data" / "test.json"]
    input_file = Config.INPUT_FILE
    if not input_file: 
        logger.error("❌ Input file not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f: data = json.load(f)

    # 2. Check Resume (Đọc file đã lưu để chạy tiếp)
    processed_ids = set()
    if OUTPUT_FILE.exists():
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(df_done['qid'].astype(str))
            logger.info(f"RESUMING... Found {len(processed_ids)} processed questions.")
        except: pass
    
    # Lọc ra những câu chưa làm
    data_to_process = [r for r in data if str(r.get('qid', r.get('id'))) not in processed_ids]
    
    if not data_to_process:
        logger.info("✅ ALL DONE! Nothing to process.")
        return

    logger.info(f"🚀 REMAINING: {len(data_to_process)}/{len(data)} questions")

    # 3. Setup Qdrant & Retriever
    # qdrant_client = AsyncQdrantClient(
    #     url=Config.QDRANT_URL,
    #     api_key=Config.QDRANT_API_KEY,
    #     timeout=30,  # Tăng timeout
    #     # Thêm config pool
    #     grpc_options={
    #         'grpc.max_connection_idle_ms': 60000,  # 60s
    #         'grpc.keepalive_time_ms': 30000,       # 30s
    #         'grpc.http2.max_pings_without_data': 0,
    #     }
    # )

    retriever = HybridRetriever(
        collection_name=Config.COLLECTION_NAME
    )

    stats = {'used_large': 0, 'used_small': 0}
    
    # 4. Run Sequential (Vòng lặp đơn luồng - AN TOÀN NHẤT)
    # limit=1 để đảm bảo chỉ có 1 request tại 1 thời điểm
    sem = asyncio.Semaphore(Config.MAX_CONCURRENT_TASKS) 
    
    # Lock: Đảm bảo khi ghi file không bị tranh chấp
    write_lock = asyncio.Lock()
    
    # Connection Pool lớn hơn số task một chút
    conn = aiohttp.TCPConnector(limit=0) # limit=0 để Semaphore lo việc giới hạn

    # --- WORKER FUNCTION (Chạy song song) ---
    async def worker(session, row):
        async with sem: # Chiếm 1 slot
            qid = str(row.get('qid', row.get('id')))
            
            try:
                # Jitter nhẹ để tránh gửi request đồng loạt đúng 1 thời điểm
                await asyncio.sleep(random.uniform(0.1, 1.5))
                
                # Gọi xử lý chính (đã bao gồm retry bên trong rồi)
                result = await asyncio.wait_for(
                    process_row_logic(session, retriever, row, stats=None),
                    timeout=TIMEOUT_PER_QUESTION
                )
                
                # Chuẩn hóa output (Hàm chốt chặn tôi đã gửi trước đó)
                # final_result = standardize_submission_output(result, row) 
                # (Nếu chưa có hàm trên thì dùng result trực tiếp nhưng rủi ro hơn)
                final_result = result if result else {"qid": qid, "answer": "A"}

                # Ghi file an toàn (Thread-safe write)
                async with write_lock:
                    df_res = pd.DataFrame([final_result])
                    need_header = not OUTPUT_FILE.exists()
                    df_res[['qid', 'answer']].to_csv(OUTPUT_FILE, mode='a', header=need_header, index=False)
                    logger.info(f"💾 Saved Q:{qid}")

            except asyncio.TimeoutError:
                logger.error(f"⏰ TIMEOUT Q:{qid}")
                # Fallback ghi A để không mất bài
                async with write_lock:
                    pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)
            
            except Exception as e:
                logger.error(f"❌ ERROR Q:{qid}: {e}")
                async with write_lock:
                    pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)

    # 5. EXECUTE BATCH
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for row in data_to_process:
            # Tạo task nhưng chưa await ngay -> Nó sẽ chạy nền
            task = asyncio.create_task(worker(session, row))
            tasks.append(task)
        
        # Chờ tất cả xong
        await asyncio.gather(*tasks)

    await retriever.client.close()
    logger.info("🎉 BATCH COMPLETED!")
    
    # 6. Verify Output
    if OUTPUT_FILE.exists():
        logger.info(f"📁 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())