import asyncio
import sys
import json
import logging
import pandas as pd
import aiohttp
from pathlib import Path

# Qdrant
from qdrant_client import AsyncQdrantClient

# modules
from config import Config, TIMEOUT_PER_QUESTION, LIMITER_EMBED, LIMITER_LARGE, LIMITER_SMALL
from utils.logger import logger
from core.retriever import HybridRetriever
from core.logic import process_row_logic


# ==============================================================================
# 0. CẤU HÌNH CHIẾN THUẬT (Tactical Config)
# ==============================================================================
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File lưu kết quả (Dùng để Resume)
OUTPUT_FILE = Config.OUTPUT_FILE
DEBUG_LOG_FILE = Config.DEBUG_LOG_FILE

# Constants
BM25_FILE = Config.BM25_FILE

async def main():
    # 1. Load Data
    # files = [Config.BASE_DIR / "data" / "val.json", Config.BASE_DIR / "data" / "test.json"]
    files = [Config.BASE_DIR / "data" / "test.json"]
    input_file = next((f for f in files if f.exists()), None)
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
    qdrant_client = AsyncQdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY, timeout=30)
    retriever = HybridRetriever(
        qdrant_client=qdrant_client, 
        collection_name=Config.COLLECTION_NAME
    )
    stats = {'used_large': 0, 'used_small': 0}
    
    # 4. Run Sequential (Vòng lặp đơn luồng - AN TOÀN NHẤT)
    # limit=1 để đảm bảo chỉ có 1 request tại 1 thời điểm
    conn = aiohttp.TCPConnector(limit=1, force_close=True, enable_cleanup_closed=True)
    
    async with aiohttp.ClientSession(connector=conn) as session:
        
        for i, row in enumerate(data_to_process):
            qid = row.get('qid', row.get('id'))
            
            # Retry loop cho từng câu (Thử lại tối đa 3 lần nếu lỗi mạng)
            for attempt in range(3):
                try:
                    # Timeout cứng cho mỗi câu hỏi
                    result = await asyncio.wait_for(
                        process_row_logic(session, retriever, row, stats),
                        timeout=TIMEOUT_PER_QUESTION
                    )
                    
                    # --- GHI FILE NGAY LẬP TỨC (Save Scumming) ---
                    df_res = pd.DataFrame([result])
                    need_header = not OUTPUT_FILE.exists()
                    df_res[['qid', 'answer']].to_csv(OUTPUT_FILE, mode='a', header=need_header, index=False)
                    
                    # Done câu này -> Thoát vòng lặp retry -> Sang câu tiếp theo
                    break 
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout Q:{qid} (Attempt {attempt+1})")
                    # Nếu thử đến lần cuối vẫn timeout -> Điền đáp án 'A' để không bị kẹt mãi
                    if attempt == 2:
                        pd.DataFrame([{"qid": qid, "answer": "A"}]).to_csv(OUTPUT_FILE, mode='a', header=not OUTPUT_FILE.exists(), index=False)
                        
                except Exception as e:
                    logger.error(f"❌ Error Q:{qid}: {e}")
                    await asyncio.sleep(5) # Chờ 5s trước khi thử lại

            # [QUAN TRỌNG] Nghỉ 1 giây giữa các câu hỏi để Server VNPT hồi phục quota
            await asyncio.sleep(1)

    # 5. Cleanup & Stats
    await qdrant_client.close()
    logger.info("🎉 BATCH COMPLETED!")

    # In thống kê (nếu có đáp án mẫu)
    if OUTPUT_FILE.exists():
        print("\n" + "="*40)
        print("TỔNG KẾT TOÀN BỘ (CUMULATIVE STATS)")
        print("="*40)
        try:
            df_results = pd.read_csv(OUTPUT_FILE)
            ground_truth = {
                str(r.get('qid', r.get('id'))): str(r.get('answer')).strip() 
                for r in data if r.get('answer')
            }
            
            if not ground_truth:
                print("⚠️ Tập dữ liệu Test (không có đáp án) -> Không tính điểm.")
            else:
                correct_count = 0
                total_checked = 0
                for _, row in df_results.iterrows():
                    qid = str(row['qid'])
                    pred = str(row['answer']).strip()
                    if qid in ground_truth:
                        total_checked += 1
                        if pred == ground_truth[qid]:
                            correct_count += 1
                
                if total_checked > 0:
                    acc = (correct_count / total_checked) * 100
                    print(f"✅ Đã làm: {total_checked}/{len(ground_truth)} câu")
                    print(f"🎯 Đúng  : {correct_count} câu")
                    print(f"📈 Tỷ lệ : {acc:.2f}%")
        except Exception as e:
            print(f"Lỗi tính điểm: {e}")

        print(f"📁 File kết quả: {OUTPUT_FILE}")

if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

    