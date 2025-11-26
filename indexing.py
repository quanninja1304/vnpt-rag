"""
RAG Indexing Pipeline
================================================
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import torch
import os
import uuid
from config import Config

def generate_uuid5(unique_string):
    """Tạo UUID cố định dựa trên chuỗi input (Deterministic)"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def main():
    print("="*50)
    print("BẮT ĐẦU INDEXING (PRO VERSION)")
    print("="*50)
    
    # 1. Load Data
    if not os.path.exists(Config.INPUT_FILE):
        print(f"Lỗi: Không tìm thấy {Config.INPUT_FILE}")
        return
    df = pd.read_parquet(Config.INPUT_FILE)
    print(f"Dữ liệu: {len(df):,} chunks")

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    try:
        # Thêm normalize_embeddings=True vào encode sau này, không cần config ở load
        model = SentenceTransformer(Config.MODEL_PATH, device=device)
        print(f"Loaded Local Model: {Config.MODEL_PATH}")
    except:
        print("Fallback to dummy model")
        model = SentenceTransformer(Config.DUMMY_MODEL_NAME, device=device)

    vector_size = model.get_sentence_embedding_dimension()

    # 3. Connect Qdrant
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    
    # 4. Create Collection (Smart Config)
    # Kiểm tra xem có cần xóa không
    if client.collection_exists(Config.COLLECTION_NAME):
        if Config.FORCE_RECREATE:
            print(f"Force Recreate: Đang xóa collection cũ...")
            client.delete_collection(Config.COLLECTION_NAME)
        else:
            # Check vector size (Idea C của bạn)
            info = client.get_collection(Config.COLLECTION_NAME)
            if info.config.params.vectors.size != vector_size:
                print(f"Vector size thay đổi ({info.config.params.vectors.size} -> {vector_size}). Xóa cũ tạo mới!")
                client.delete_collection(Config.COLLECTION_NAME)
            else:
                print("Collection khớp config. Sẽ nạp nối tiếp (Append).")

    # Tạo mới nếu chưa có
    if not client.collection_exists(Config.COLLECTION_NAME):
        print("Đang cấu hình Collection (HNSW + Quantization)...")
        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            ),
            # Tối ưu HNSW (Idea D)
            hnsw_config=models.HnswConfigDiff(
                m=32,               # Tăng liên kết để tìm chính xác hơn
                ef_construct=200,   # Xây graph kỹ hơn (tốn time index bù lại search ngon)
                full_scan_threshold=10000,
                on_disk=True        # Tiết kiệm RAM cho máy yếu
            ),
            # Tối ưu Quantization (Idea F - Siêu quan trọng)
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True # Vector nén giữ trên RAM -> Search siêu nhanh
                )
            )
        )

    # 5. Indexing Loop
    print(f"\nBắt đầu nạp {len(df)} chunks...")
    records = df.to_dict('records')
    
    # Progress bar chuẩn (Idea F)
    for i in tqdm(range(0, len(records), Config.BATCH_SIZE), unit="batch"):
        batch = records[i : i + Config.BATCH_SIZE]
        if not batch: break
        
        try:
            # Lấy text
            texts = [item['vector_text'] for item in batch]
            
            # Encode + Normalize (Idea E)
            embeddings = model.encode(
                texts, 
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=False, 
                normalize_embeddings=True # Quan trọng cho Cosine
            )
            
            points = []
            for idx, item in enumerate(batch):
                # Tạo Deterministic ID (Idea A - Fixed)
                # Dùng chunk_id (vd: 0_1, 0_2) để tạo UUID cố định
                # Nếu chạy lại, ID này không đổi -> Qdrant tự update đè -> Không trùng lặp
                point_id = generate_uuid5(str(item['chunk_id']))
                
                points.append(models.PointStruct(
                    id=point_id, 
                    vector=embeddings[idx].tolist(),
                    payload={
                        "title": item['doc_title'],
                        "text": item['display_text'],
                        "url": item['doc_url'],
                        "category": item['doc_category'],
                        "chunk_id": item['chunk_id']
                    }
                ))
            
            client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=points,
                wait=False
            )
            
        except Exception as e:
            print(f"Lỗi batch {i}: {e}")

    # 6. Final Check
    info = client.get_collection(Config.COLLECTION_NAME)
    print("\n" + "="*50)
    print(f"HOÀN TẤT! Đã index {info.points_count:,} points.")
    print(f"Trạng thái: {info.status}")
    print("="*50)

if __name__ == "__main__":
    main()