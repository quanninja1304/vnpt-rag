import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import torch
import os
import uuid
from fastembed import SparseTextEmbedding
from config import Config

VIETNAMESE_STOPWORDS = set([
    "và", "của", "là", "những", "các", "tại", "trong", "cho", "được", "với", 
    "có", "để", "người", "này", "khi", "ra", "đã", "đang", "sẽ", "về", "như"
])
def remove_stopwords(text):
    """Xóa stopwords đơn giản cho luồng Sparse"""
    if not isinstance(text, str): return ""
    return " ".join([w for w in text.split() if w.lower() not in VIETNAMESE_STOPWORDS])

def generate_uuid5(unique_string):
    """Tạo UUID cố định dựa trên chuỗi input (Deterministic)"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def main():
    print("="*50)
    print("BẮT ĐẦU INDEXING")
    print("="*50)
    
    # 1. Load Data
    if not os.path.exists(Config.INDEXING_INPUT_FILE):
        print(f"Lỗi: Không tìm thấy {Config.INDEXING_INPUT_FILE}")
        return
    df = pd.read_parquet(Config.INDEXING_INPUT_FILE)
    print(f"Dữ liệu: {len(df):,} chunks")

    # 2. Load Model embedding (dense)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    try:
        # Thêm normalize_embeddings=True vào encode sau này, không cần config ở load
        dense_model = SentenceTransformer(Config.MODEL_PATH, device=device)
        print(f"Loaded vnpt embedding Model: {Config.MODEL_PATH}")
    except:
        print("Fallback to dummy model")
        dense_model = SentenceTransformer(Config.DUMMY_MODEL_NAME, device=device)

    dense_size = dense_model.get_sentence_embedding_dimension()
    print(f"Kích thước Dense Vector: {dense_size}")

    # 2b. load sparse model
    sparse_model = None
    use_sparse = False
    if Config.SPARSE_AVAILABLE:
        print("Đang load SPARSE Model (BM25)...")
        try:
            # Model này hỗ trợ đa ngôn ngữ tốt và nhẹ
            sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            use_sparse = True
            print("   Sparse Model đã sẵn sàng!")
        except Exception as e:
            print(f"   Lỗi load Sparse Model: {e}. Sẽ chạy chế độ Dense-Only.")

    # 3. Connect Qdrant
    try:
        if Config.USE_QDRANT_CLOUD:
            print(f"Kết nối Qdrant Cloud...")
            client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        else:
            print(f"Kết nối Qdrant Local ({Config.QDRANT_URL})...")
            client = QdrantClient(url=Config.QDRANT_URL)
        
        # Test kết nối
        client.get_collections()
    except Exception as e:
        print(f"Lỗi kết nối Qdrant: {e}")
        return
    
    # 4. Create Collection (Smart Config)
    # Kiểm tra xem có cần xóa không
    if client.collection_exists(Config.COLLECTION_NAME):
        if Config.FORCE_RECREATE:
            print(f"Force Recreate: Đang xóa collection cũ...")
            client.delete_collection(Config.COLLECTION_NAME)
        else:
            pass

    # Tạo mới nếu chưa có
    if not client.collection_exists(Config.COLLECTION_NAME):
        print("Đang tạo Collection mới với cấu hình Hybrid (Dense + Sparse)...")

        # Cấu hình Vector (Dictionary cho Multi-vector)
        vectors_config = {
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE,
                # Tối ưu HNSW cho Dense
                hnsw_config=models.HnswConfigDiff(
                    m=32, 
                    ef_construct=200, 
                    on_disk=True # Tiết kiệm RAM
                ),
                # Tối ưu nén cho Dense
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
        }

        # Cấu hình Sparse Vector
        sparse_vectors_config = None
        if use_sparse:
            sparse_vectors_config = {
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True, # Sparse index cũng lưu disk cho nhẹ
                    )
                )
            }
        
        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
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
            
            # Dense vectors 
            dense_embeddings = dense_model.encode(
                texts, 
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=False, 
                normalize_embeddings=True # chuan hoa cosine
            )
            
            # Sparse vectors
            sparse_embeddings = []
            if use_sparse:
                # FastEmbed trả về generator, cần list() để lấy kết quả
                # clean_texts_sparse = [remove_stopwords(t) for t in texts]

                # sparse_embeddings = list(sparse_model.embed(clean_texts_sparse))
                sparse_embeddings = list(sparse_model.embed(texts))

            points = []
            for idx, item in enumerate(batch):
                # Tạo Deterministic ID (Idea A - Fixed)
                # Dùng chunk_id (vd: 0_1, 0_2) để tạo UUID cố định
                # Nếu chạy lại, ID này không đổi -> Qdrant tự update đè -> Không trùng lặp
                point_id = generate_uuid5(str(item['chunk_id']))
                
                # Xây dựng Multi-vector dictionary
                vector_dict = {
                    "dense": dense_embeddings[idx].tolist()
                }
                # Chuyển đổi format Sparse của FastEmbed sang Qdrant
                if use_sparse:
                    vector_dict["sparse"] = models.SparseVector(
                        indices=sparse_embeddings[idx].indices.tolist(),
                        values=sparse_embeddings[idx].values.tolist()
                    )

                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector_dict, # Multi-vector
                    payload={
                        "title": item['doc_title'],
                        "text": item['display_text'], # Chỉ lưu text sạch để hiển thị
                        "url": item['doc_url'],
                        "category": item['doc_category'],
                        "chunk_id": item['chunk_id']
                    }
                ))
            # D. Upsert
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