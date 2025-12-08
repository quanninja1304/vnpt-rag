import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from tqdm import tqdm
import torch
import uuid
import json
import os
from config import Config

def generate_uuid5(unique_string):
    """Tạo UUID cố định từ string (Deterministic)"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def main():
    # 1. Check Input
    if not Config.LATEST_CHUNKS_FILE.exists():
        print(f"❌ File not found: {Config.LATEST_CHUNKS_FILE}")
        return
        
    df = pd.read_parquet(Config.LATEST_CHUNKS_FILE)
    print(f"🔥 Loading {len(df)} chunks to index...")

    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} (Optimization enabled for RTX 3090)")
    
    dense_model = SentenceTransformer(Config.MODEL_PATH, device=device)
    
    sparse_model = None
    if Config.SPARSE_AVAILABLE:
        try:
            # BM25 embedding (FastEmbed)
            sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            print("✅ Sparse Model loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load Sparse model: {e}")
            Config.SPARSE_AVAILABLE = False

    # 3. Connect Qdrant
    client = QdrantClient(
        url=Config.QDRANT_URL, 
        api_key=Config.QDRANT_API_KEY
    )

    if Config.FORCE_RECREATE and client.collection_exists(Config.COLLECTION_NAME):
        client.delete_collection(Config.COLLECTION_NAME)

    if not client.collection_exists(Config.COLLECTION_NAME):
        print("🛠 Creating Collection...")
        client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                    on_disk=True, # Index trên disk cho an toàn
                    # RTX 3090 dư VRAM -> Quantization để always_ram=True để search siêu nhanh
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True 
                        )
                    )
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            } if Config.SPARSE_AVAILABLE else None
        )

    # 4. Indexing Process
    records = df.to_dict('records')
    failed_batches = []
    
    # Batch size 256 from Config
    for i in tqdm(range(0, len(records), Config.BATCH_SIZE), desc="Indexing"):
        batch = records[i : i + Config.BATCH_SIZE]
        texts = [r['vector_text'] for r in batch]
        
        try:
            # Dense Embedding
            dense_vecs = dense_model.encode(texts, batch_size=len(batch), normalize_embeddings=True)
            
            # Sparse Embedding
            sparse_vecs = []
            if Config.SPARSE_AVAILABLE:
                sparse_vecs = list(sparse_model.embed(texts))

            points = []
            for idx, item in enumerate(batch):
                # IDMPOTENCY: Tạo ID từ chunk_id. Chạy lại code sẽ overwrite chứ không tạo trùng lặp.
                point_id = generate_uuid5(str(item['chunk_id']))
                
                # Vectors
                vectors = {"dense": dense_vecs[idx].tolist()}
                if Config.SPARSE_AVAILABLE:
                    vectors["sparse"] = models.SparseVector(
                        indices=sparse_vecs[idx].indices.tolist(),
                        values=sparse_vecs[idx].values.tolist()
                    )

                # Payload
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={
                        "title": item['doc_title'],
                        "url": item['doc_url'],
                        "category": item['doc_category'],
                        "text": item['display_text'], # Chỉ lưu text hiển thị
                        "full_vector_text": item['vector_text'] # Lưu cả text đã inject context (optional)
                    }
                ))

            # Upsert to Qdrant
            client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=points,
                wait=False
            )
            
        except Exception as e:
            print(f"❌ Error batch {i}: {e}")
            failed_batches.append({
                "batch_index": i,
                "error": str(e),
                "chunk_ids": [r['chunk_id'] for r in batch]
            })

    # 5. Final Report
    if failed_batches:
        print(f"⚠️ Completed with {len(failed_batches)} failed batches. See logs.")
        with open(Config.FAILED_BATCHES_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_batches, f, indent=2)
    else:
        print("✅ Indexing SUCCESS (100%)!")

if __name__ == "__main__":
    main()