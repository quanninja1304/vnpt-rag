import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import json
import os
from config import Config

# --- 1. SETUP TOKENIZER ---
print("⏳ Loading Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    def count_tokens(text):
        return len(tokenizer.encode(text))
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Tokenizer error ({e}). Using Vietnamese-optimized fallback.")
    def count_tokens(text):
        # Fallback: Đếm từ (split space) * 1.3
        if not text: return 0
        return int(len(text.split()) * 1.3)

# --- 2. CLEAN TEXT ---
def clean_wiki_text(text):
    if not text: return ""
    
    # Cắt footer
    stop_phrases = ['tham khảo', 'liên kết ngoài', 'chú thích', 'đọc thêm']
    lines = text.split('\n')
    cut_index = len(lines)
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        if len(line_clean) < 40 and any(p == line_clean.strip('.:-=') for p in stop_phrases):
            cut_index = i
            break
    text = '\n'.join(lines[:cut_index])
    
    # Clean artifacts & format
    text = re.sub(r'\[\d+\]', '', text) # Remove citation [1]
    text = re.sub(r'(?<!\n)\n(?!\n)', '. ', text) # Fix broken lines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. MAIN CHUNKING ---
def process_chunking():
    # Load Data
    if not Config.CRAWL_OUTPUT_PARQUET.exists():
        print(f"❌ Missing file: {Config.CRAWL_OUTPUT_PARQUET}")
        return
    
    df = pd.read_parquet(Config.CRAWL_OUTPUT_PARQUET)
    
    # Load State (Incremental check)
    processed_titles = set()
    if Config.CHUNKING_STATE_FILE.exists():
        with open(Config.CHUNKING_STATE_FILE, 'r', encoding='utf-8') as f:
            processed_titles = set(json.load(f))
            
    df_new = df[~df['title'].isin(processed_titles)]
    print(f"📦 Total Articles: {len(df)} | 🔄 New to Process: {len(df_new)}")
    
    if len(df_new) == 0:
        print("✅ No new articles.")
        return

    # Splitter config
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE_TOKENS,
        chunk_overlap=Config.CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", ".", ";", " ", ""]
    )

    new_chunks = []
    
    for idx, row in tqdm(df_new.iterrows(), total=len(df_new)):
        text = clean_wiki_text(row.get('text', ''))
        title = row.get('title', '')
        url = row.get('url', '')
        
        # Lấy category đầu tiên làm metadata
        cats = row.get('categories', [])
        cat_str = cats[0] if isinstance(cats, list) and cats else "Tổng hợp"
        cat_str = cat_str.replace('_', ' ')

        # Filter bài quá ngắn
        if len(text) < 50: 
            processed_titles.add(title)
            continue

        chunks = splitter.create_documents([text])
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            
            # --- FILTERS (Relaxed) ---
            # Giữ lại các chunk "Niên biểu", "Sự kiện" kể cả khi ngắn
            is_timeline = any(kw in content for kw in ["Niên biểu", "Sự kiện", "năm"])
            has_number = any(char.isdigit() for char in content)
            
            if len(content) < 30 and not (is_timeline and has_number):
                continue
                
            if "Mục lục" in content and len(content) < 50:
                continue

            # --- CONTEXT INJECTION ---
            # Thêm Title và Category vào đầu đoạn văn để model hiểu ngữ cảnh
            vector_text = f"Lĩnh vực: {cat_str}. Chủ đề: {title}.\nNội dung: {content}"
            
            new_chunks.append({
                "chunk_id": f"{title}_{i}",
                "doc_title": title,
                "doc_category": cat_str,
                "vector_text": vector_text, # Dùng để Embed
                "display_text": content,    # Dùng để hiển thị
                "doc_url": url
            })
        
        processed_titles.add(title)

    # Save
    if new_chunks:
        df_delta = pd.DataFrame(new_chunks)
        df_delta.to_parquet(Config.LATEST_CHUNKS_FILE, index=False)
        print(f"💾 Saved {len(df_delta)} chunks to {Config.LATEST_CHUNKS_FILE}")
        
        # Update State
        with open(Config.CHUNKING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_titles), f, ensure_ascii=False)
    else:
        print("⚠️ Processed articles but generated no chunks.")

if __name__ == "__main__":
    process_chunking()