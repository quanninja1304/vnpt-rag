import pandas as pd
import re
import json
import os
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Config từ file config của bạn
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING & STRUCTURE PRESERVATION (Core Logic từ chunking_stem)
# ════════════════════════════════════════════════════════════════════════════

def clean_wiki_text_master(text: str) -> str:
    """
    Làm sạch văn bản nhưng giữ nguyên cấu trúc Bảng, List, Công thức.
    """
    if not text: return ""
    
    # 1. Footer Detection (Cắt bỏ phần tham khảo rác)
    stop_phrases = ['tham khảo', 'liên kết ngoài', 'chú thích', 'đọc thêm', 
                    'nguồn', 'xem thêm', 'thư mục', 'bài liên quan']
    
    lines = text.split('\n')
    cut_index = len(lines)
    
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        # Header ngắn + Chứa từ khóa dừng = Footer
        is_heading = (line_clean.startswith('=') or line_clean.startswith('#') or line == line.upper())
        
        # Bỏ các ký tự trang trí để check
        line_core = line_clean.strip('=#-=:. ')
        
        if len(line_clean) < 50 and is_heading and line_core in stop_phrases:
            cut_index = i
            break
    
    content_lines = lines[:cut_index]
    
    # 2. Structure Detection (State Machine)
    processed_blocks = []
    current_block = []
    in_structure = None  # None | 'table' | 'list' | 'formula'
    
    for line in content_lines:
        line = line.strip()
        if not line:
            if in_structure: current_block.append("")
            continue
        
        # Xóa citation [1], [2]
        line = re.sub(r'\[(?:\d+|cần dẫn nguồn|citation needed)\]', '', line)
        
        # Detect types
        is_table = line.startswith('|')
        is_list = bool(re.match(r'^[\-\*•]\s+|^\d+\.\s+', line))
        is_formula = bool(re.search(r'\$|\\[a-z]+\{|[∑∫√±≠≤≥]', line))
        
        # State Machine Logic
        if is_table:
            if in_structure != 'table':
                if current_block: processed_blocks.append(_join_block(current_block, in_structure))
                current_block = []
                in_structure = 'table'
            current_block.append(line)
        elif is_list:
            if in_structure != 'list':
                if current_block: processed_blocks.append(_join_block(current_block, in_structure))
                current_block = []
                in_structure = 'list'
            current_block.append(line)
        elif is_formula:
            if in_structure != 'formula':
                if current_block: processed_blocks.append(_join_block(current_block, in_structure))
                current_block = []
                in_structure = 'formula'
            current_block.append(line)
        else:
            # Regular text
            if in_structure:
                if current_block: processed_blocks.append(_join_block(current_block, in_structure))
                current_block = []
                in_structure = None
            current_block.append(line)
    
    if current_block:
        processed_blocks.append(_join_block(current_block, in_structure))
    
    return '\n\n'.join(processed_blocks)

def _join_block(lines: List[str], structure_type: str) -> str:
    # Cấu trúc đặc biệt thì giữ nguyên xuống dòng
    if structure_type in ['table', 'list', 'formula']:
        return '\n'.join(lines)
    # Văn bản thường thì nối lại thành đoạn văn bằng \n (để splitter dễ cắt hơn là space)
    return '\n'.join(lines)

# ════════════════════════════════════════════════════════════════════════════
# 2. DOMAIN-SPECIFIC SPLITTERS (Kết hợp cả 2 logic)
# ════════════════════════════════════════════════════════════════════════════

def get_domain_splitter(domain: str = "general") -> RecursiveCharacterTextSplitter:
    """Tạo Splitter tối ưu cho từng lĩnh vực"""
    chunk_size = 1024
    chunk_overlap = 200
    
    if domain == "legal":
        # Ưu tiên cấu trúc Luật
        separators = ["\n\nĐiều ", "\n\nKhoản ", "\n\nChương ", "\n\n", "\n", "; ", ". ", " "]
    elif domain == "stem":
        # Ưu tiên cấu trúc Toán/Lý
        separators = ["\n\n", "\n### ", "\n- ", "\n1. ", "\n", "; ", ". ", " "]
    else: 
        # General (Wiki thường): Logic của chunking_wiki.py nhưng tối ưu hơn
        separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len, # Dùng độ dài ký tự chuẩn (tốt hơn count_tokens cho regex)
        strip_whitespace=True
    )

# ════════════════════════════════════════════════════════════════════════════
# 3. CONTENT DETECTION & FILTERING
# ════════════════════════════════════════════════════════════════════════════

def detect_content_type(text: str) -> Dict[str, bool]:
    result = {'has_math': False, 'has_legal': False, 'is_substantial': False}
    
    # Math Detection
    math_pattern = r'[+\-*/=<>^]{1,2}\s*\d|\d\s*[+\-*/=<>^]|\d+/\d+|√\d+|\$|\\frac'
    result['has_math'] = bool(re.search(math_pattern, text))
    
    # Legal Detection
    legal_pattern = r'(?:Điều|Khoản|Chương)\s+\d+|Bộ\s+luật|Nghị\s+định'
    result['has_legal'] = bool(re.search(legal_pattern, text, re.IGNORECASE))
    
    # Substantial (Độ dài nội dung thực)
    result['is_substantial'] = len(re.findall(r'\w+', text)) >= 15
    return result

def should_keep_chunk(content: str) -> bool:
    # Nếu dài > 50 ký tự -> Giữ
    if len(content) >= 50: return True
    
    # Nếu ngắn nhưng là công thức hoặc điều luật -> Giữ
    info = detect_content_type(content)
    if info['has_math'] or info['has_legal']: return True
    
    # Giữ lại các mốc thời gian (Logic của chunking_wiki.py)
    is_timeline = any(kw in content for kw in ["Niên biểu", "Sự kiện", "năm"])
    has_number = any(char.isdigit() for char in content)
    if is_timeline and has_number: return True
    
    return False

# ════════════════════════════════════════════════════════════════════════════
# 4. METADATA ENRICHMENT & ROUTING
# ════════════════════════════════════════════════════════════════════════════

def auto_detect_domain(category: str) -> str:
    """Tự động xác định loại bài viết dựa trên Category"""
    cat_lower = str(category).lower()
    
    stem_keywords = ['toán', 'lý', 'hóa', 'sinh', 'tin', 'công nghệ', 'kỹ thuật', 'khoa học']
    if any(kw in cat_lower for kw in stem_keywords):
        return 'stem'
        
    legal_keywords = ['luật', 'nghị định', 'pháp luật', 'hiến pháp', 'thông tư']
    if any(kw in cat_lower for kw in legal_keywords):
        return 'legal'
        
    return 'general'

def shorten_category(category: str) -> str:
    # Rút gọn category để tiết kiệm token embedding
    parts = [p.strip() for p in str(category).split('>')] if category else ["Tổng hợp"]
    if len(parts) > 1:
        return f"{parts[0].split('_')[0]}-{parts[-1]}".replace('_', ' ')
    return parts[0].replace('_', ' ')

def create_enriched_chunk(content, title, category, idx, url, domain):
    cat_short = shorten_category(category)
    
    # --- CONTEXT INJECTION ---
    # Format chuẩn cho cả STEM và Wiki thường: [Category] Title \n Content
    # Đây là format tối ưu nhất cho Vector Search
    vector_text = f"[{cat_short}] {title}\n{content}"
    
    info = detect_content_type(content)
    
    return {
        "chunk_id": f"{title}_{idx}",
        "doc_title": title,
        "doc_category": category,
        "vector_text": vector_text,   # Dùng để Embed
        "display_text": content,      # Dùng để hiển thị cho LLM
        "doc_url": url,
        "metadata": {
            "has_math": info['has_math'],
            "has_legal": info['has_legal'],
            "domain": domain,
            "chunk_index": idx
        }
    }

# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN PROCESS
# ════════════════════════════════════════════════════════════════════════════

def process_chunking():
    # 1. Load Data
    if not Config.CRAWL_OUTPUT_PARQUET.exists():
        print(f"❌ Missing file: {Config.CRAWL_OUTPUT_PARQUET}")
        return
    
    print("⏳ Loading Parquet Data...")
    df = pd.read_parquet(Config.CRAWL_OUTPUT_PARQUET)
    
    # 2. Load State (Incremental Processing)
    processed_titles = set()
    if Config.CHUNKING_STATE_FILE.exists():
        try:
            with open(Config.CHUNKING_STATE_FILE, 'r', encoding='utf-8') as f:
                processed_titles = set(json.load(f))
        except: pass
            
    df_new = df[~df['title'].isin(processed_titles)]
    print(f"📦 Total: {len(df)} | 🔄 New: {len(df_new)}")
    
    if len(df_new) == 0:
        print("✅ No new articles to process.")
        return

    all_chunks = []
    
    print("🚀 Starting Master Chunking Pipeline...")
    
    for idx, row in tqdm(df_new.iterrows(), total=len(df_new)):
        try:
            raw_text = row.get('text', '')
            title = row.get('title', 'Unknown')
            url = row.get('url', '')
            
            # Handle Category
            cats = row.get('categories', [])
            if isinstance(cats, list):
                cat_full = " > ".join(cats)
            else:
                cat_full = str(cats)
            
            # A. Detect Domain (Quyết định cách xử lý)
            domain = auto_detect_domain(cat_full)
            
            # B. Clean Text (Dùng bản xịn nhất)
            cleaned_text = clean_wiki_text_master(raw_text)
            if len(cleaned_text) < 50:
                processed_titles.add(title)
                continue
            
            # C. Split (Dùng splitter tương ứng với Domain)
            splitter = get_domain_splitter(domain)
            
            # Thêm title vào đầu để chunk 0 luôn có ngữ cảnh
            text_with_header = f"# {title}\n\n{cleaned_text}" 
            raw_chunks = splitter.create_documents([text_with_header])
            
            # D. Enrich & Filter
            for i, chunk in enumerate(raw_chunks):
                content = chunk.page_content.strip()
                if not should_keep_chunk(content): continue
                
                chunk_data = create_enriched_chunk(content, title, cat_full, i, url, domain)
                all_chunks.append(chunk_data)
            
            processed_titles.add(title)

        except Exception as e:
            logger.error(f"Error processing {title}: {e}")
            continue

    # 3. Save Results
    if all_chunks:
        # Append mode logic
        if Config.LATEST_CHUNKS_FILE.exists():
            try:
                df_old = pd.read_parquet(Config.LATEST_CHUNKS_FILE)
                df_new_chunks = pd.DataFrame(all_chunks)
                # Đảm bảo cột khớp nhau
                df_final = pd.concat([df_old, df_new_chunks], ignore_index=True)
            except:
                df_final = pd.DataFrame(all_chunks)
        else:
            df_final = pd.DataFrame(all_chunks)
            
        # Ensure string types
        for col in ['chunk_id', 'vector_text', 'display_text', 'doc_title']:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(str)

        df_final.to_parquet(Config.LATEST_CHUNKS_FILE, index=False)
        print(f"💾 Saved total {len(df_final)} chunks to {Config.LATEST_CHUNKS_FILE}")
        
        with open(Config.CHUNKING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_titles), f, ensure_ascii=False)
            
    print("✅ Chunking pipeline finished.")

if __name__ == "__main__":
    process_chunking()