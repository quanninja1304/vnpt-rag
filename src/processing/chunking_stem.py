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
# 1. TEXT CLEANING - STRUCTURE PRESERVATION
# ════════════════════════════════════════════════════════════════════════════

def clean_wiki_text_production(text: str) -> str:
    """
    Clean Wikipedia text while preserving tables, lists, and formulas
    
    Key improvements:
    - Detect and preserve table structure (|...| format)
    - Detect and preserve lists (-, *, •, 1., 2., etc.)
    - Smart footer detection (must be heading-style)
    - Remove citations without breaking text
    """
    
    if not text:
        return ""
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: Footer Detection (Improved)
    # ────────────────────────────────────────────────────────────────────────
    stop_phrases = ['tham khảo', 'liên kết ngoài', 'chú thích', 'đọc thêm', 
                    'nguồn', 'xem thêm', 'thư mục', 'bài liên quan']
    
    lines = text.split('\n')
    cut_index = len(lines)
    
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        
        # Must be: (1) Short, (2) Heading-style, (3) Match stop phrase
        is_short = len(line_clean) < 50
        is_heading = (line_clean.startswith('=') or 
                     line_clean.startswith('#') or
                     line == line.upper())  # All caps
        
        # Remove heading markers for comparison
        line_core = line_clean.strip('=#-=:.')
        
        if is_short and is_heading and line_core in stop_phrases:
            cut_index = i
            break
    
    content_lines = lines[:cut_index]
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: Structure Detection & Preservation
    # ────────────────────────────────────────────────────────────────────────
    processed_blocks = []
    current_block = []
    in_structure = None  # None | 'table' | 'list' | 'formula'
    
    for line in content_lines:
        original_line = line
        line = line.strip()
        
        # Skip completely empty lines outside structures
        if not line:
            if in_structure:
                current_block.append("")  # Keep empty lines in structures
            continue
        
        # Remove citations: [1], [2], [cần dẫn nguồn]
        line = re.sub(r'\[(?:\d+|cần dẫn nguồn|citation needed)\]', '', line)
        
        # ────────────────────────────────────────────────────────────────────
        # Detect Structure Type
        # ────────────────────────────────────────────────────────────────────
        
        # TABLE: Lines starting with |
        is_table_line = line.startswith('|')
        
        # LIST: Lines starting with -, *, •, or numbered (1., 2., etc.)
        is_list_line = bool(re.match(r'^[\-\*•]\s+|^\d+\.\s+', line))
        
        # FORMULA: Lines with LaTeX or math symbols
        is_formula_line = bool(re.search(r'\$|\\[a-z]+\{|[∑∫√±≠≤≥]', line))
        
        # ────────────────────────────────────────────────────────────────────
        # State Machine: Manage Structure Blocks
        # ────────────────────────────────────────────────────────────────────
        
        if is_table_line:
            if in_structure != 'table':
                # Flush previous block
                if current_block:
                    processed_blocks.append(_join_block(current_block, in_structure))
                    current_block = []
                in_structure = 'table'
            current_block.append(line)
        
        elif is_list_line:
            if in_structure != 'list':
                if current_block:
                    processed_blocks.append(_join_block(current_block, in_structure))
                    current_block = []
                in_structure = 'list'
            current_block.append(line)
        
        elif is_formula_line:
            if in_structure != 'formula':
                if current_block:
                    processed_blocks.append(_join_block(current_block, in_structure))
                    current_block = []
                in_structure = 'formula'
            current_block.append(line)
        
        else:
            # Regular text
            if in_structure:
                # End of structure
                if current_block:
                    processed_blocks.append(_join_block(current_block, in_structure))
                    current_block = []
                in_structure = None
            current_block.append(line)
    
    # Flush final block
    if current_block:
        processed_blocks.append(_join_block(current_block, in_structure))
    
    # Join all blocks with double newline
    return '\n\n'.join(processed_blocks)


def _join_block(lines: List[str], structure_type: str) -> str:
    """Helper: Join lines based on structure type"""
    if structure_type in ['table', 'list', 'formula']:
        # Preserve exact formatting for structures
        return '\n'.join(lines)
    else:
        # Regular text: merge into paragraphs
        return '\n\n'.join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 2. SPLITTER CONFIGURATION - DOMAIN-SPECIFIC
# ════════════════════════════════════════════════════════════════════════════

def get_domain_splitter(domain: str = "general") -> RecursiveCharacterTextSplitter:
    """
    Get text splitter optimized for specific domain
    
    Args:
        domain: 'general' | 'legal' | 'stem' | 'technical'
    
    Returns:
        RecursiveCharacterTextSplitter with domain-specific separators
    """
    
    # Base configuration
    chunk_size = 1024  # Characters, not tokens
    chunk_overlap = 200
    
    # ────────────────────────────────────────────────────────────────────────
    # Domain-Specific Separator Hierarchies
    # ────────────────────────────────────────────────────────────────────────
    
    if domain == "legal":
        # Legal documents: Prioritize Điều, Khoản, Chương
        separators = [
            "\n\nĐiều ",      # Between articles
            "\n\nKhoản ",     # Between clauses
            "\n\nChương ",    # Between chapters
            "\n\n",           # Paragraphs
            "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ",  # Numbered items
            "\n",
            ". ",
            "; ",  # Important in legal text
            ", ",
            " ",
            ""
        ]
    
    elif domain == "stem":
        # STEM: Prioritize formulas, lists, equations
        separators = [
            "\n\n",
            "\n### ",         # Math section headers
            "\n- ", "\n• ", "\n* ",  # Bullet lists
            "\n1. ", "\n2. ", "\n3. ",  # Numbered steps
            "\n",
            "; ",             # Common in definitions
            ". ",
            ", ",
            " ",
            ""
        ]
    
    elif domain == "technical":
        # Technical docs: Code blocks, procedures
        separators = [
            "\n```",          # Code blocks
            "\n\n",
            "\n## ",          # Section headers
            "\n- ",
            "\n",
            "; ",
            ". ",
            " ",
            ""
        ]
    
    else:  # general
        separators = [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        # CRITICAL: Use default length function (character count)
        # NOT token count, as it breaks chunk size logic
        length_function=len,
        strip_whitespace=True,
        keep_separator=False
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. CONTENT DETECTION - ACCURATE STEM/LEGAL IDENTIFICATION
# ════════════════════════════════════════════════════════════════════════════

def detect_content_type(text: str) -> Dict[str, bool]:
    """
    Accurately detect if text contains STEM or legal content
    
    Returns:
        {
            'has_math': bool,
            'has_formula': bool,
            'has_table': bool,
            'has_list': bool,
            'has_legal': bool,
            'is_substantial': bool  # Not just metadata
        }
    """
    
    result = {
        'has_math': False,
        'has_formula': False,
        'has_table': False,
        'has_list': False,
        'has_legal': False,
        'is_substantial': False
    }
    
    # ────────────────────────────────────────────────────────────────────────
    # Math Detection (Improved)
    # ────────────────────────────────────────────────────────────────────────
    
    # Pattern 1: Math operations WITH numbers (not just text)
    math_ops_pattern = r'[+\-*/=<>^]{1,2}\s*\d|\d\s*[+\-*/=<>^]'
    
    # Pattern 2: Fractions, powers, roots
    math_notation = r'\d+/\d+|\d+\^\d+|√\d+|\d+²|\d+³'
    
    result['has_math'] = bool(
        re.search(math_ops_pattern, text) or
        re.search(math_notation, text)
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Formula Detection
    # ────────────────────────────────────────────────────────────────────────
    
    # LaTeX symbols
    latex_pattern = r'\$|\\(?:frac|int|sum|sqrt|alpha|beta|gamma|pi|infty)\{?'
    
    # Unicode math symbols
    unicode_math = r'[∑∫√±≠≤≥×÷∞αβγπ]'
    
    # Common formulas (e.g., "E = mc²", "F = ma")
    formula_pattern = r'\b[A-Z]\s*=\s*[A-Za-z0-9²³√]'
    
    result['has_formula'] = bool(
        re.search(latex_pattern, text) or
        re.search(unicode_math, text) or
        re.search(formula_pattern, text)
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Structure Detection
    # ────────────────────────────────────────────────────────────────────────
    
    # Tables (at least 2 rows with | separators)
    table_lines = [line for line in text.split('\n') if line.strip().startswith('|')]
    result['has_table'] = len(table_lines) >= 2
    
    # Lists (at least 2 items)
    list_pattern = r'^[\-\*•]\s+.+|^\d+\.\s+.+'
    list_matches = re.findall(list_pattern, text, re.MULTILINE)
    result['has_list'] = len(list_matches) >= 2
    
    # ────────────────────────────────────────────────────────────────────────
    # Legal Content Detection
    # ────────────────────────────────────────────────────────────────────────
    
    legal_patterns = [
        r'Điều\s+\d+',           # Điều 123
        r'Khoản\s+\d+',          # Khoản 1
        r'Chương\s+[IVX]+',      # Chương I, II, III
        r'Bộ\s+luật\s+\w+',      # Bộ luật Hình sự
        r'Nghị\s+định\s+\d+',    # Nghị định 123
        r'Thông\s+tư\s+\d+',     # Thông tư 456
    ]
    
    result['has_legal'] = any(
        re.search(pattern, text, re.IGNORECASE) 
        for pattern in legal_patterns
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # Substantiality Check
    # ────────────────────────────────────────────────────────────────────────
    
    # Must have enough actual content (not just headers/metadata)
    word_count = len(re.findall(r'\w+', text))
    result['is_substantial'] = word_count >= 20
    
    return result


def should_keep_chunk(content: str, min_length: int = 50) -> bool:
    """
    Decide if a chunk should be kept based on content quality
    
    Args:
        content: Chunk text
        min_length: Minimum character length
    
    Returns:
        True if chunk should be kept
    """
    
    if len(content) >= min_length:
        return True  # Long enough by default
    
    # Short chunks only kept if they have valuable content
    detection = detect_content_type(content)
    
    # Keep if has: formula, table, or legal content
    # Even if short, these are often self-contained units
    return (
        detection['has_formula'] or
        detection['has_table'] or
        detection['has_legal']
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. METADATA ENRICHMENT - LIGHTWEIGHT STRATEGY
# ════════════════════════════════════════════════════════════════════════════

def shorten_category(category: str) -> str:
    """
    Convert long category path to short tag
    
    Examples:
        "Toán_học > Đại_số > Phương_trình" → "Toán-Phương trình"
        "Vật_lý > Cơ_học > Động_lực_học" → "Lý-Cơ học"
    """
    
    parts = [p.strip() for p in category.split('>')]
    
    if len(parts) == 1:
        # Single category
        return parts[0].replace('_', ' ')
    
    # Take first and last, remove underscores
    first = parts[0].replace('_', ' ')
    last = parts[-1].replace('_', ' ')
    
    # Abbreviate first part for common subjects
    abbrev_map = {
        'Toán học': 'Toán',
        'Vật lý': 'Lý',
        'Hóa học': 'Hóa',
        'Sinh học': 'Sinh',
        'Kinh tế học': 'Kinh tế',
        'Tin học': 'Tin',
    }
    
    first_short = abbrev_map.get(first, first)
    
    return f"{first_short}-{last}"


def create_enriched_chunk(
    content: str,
    title: str,
    category: str,
    chunk_index: int,
    url: str = ""
) -> Dict[str, any]:
    """
    Create final chunk with metadata
    
    Strategy:
    - vector_text: Lightweight prefix for embedding (saves tokens)
    - display_text: Clean content for LLM to read
    - metadata: Full context stored separately
    """
    
    # Shorten category for embedding
    cat_short = shorten_category(category)
    
    # Lightweight prefix (saves ~20 tokens vs full path)
    vector_text = f"[{cat_short}] {content}"
    
    # Detect content type for filtering
    content_info = detect_content_type(content)
    
    return {
        "chunk_id": f"{title}_{chunk_index}",
        "vector_text": vector_text,      # FOR EMBEDDING
        "display_text": content,          # FOR LLM CONTEXT
        "metadata": {
            "doc_title": title,
            "category_full": category,
            "category_short": cat_short,
            "doc_url": url,
            "chunk_index": chunk_index,
            "has_math": content_info['has_math'],
            "has_formula": content_info['has_formula'],
            "has_table": content_info['has_table'],
            "has_legal": content_info['has_legal'],
            "char_count": len(content),
            "word_count": len(re.findall(r'\w+', content))
        }
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN CHUNKING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def chunk_wikipedia_article(
    text: str,
    title: str,
    category: str,
    url: str = "",
    domain: str = "general"
) -> List[Dict[str, any]]:
    """
    Complete pipeline to chunk a Wikipedia article
    
    Args:
        text: Raw Wikipedia text
        title: Article title
        category: Category path (e.g., "Toán_học > Đại_số")
        url: Source URL
        domain: 'general' | 'legal' | 'stem' | 'technical'
    
    Returns:
        List of enriched chunks ready for indexing
    """
    
    # Step 1: Clean text
    cleaned_text = clean_wiki_text_production(text)
    
    if not cleaned_text or len(cleaned_text) < 100:
        return []  # Article too short or empty
    
    # Step 2: Split into chunks
    splitter = get_domain_splitter(domain)
    raw_chunks = splitter.create_documents([cleaned_text])
    
    # Step 3: Filter and enrich
    enriched_chunks = []
    
    for i, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        
        # Quality filter
        if not should_keep_chunk(content):
            continue
        
        # Create enriched chunk
        enriched = create_enriched_chunk(
            content=content,
            title=title,
            category=category,
            chunk_index=i,
            url=url
        )
        
        enriched_chunks.append(enriched)
    
    return enriched_chunks


# ════════════════════════════════════════════════════════════════════════════
# 6. BATCH PROCESSING UTILITY
# ════════════════════════════════════════════════════════════════════════════

def process_wikipedia_batch(
    articles: List[Dict[str, str]],
    domain_detector=None
) -> List[Dict[str, any]]:
    """
    Process multiple Wikipedia articles in batch
    
    Args:
        articles: List of dicts with keys: 'text', 'title', 'category', 'url'
        domain_detector: Optional function to auto-detect domain from category
    
    Returns:
        List of all chunks from all articles
    """
    
    all_chunks = []
    
    for article in articles:
        # Auto-detect domain if function provided
        if domain_detector:
            domain = domain_detector(article.get('category', ''))
        else:
            domain = 'general'
        
        # Chunk article
        chunks = chunk_wikipedia_article(
            text=article['text'],
            title=article['title'],
            category=article.get('category', 'Uncategorized'),
            url=article.get('url', ''),
            domain=domain
        )
        
        all_chunks.extend(chunks)
    
    return all_chunks


def auto_detect_domain(category: str) -> str:
    """
    Auto-detect domain from category string
    
    Returns: 'stem' | 'legal' | 'general'
    """
    
    cat_lower = category.lower()
    
    # STEM keywords
    stem_keywords = [
        'toán', 'vật_lý', 'hóa_học', 'sinh_học', 'tin_học',
        'kỹ_thuật', 'công_thức', 'phương_trình', 'định_lý'
    ]
    
    # Legal keywords
    legal_keywords = [
        'luật', 'pháp', 'điều', 'nghị_định', 'thông_tư',
        'bộ_luật', 'hiến_pháp', 'quy_định'
    ]
    
    if any(kw in cat_lower for kw in stem_keywords):
        return 'stem'
    elif any(kw in cat_lower for kw in legal_keywords):
        return 'legal'
    else:
        return 'general'

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
    
    print("🚀 Starting Production Chunking Pipeline...")
    for idx, row in tqdm(df_new.iterrows(), total=len(df_new)):
        raw_text = row.get('text', '')
        title = row.get('title', 'Unknown')
        url = row.get('url', '')
        
        # Handle Category (List or String)
        cats = row.get('categories', [])
        if isinstance(cats, list):
            cat_full = " > ".join(cats)
        else:
            cat_full = str(cats)
            
        # A. Detect Domain
        domain = auto_detect_domain(cat_full)
        
        # B. Clean Text
        cleaned_text = clean_wiki_text_production(raw_text)
        if len(cleaned_text) < 50:
            processed_titles.add(title)
            continue
            
        # C. Split
        splitter = get_domain_splitter(domain)
        # Hack: Thêm Title vào đầu text để chunk đầu tiên luôn chứa Title context
        text_with_header = f"# {title}\n\n{cleaned_text}" 
        raw_chunks = splitter.create_documents([text_with_header])
        
        # D. Enrich & Filter
        for i, chunk in enumerate(raw_chunks):
            content = chunk.page_content.strip()
            
            # Filter rác
            if not should_keep_chunk(content):
                continue
            
            # Tạo Context Text cho Embedding (Nhẹ & Sâu)
            cat_short = shorten_category(cat_full)
            # Format: [Category] Title \n Content
            vector_text = f"[{cat_short}] {title}\n{content}"
            
            # Detect metadata
            content_info = detect_content_type(content)
            
            all_chunks.append({
                "chunk_id": f"{title}_{i}",   # ID duy nhất
                "doc_title": title,
                "doc_category": cat_full,
                "vector_text": vector_text,   # Dùng để Embed (Có ngữ cảnh)
                "display_text": content,      # Dùng để hiển thị cho LLM (Sạch)
                "doc_url": url,
                # Metadata phẳng để dễ lưu vào Qdrant payload
                "has_math": content_info['has_math'],
                "has_legal": content_info['has_legal'],
                "domain": domain
            })
        
        processed_titles.add(title)

    # 3. Save Results
    if all_chunks:
        # Append mode logic (nếu file đã tồn tại)
        if Config.LATEST_CHUNKS_FILE.exists():
            df_old = pd.read_parquet(Config.LATEST_CHUNKS_FILE)
            df_final = pd.concat([df_old, pd.DataFrame(all_chunks)], ignore_index=True)
        else:
            df_final = pd.DataFrame(all_chunks)
            
        df_final.to_parquet(Config.LATEST_CHUNKS_FILE, index=False)
        print(f"💾 Saved {len(all_chunks)} new chunks. Total: {len(df_final)}")
        
        # Update State
        with open(Config.CHUNKING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_titles), f, ensure_ascii=False)
            
    print("✅ Chunking pipeline finished.")

if __name__ == "__main__":
    process_chunking()