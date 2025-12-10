import asyncio
from qdrant_client import QdrantClient
from config import Config
from vnpt_client import get_vnpt_embedding, call_vnpt_llm
import json

# --- CẤU HÌNH ---
TOP_K = 5 # Lấy 5 đoạn văn bản liên quan nhất (Với chunk to, 5 đoạn là rất nhiều thông tin)

async def search_qdrant(query_text):
    """Tìm kiếm semantic search trên Qdrant"""
    # 1. Embed câu hỏi
    query_vector = get_vnpt_embedding(query_text)
    if not query_vector:
        print("❌ Lỗi embedding câu hỏi")
        return []

    # 2. Search Qdrant
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    
    search_result = client.search(
        collection_name=Config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
        with_payload=True
    )
    
    return search_result

def build_prompt(query, retrieved_chunks):
    """Ghép context vào prompt"""
    context_text = ""
    for i, hit in enumerate(retrieved_chunks):
        # Format: [Document Title] Content
        context_text += f"\n--- TÀI LIỆU {i+1} (Nguồn: {hit.payload.get('title', 'Unknown')}) ---\n"
        context_text += hit.payload.get('text', '') + "\n"

    # Prompt Template (Tối ưu cho Tiếng Việt & Trắc nghiệm)
    prompt = [
                {"role": "system", "content": """Bạn là một trợ lý AI thông minh tham gia cuộc thi hỏi đáp về Việt Nam.
        Nhiệm vụ của bạn là trả lời câu hỏi dựa CHÍNH XÁC và DUY NHẤT trên các đoạn văn bản được cung cấp bên dưới.
        Nếu thông tin không có trong văn bản, hãy trả lời là không biết, đừng bịa ra.
        Đối với câu hỏi trắc nghiệm, hãy suy luận và chọn đáp án đúng nhất (A, B, C, hoặc D)."""},
                
                {"role": "user", "content": f"""
        Dưới đây là thông tin tham khảo:
        {context_text}

        ----------------
        CÂU HỎI: {query}
        ----------------
        Hãy đưa ra câu trả lời cuối cùng:"""}
            ]
    return prompt

async def run_test(question):
    print(f"❓ Đang hỏi: {question}")
    
    # 1. Retrieval
    results = await search_qdrant(question)
    if not results:
        print("⚠️ Không tìm thấy tài liệu liên quan.")
        return

    print(f"✅ Tìm thấy {len(results)} chunks. Top 1 score: {results[0].score:.4f}")
    # In thử tiêu đề top 1 xem có đúng chủ đề không
    print(f"   -> Top 1 Document: {results[0].payload['title']}")

    # 2. Generation
    messages = build_prompt(question, results)
    
    # 3. Call LLM
    print("🤖 Đang suy nghĩ...")
    answer = call_vnpt_llm(messages, model=Config.LLM_MODEL_LARGE) # Dùng Large cho chắc
    
    print("\n" + "="*50)
    print("CÂU TRẢ LỜI CỦA MODEL:")
    print(answer)
    print("="*50)

if __name__ == "__main__":
    # Test thử với dữ liệu Cư Bao bạn vừa index
    # test_question = "Theo nghị quyết năm 2025, diện tích tự nhiên của phường Cư Bao mới là bao nhiêu?"
    # Hoặc test câu trắc nghiệm
    # test_question = "Đến năm 2025, phường Cư Bao thuộc đơn vị hành chính nào? \nA. Huyện Krông Búk\nB. Thị xã Buôn Hồ\nC. Thành phố Buôn Ma Thuột\nD. Tỉnh Đắk Nông"
    test_question = "Hồ Chí Minh là ai?"
    asyncio.run(run_test(test_question))