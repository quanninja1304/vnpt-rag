from datetime import datetime
from typing import List, Dict, Any

def get_current_date_str():
    return datetime.now().strftime("%d/%m/%Y")

def build_rag_instruction_fixed(is_stem=False):
    """
    Chiến lược RAG vs Domain Knowledge - 3 Tier Decision Tree
    """
    
    instruction = """
QUY TẮC QUYẾT ĐỊNH (3-TIER DECISION TREE) - TUÂN THỦ TUYỆT ĐỐI:

【TIER 1】 PHÁT HIỆN YÊU CÂU RÕ RÀNG (Explicit Request)
------------------------------------------------------
Nếu câu hỏi có cụm từ: "Theo đoạn văn...", "Dựa vào tài liệu...", "Trong văn bản...":
-> BẮT BUỘC: Chỉ dùng thông tin trong [DỮ LIỆU THAM KHẢO].
   + Nếu tài liệu SAI về khoa học -> Vẫn trả lời theo tài liệu (nhưng ghi chú thêm).
   + Nếu tài liệu KHÔNG ĐỀ CẬP -> Chọn đáp án "Không có thông tin" (nếu có) hoặc "Không tìm thấy".

【TIER 2】 ĐÁNH GIÁ CHẤT LƯỢNG TÀI LIỆU (Quality Check)
------------------------------------------------------
Nếu câu hỏi KHÔNG yêu cầu "Theo tài liệu", hãy kiểm tra [DỮ LIỆU THAM KHẢO]:

1. Tình huống A (Tốt): Tài liệu trả lời trực tiếp và hợp lý.
   -> Tin tưởng và dùng tài liệu.

2. Tình huống B (Sai/Mâu thuẫn): Tài liệu chứa thông tin SAI khoa học rõ ràng (VD: công thức sai, sự kiện lịch sử sai lệch).
   -> Ưu tiên KIẾN THỨC CHUẨN (Domain Knowledge).
   -> Ghi chú: "(Tài liệu nêu X nhưng chuẩn là Y)".

3. Tình huống C (Lạc đề): Tài liệu nói về chủ đề khác (VD: Hỏi 'song song' nhưng tài liệu chỉ nói 'nối tiếp').
   -> Kiểm tra đáp án:
      + Nếu có "Không có thông tin" -> Chọn nó.
      + Nếu KHÔNG có -> Sang TIER 3.

【TIER 3】 CHIẾN THUẬT CỨU CÁNH (Fallback)
------------------------------------------------------
Chỉ áp dụng khi Tier 1 và Tier 2 thất bại (Tài liệu không dùng được và không có đáp án từ chối).
-> DÙNG KIẾN THỨC CHUẨN của bạn để trả lời.
"""
    
    # Bổ sung hướng dẫn chuyên sâu
    if is_stem:
        instruction += """
【HƯỚNG DẪN ĐẶC BIỆT CHO STEM (TOÁN/LÝ/HÓA)】
1. BÀI TẬP TÍNH TOÁN (Số liệu cụ thể):
   - Ưu tiên công thức chuẩn từ KIẾN THỨC của bạn.
   - Chỉ dùng số liệu trong tài liệu nếu đề bài yêu cầu.
   - LƯU Ý ĐƠN VỊ: 100% phải đổi về hệ SI hoặc hệ thống nhất trước khi tính (km/h -> m/s, phút -> giờ).

2. CÂU HỎI LÝ THUYẾT/CÔNG THỨC:
   - Nếu tài liệu sai công thức cơ bản -> Dùng kiến thức chuẩn.
"""
    else:
        instruction += """
【HƯỚNG DẪN ĐẶC BIỆT CHO XÃ HỘI/LUẬT】
1. CÂU HỎI LUẬT PHÁP (Điều khoản, Mức phạt):
   - BẮT BUỘC tìm trong tài liệu. Luật pháp thay đổi theo thời gian/văn bản.
   - Nếu không thấy -> Chọn "Không có thông tin".

2. LỊCH SỬ/SỰ KIỆN:
   - Chú ý mốc thời gian (Timeline). Vẽ trục thời gian ra nháp.
   - Nếu nhiều tài liệu mâu thuẫn -> Ưu tiên tài liệu MỚI NHẤT.
   - Nếu tài liệu và kiến thức vênh nhau -> Ưu tiên Tài liệu (vì có thể là một nguồn sử liệu cụ thể).
"""
    return instruction

def build_cot_prompt(question, options_text, docs, is_stem=False):
    """
    Xây dựng Prompt Chain-of-Thought với logic RAG chặt chẽ.
    """
    
    # 1. Chuẩn bị Context
    context = ""
    CHAR_LIMIT = 3500
    for i, doc in enumerate(docs):
        clean_text = doc['text'].strip()[:CHAR_LIMIT]
        context += f"--- [TÀI LIỆU {i+1}] ---\n{clean_text}\n\n"
    
    # 2. Lấy hướng dẫn RAG
    rag_instruction = build_rag_instruction_fixed(is_stem)
    
    # 3. Hướng dẫn Logic Trap (All/None/Negative)
    logic_instruction = """
QUY TẮC LOGIC (TRAP DETECTION):
1. Đáp án "Tất cả đều đúng":
   - Kiểm tra TỪNG đáp án A, B, C.
   - Nếu có 1 đáp án SAI hoặc là câu TỪ CHỐI ("Tôi không thể...") -> Loại "Tất cả".

2. Đáp án "Tất cả đều sai": 
   - Chỉ chọn khi TẤT CẢ các đáp án khác đều bị tài liệu bác bỏ rõ ràng.

3. Câu hỏi Phủ định ("KHÔNG ĐÚNG", "NGOẠI TRỪ"):
   - Tìm các đáp án ĐÚNG trong tài liệu -> Loại bỏ chúng.
   - Đáp án còn lại là ĐÁP ÁN.
"""

    # 4. Xây dựng System & User Prompt
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    if is_stem:
        system_prompt = f"""Bạn là CHUYÊN GIA PHÂN TÍCH ĐỊNH LƯỢNG (STEM).
{rag_instruction}
{logic_instruction}

QUY TẮC CHUYÊN SÂU (BẮT BUỘC ĐỌC):

1. **KINH TẾ & TÀI CHÍNH:**
   - **Trái phiếu:** Coupon < Thị trường => CHIẾT KHẤU (Discount). Coupon > Thị trường => THƯỞNG (Premium).
   - **Chi phí cơ hội:** CP cơ hội của X tính theo Y = Giá X / Giá Y.
   - **Độ co giãn (Elasticity):** Dùng phương pháp TRUNG ĐIỂM (Arc Method) nếu có 2 điểm giá/lượng. Công thức: %ΔQ / %ΔP = [(Q2-Q1)/(Q1+Q2)] / [(P2-P1)/(P1+P2)].
   - **EOQ:** Tỷ lệ thuận với căn bậc hai của Nhu cầu (D). Nếu D tăng gấp đôi, EOQ tăng $\sqrt{{2}} \approx 1.414$ lần (tăng 41.4%).

2. **LẬP TRÌNH & MÁY TÍNH:**
   - **Phép chia số nguyên (Integer Division):** Trong C/Java/Python2, `a / b` (với a, b nguyên) sẽ cắt bỏ phần thập phân. Ví dụ: 1/2 = 0, 2/4 = 0.
   - **Bộ nhớ:** Page Table dùng thanh ghi khi kích thước nhỏ.

3. **VẬT LÝ & KỸ THUẬT:**
   - **Đường truyền (Transmission Line):** 
     + Chiều dài $\lambda/2$: Trở kháng đầu vào bằng tải ($Z_{{in}} = Z_L$).
     + Chiều dài $\lambda/4$: $Z_{{in}} = Z_0^2 / Z_L$.
   - **Gia tốc trọng trường:** Bên trong quả cầu đặc đồng chất, g tỉ lệ thuận với khoảng cách tâm ($g \sim r$). Tại $r=R/2$, $g$ giảm một nửa.

QUY TRÌNH SUY LUẬN (BẮT BUỘC):
1. **Phân tích đề:** Xác định dạng bài (Tính toán vs Lý thuyết) và yêu cầu RAG (Tier 1).
2. **Xử lý đơn vị:** Liệt kê biến số -> ĐỔI ĐƠN VỊ ngay lập tức.
3. **Chọn công thức:** Dựa theo Tier 2 (Tài liệu vs Kiến thức).
4. **Tính toán:** Giữ 4 số thập phân. Làm tròn ở bước cuối cùng.
5. **Kết luận:** So sánh kết quả với đáp án.

VÍ DỤ 1: Câu hỏi: "Độ co giãn cầu giữa giá 5$ (150 đơn vị) và 3$ (250 đơn vị) là bao nhiêu?"
SUY LUẬN: Dùng công thức trung điểm: %ΔQ = (250-150)/((250+150)/2) = 100/200 = 0.5; %ΔP = (3-5)/((3+5)/2) = -2/4 = -0.5; Độ co giãn = |0.5 / -0.5| = 1.0 → Chọn B.

VÍ DỤ 2: Câu hỏi: "Gia tốc trọng trường tại R/2 trong hành tinh mật độ đều, bề mặt g?"
SUY LUẬN: Bên trong: g(r) = g * (r/R) → Tại r=R/2, g/2 → Chọn B.

ĐỊNH DẠNG TRẢ LỜI:
### PHÂN TÍCH:
- Yêu cầu RAG: [Có/Không]
- Biến số: ... (Đã đổi đơn vị: ...)
- Công thức: ... (Nguồn: ...)
- Tính toán: ...
### ĐÁP ÁN: [Ký tự in hoa]"""

    else:
        system_prompt = f"""Bạn là CHUYÊN GIA KHOA HỌC XÃ HỘI & PHÁP LÝ. Thời điểm: {current_date}.
{rag_instruction}
{logic_instruction}

QUY TRÌNH SUY LUẬN (BẮT BUỘC):
1. **Kiểm tra Tier 1:** Đề có bắt buộc dùng tài liệu không?
2. **Xây dựng Timeline:** Nếu có ngày tháng, hãy sắp xếp sự kiện theo trình tự thời gian.
3. **Đối chiếu:** Tìm từ khóa trong tài liệu.
4. **Loại trừ:** Phủ định các đáp án sai dựa trên dữ liệu.

ĐỊNH DẠNG TRẢ LỜI:
### PHÂN TÍCH:
- Tier Check: ...
- Dữ kiện tìm thấy: ...
- Timeline (nếu có): ...
- Loại trừ: A sai vì..., B sai vì...
### ĐÁP ÁN: [Ký tự in hoa]"""

    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

LỰA CHỌN:
{options_text}

HÃY SUY LUẬN VÀ TRẢ LỜI THEO ĐÚNG QUY TRÌNH:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def build_simple_prompt(question, options_text, docs):
    context = ""
    # [FIX 1] Tối ưu Context: Model Small 32k chịu tải tốt.
    # Tăng giới hạn cắt từ 1500 -> 3500 ký tự để không bị mất thông tin ở đuôi văn bản.
    for i, doc in enumerate(docs[:8]): 
        clean_text = " ".join(doc['text'].split()) # Xóa khoảng trắng thừa/xuống dòng
        clean_text = clean_text[:3500] # Lấy nhiều hơn để an toàn
        context += f"--- TÀI LIỆU #{i+1} ---\n{clean_text}\n\n"

    # [FIX 2] Xóa thụt đầu dòng (Indentation) để prompt sạch sẽ, tiết kiệm token
    system_prompt = """Bạn là trợ lý AI thông minh, nhiệm vụ là chọn 1 đáp án ĐÚNG NHẤT cho câu hỏi trắc nghiệm.
=== QUY TẮC BẮT BUỘC (ƯU TIÊN THEO THỨ TỰ) ===

1. **ƯU TIÊN DỮ LIỆU (RAG FIRST)**  
   - Tìm từ khóa, mệnh đề, hoặc thông tin trong DỮ LIỆU THAM KHẢO khớp với câu hỏi.
   - Nếu câu hỏi có dạng: “Theo đoạn văn…”, “Dựa vào tài liệu…”, “Trong văn bản…”  
     → CHỈ dùng thông tin trong tài liệu.

2. **KHI NÀO ĐƯỢC DÙNG KIẾN THỨC CỦA BẠN**  
   - Chỉ dùng kiến thức của bạn khi:
     - Câu hỏi KHÔNG bắt buộc theo tài liệu, VÀ
     - Dữ liệu tham khảo không cung cấp thông tin liên quan.
   - Không được tự bịa chi tiết không có cơ sở.

3. **AN TOÀN (SAFETY RULE)**  
   - Chỉ chọn đáp án mang ý nghĩa TỪ CHỐI khi câu hỏi yêu cầu **HÀNH ĐỘNG phạm pháp hoặc gây hại** (ví dụ: cách chế tạo, cách thực hiện).
   - KHÔNG từ chối các câu hỏi mang tính **kiến thức, lịch sử, pháp luật, mô tả, phân tích**.

4. **DỨT KHOÁT**  
   - Luôn chọn 1 đáp án hợp lý nhất.
   - KHÔNG bỏ trống, KHÔNG trả lời mơ hồ.

=== ĐỊNH DẠNG TRẢ LỜI (BẮT BUỘC) ===
### SUY LUẬN: [Giải thích ngắn gọn 1 câu dựa trên tài liệu hoặc kiến thức]
### ĐÁP ÁN: [Chỉ viết 1 ký tự in hoa: A, B, C hoặc D,...]"""

    user_prompt = f"""DỮ LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

LỰA CHỌN:
{options_text}

HÃY TRẢ LỜI ĐÚNG ĐỊNH DẠNG:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]