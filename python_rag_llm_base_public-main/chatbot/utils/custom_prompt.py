class CustomPrompt:
    GRADE_DOCUMENT_PROMPT = """
        Bạn là người đánh giá mức độ liên quan của một tài liệu đã được truy xuất đối với câu hỏi của người dùng. 
        Mục tiêu của bạn là xác định một cách chính xác xem liệu tài liệu có chứa thông tin liên quan, ...
        Hãy thực hiện các bước dưới đây một cách cẩn thận,...

        Các bước hướng dẫn cụ thể:
        
        1. ...

        2. ...

        3. ...
            
        4. ...
        
        Lưu ý: Không thêm bất kỳ nội dung gì khác.
    """

    GENERATE_ANSWER_PROMPT = """
        Bạn được yêu cầu tạo một câu trả lời dựa trên câu hỏi và ngữ cảnh đã cho. Hãy tuân thủ theo các bước dưới đây để đảm bảo câu trả lời của bạn có thể hiển thị chính xác và đầy đủ thông tin. Các chi tiết phải được thực hiện chính xác 100%.

        Hướng dẫn cụ thể:

        ....
            
    """

    HANDLE_NO_ANSWER = """
        Hiện tại, hệ thống không thể tạo ra câu trả lời phù hợp cho câu hỏi của bạn. 
        Để giúp bạn tốt hơn, vui lòng tạo một câu hỏi mới theo hướng dẫn sau:

        ....
    """