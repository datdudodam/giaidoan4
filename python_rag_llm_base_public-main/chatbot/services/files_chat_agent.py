from chatbot.utils.llm import LLM  # noqa: I001
from chatbot.utils.retriever import Retriever
from chatbot.utils.document_grader import DocumentGrader
from chatbot.utils.answer_generator import AnswerGenerator
from chatbot.utils.no_answer_handler import NoAnswerHandler

from langgraph.graph import END, StateGraph, START
from chatbot.utils.graph_state import GraphState
from typing import Dict, Any

from app.config import settings


class FilesChatAgent:
    """
    Lớp FilesChatAgent chịu trách nhiệm quản lý quy trình chatbot,
    từ tìm kiếm tài liệu, đánh giá độ liên quan đến tạo câu trả lời và xuất kết quả HTML.
    """

    def __init__(self, path_vector_store: str) -> None:
        """
        Khởi tạo FilesChatAgent với các thành phần chính.

        Args:
            path_vector_store (str): Đường dẫn đến thư mục lưu trữ vector store.
        """
        self.retriever = Retriever(settings.LLM_NAME).set_retriever(path_vector_store)  # Khởi tạo trình tìm kiếm tài liệu
        self.llm = LLM().get_llm(settings.LLM_NAME)  # Khởi tạo mô hình ngôn ngữ
        self.document_grader = DocumentGrader(self.llm)  # Bộ đánh giá tài liệu
        self.answer_generator = AnswerGenerator(self.llm)  # Bộ tạo câu trả lời
        self.no_answer_handler = NoAnswerHandler(self.llm)  # Xử lý trường hợp không có câu trả lời

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        Tìm kiếm các tài liệu liên quan đến câu hỏi.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi.

        Returns:
            dict: Chứa danh sách tài liệu và câu hỏi.
        """
        question = state["question"]
        documents = self.retriever.get_documents(question, int(settings.NUM_DOC))
        return {"documents": documents, "question": question}

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Tạo câu trả lời dựa trên các tài liệu liên quan.

        Args:
            state (GraphState): Trạng thái hiện tại chứa câu hỏi và tài liệu.

        Returns:
            dict: Chứa câu trả lời đã được tạo.
        """
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(doc.page_content for doc in documents)  # Ghép nội dung các tài liệu thành một đoạn văn
        generation = self.answer_generator.get_chain().invoke({"question": question, "context": context})
        return {"generation": generation}

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Xác định xem có nên tạo câu trả lời hay không dựa trên tài liệu tìm được.

        Args:
            state (GraphState): Trạng thái hiện tại chứa danh sách tài liệu.

        Returns:
            str: "no_document" nếu không có tài liệu, "generate" nếu có thể tạo câu trả lời.
        """
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---QUYẾT ĐỊNH: KHÔNG CÓ VĂN BẢN LIÊN QUAN ĐẾN CÂU HỎI, BIẾN ĐỔI TRUY VẤN---")
            return "no_document"
        else:
            print("---QUYẾT ĐỊNH: TẠO CÂU TRẢ LỜI---")
            return "generate"

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
       📌 **Nhiệm vụ**:
        Bạn được yêu cầu tạo một câu trả lời chi tiết dựa trên câu hỏi của người dùng và ngữ cảnh đã cho. Hãy tuân thủ theo các bước dưới đây để đảm bảo câu trả lời rõ ràng, chính xác và có dẫn chứng khi cần thiết.

        🔹 **Hướng dẫn tạo câu trả lời**:
        1️⃣ **Xác định chủ đề chính của câu hỏi** (tuyển dụng, kinh tế, công nghệ, v.v.).
        2️⃣ **Trích xuất thông tin từ dữ liệu có sẵn** để đảm bảo câu trả lời dựa trên bằng chứng.
        3️⃣ **Diễn giải lại thông tin một cách dễ hiểu** thay vì chỉ trích xuất thô từ tài liệu.
        4️⃣ **Cung cấp ví dụ cụ thể (nếu có thể)** để tăng tính thuyết phục.
        5️⃣ **Kết thúc bằng một kết luận ngắn gọn** để tổng hợp thông tin quan trọng nhất.

        ✅ **Lưu ý**:
        - Nếu câu hỏi yêu cầu số liệu, hãy cố gắng cung cấp thông tin có dẫn chứng.
        - Tránh sử dụng câu trả lời chung chung hoặc không rõ ràng.
        - Nếu có nhiều nguồn thông tin, hãy chọn nguồn phù hợp nhất.

        📝 **Ví dụ câu trả lời**:
        - **Câu hỏi**: "Mức lương trung bình của nhân viên kinh doanh là bao nhiêu?"
        - **Trả lời**: "Theo thống kê năm 2024, mức lương trung bình của nhân viên kinh doanh dao động từ 10 triệu đến 25 triệu đồng/tháng, tùy vào kinh nghiệm và ngành nghề. Trong lĩnh vực bất động sản, mức lương có thể cao hơn do hoa hồng từ giao dịch."
    """
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = self.document_grader.get_chain().invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print("---CHẤM ĐIỂM: TÀI LIỆU LIÊN QUAN---")
                filtered_docs.append(d)
            else:
                print("---CHẤM ĐIỂM: TÀI LIỆU KHÔNG LIÊN QUAN---")

        return {"documents": filtered_docs, "question": question}

    def handle_no_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        ❌ **Không tìm thấy câu trả lời phù hợp!**  
        Hiện tại, hệ thống không thể tìm thấy câu trả lời chính xác cho câu hỏi của bạn. Để giúp bạn tốt hơn, vui lòng thử một trong các cách sau:

        🔹 **Cách cải thiện câu hỏi**:
        - Tránh đặt câu hỏi quá chung chung. Ví dụ:
            ❌ "Thông tin về tuyển dụng?" (quá rộng)
            ✅ "Xu hướng tuyển dụng nhân sự ngành công nghệ năm 2024?"
        - Nếu bạn đang hỏi về một số liệu cụ thể, hãy cung cấp khoảng thời gian hoặc ngữ cảnh liên quan.

        🔹 **Bạn cũng có thể thử các câu hỏi sau**:
        - "Mức lương trung bình của nhân viên kinh doanh năm 2024?"
        - "Kỹ năng quan trọng nhất để làm nhân viên bán hàng?"
        - "Những công ty nào đang tuyển dụng nhiều nhân viên kinh doanh?"

        ✅ **Lưu ý**:
        - Hệ thống chỉ có thể trả lời dựa trên dữ liệu có sẵn.
        - Nếu thông tin bạn cần không có trong hệ thống, hãy thử tìm kiếm trên các nguồn khác.
    """
        question = state["question"]
        generation = self.no_answer_handler.get_chain().invoke({"question": question})
        return {"generation": generation}

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của chatbot, bao gồm các bước tìm kiếm, đánh giá và tạo câu trả lời.

        Returns:
            StateGraph: Đồ thị trạng thái của quy trình chatbot.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve)  # Bước tìm kiếm tài liệu
        workflow.add_node("grade_documents", self.grade_documents)  # Bước chấm điểm tài liệu
        workflow.add_node("generate", self.generate)  # Bước tạo câu trả lời
        workflow.add_node("handle_no_answer", self.handle_no_answer)  # Bước xử lý khi không có tài liệu

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "no_document": "handle_no_answer",
                "generate": "generate",
            },
        )

        workflow.add_edge("generate", END)

        return workflow
