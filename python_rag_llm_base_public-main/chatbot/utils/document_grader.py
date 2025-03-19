from langchain_core.prompts import ChatPromptTemplate  # noqa: I001
from langchain_core.pydantic_v1 import BaseModel, Field
from chatbot.utils.custom_prompt import CustomPrompt
from langchain_core.runnables import RunnableSequence


class GradeDocumentModel(BaseModel):
    """
    Mô hình dữ liệu để đánh giá mức độ liên quan của tài liệu.

    Attributes:
        binary_score (str): Giá trị điểm nhị phân xác định tài liệu có liên quan hay không ('yes' hoặc 'no').
    """

    binary_score: str = Field(description="Các tài liệu có liên quan đến câu hỏi, 'yes' or 'no'")


class DocumentGrader:
    """
    Lớp DocumentGrader chịu trách nhiệm đánh giá mức độ liên quan của tài liệu đối với câu hỏi của người dùng.
    """

    def __init__(self, llm) -> None:
        """
        Khởi tạo DocumentGrader với mô hình ngôn ngữ (LLM).

        Args:
            llm: Mô hình ngôn ngữ được sử dụng để đánh giá tài liệu.
        """

        # Tạo pipeline đầu ra có cấu trúc với mô hình GradeDocumentModel
        structured_output = llm.with_structured_output(GradeDocumentModel)

        # Xây dựng prompt cho quá trình đánh giá tài liệu
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.GRADE_DOCUMENT_PROMPT),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        # Xây dựng pipeline xử lý: nhận prompt -> xử lý với LLM -> trích xuất kết quả dạng có cấu trúc
        self.chain = prompt | structured_output

    def get_chain(self) -> RunnableSequence:
        """
        Trả về chuỗi pipeline xử lý để đánh giá tài liệu.

        Returns:
            RunnableSequence: Chuỗi thực thi pipeline đánh giá tài liệu.
        """
        return self.chain
