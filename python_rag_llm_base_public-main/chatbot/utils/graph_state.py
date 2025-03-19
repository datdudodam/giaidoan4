from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Lớp GraphState đại diện cho trạng thái của đồ thị.

    Attributes:
        question (str): Câu hỏi của người dùng.
        generation (str): Kết quả sinh ra từ mô hình LLM.
        documents (List[str]): Danh sách tài liệu được truy xuất.
    """

    question: str
    generation: str
    documents: List[str]