from langchain_openai import OpenAIEmbeddings
from app.config import settings


class ServiceManager:
    """
    Quản lý các dịch vụ liên quan đến embeddings.
    """

    def __init__(self) -> None:
        """
        Khởi tạo ServiceManager.
        """
        pass

    def get_embedding_model(self, embedding_model_name: str):
        """
        Trả về mô hình embeddings tương ứng dựa trên tên mô hình.

        Args:
            embedding_model_name (str): Tên của mô hình embeddings.

        Returns:
            OpenAIEmbeddings | None: Đối tượng OpenAIEmbeddings nếu tìm thấy, ngược lại trả về None.
        """
        embeddings = None
        if embedding_model_name == "openai":
            embeddings = OpenAIEmbeddings(openai_api_key=settings.KEY_API_GPT)
        return embeddings
