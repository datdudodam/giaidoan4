from langchain_community.vectorstores import FAISS
from ingestion.service_manager import ServiceManager


class Retriever:
    """
    Lớp khởi tạo embedding_model và lấy documents.

    Attributes:
        embedding_model (Any): Mô hình embedding được lấy từ ServiceManager.
        faiss_fetch_k (int): Số lượng document tối đa được lấy trước khi lọc.
    """

    def __init__(self, embedding_model_name: str, faiss_fetch_k: int = 20):
        """
        Khởi tạo lớp Retriever với mô hình embedding.

        Args:
            embedding_model_name (str): Tên mô hình embedding.
            faiss_fetch_k (int, optional): Số lượng document tối đa lấy trước khi lọc. Mặc định là 20.
        """
        self.faiss_fetch_k = faiss_fetch_k
        self.embedding_model = ServiceManager().get_embedding_model(embedding_model_name)

    def set_retriever(self, path_vector_store: str):
        """
        Thiết lập Retriever bằng FAISS từ đường dẫn dữ liệu vector.

        Args:
            path_vector_store (str): Đường dẫn đến kho dữ liệu vector FAISS.

        Returns:
            Retriever: Đối tượng Retriever đã được thiết lập.
        """
        self.retriever = FAISS.load_local(path_vector_store, self.embedding_model, allow_dangerous_deserialization=True)
        return self

    def get_as_retriever(self):
        """
        Lấy đối tượng retriever từ FAISS.

        Returns:
            Any: Đối tượng retriever của FAISS.
        """
        faiss_retriever = self.retriever.as_retriever()
        return faiss_retriever

    def get_documents(self, query: str, num_doc: int = 5):
        """
        Nhận vào câu hỏi và trả về danh sách các document liên quan.

        Args:
            query (str): Câu hỏi cần tìm kiếm.
            num_doc (int, optional): Số lượng document cần lấy. Mặc định là 5.

        Returns:
            List[Any]: Danh sách các document liên quan.
        """
        docs = self.retriever.similarity_search(query, k=num_doc, fetch_k=self.faiss_fetch_k)
        return docs
