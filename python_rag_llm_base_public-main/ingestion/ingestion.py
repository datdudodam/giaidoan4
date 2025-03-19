import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from ingestion.service_manager import ServiceManager
from langchain_community.vectorstores import FAISS


class Ingestion:
    """
    Lớp thực hiện quá trình ingest dữ liệu từ các tệp văn bản vào vector store.
    """

    def __init__(self, embedding_model_name: str):
        """
        Khởi tạo Ingestion với mô hình embedding cụ thể.

        Args:
            embedding_model_name (str): Tên của mô hình embedding được sử dụng.
        """
        self.chunk_size = 2000  # Kích thước đoạn văn bản tối đa khi chia nhỏ
        self.chunk_overlap = self.chunk_size * 0.2  # Độ chồng chéo giữa các đoạn văn bản
        self.embedding_model = ServiceManager().get_embedding_model(embedding_model_name)  # Lấy mô hình embedding

    def ingestion_folder(self, path_input_folder: str, path_vector_store: str):
        """
        Đọc tất cả các tệp văn bản trong thư mục, xử lý và lưu vào vector store.

        Args:
            path_input_folder (str): Đường dẫn đến thư mục chứa tệp văn bản.
            path_vector_store (str): Đường dẫn để lưu trữ vector store.
        """
        all_docs = []  # Danh sách lưu trữ toàn bộ tài liệu

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(path_input_folder):
            raise ValueError(f"Thư mục {path_input_folder} không tồn tại")

        # Duyệt qua toàn bộ thư mục và tệp tin
        for root, dirs, files in os.walk(path_input_folder):
            for file in files:
                if file.endswith("txt"):  # Chỉ xử lý tệp văn bản .txt
                    file_path = os.path.join(root, file)  # Lấy đường dẫn đầy đủ của tệp
                    try:
                        docs = self.process_txt(file_path, self.chunk_size)  # Xử lý tệp
                        if docs:  # Chỉ thêm nếu có tài liệu
                            all_docs.extend(docs)
                    except Exception as e:
                        print(f"Lỗi khi xử lý tệp {file_path}: {str(e)}")

        if not all_docs:
            raise ValueError("Không tìm thấy tài liệu nào để xử lý")

        # Tạo vector store từ các đoạn văn bản đã xử lý
        vectorstore = FAISS.from_documents(all_docs, self.embedding_model)
        vectorstore.save_local(path_vector_store)  # Lưu vector store vào thư mục đích

    def process_txt(self, path_file: str, chunk_size: int):
        """
        Xử lý tệp văn bản, chia nhỏ thành các đoạn văn bản có kích thước phù hợp.

        Args:
            path_file (str): Đường dẫn đến tệp văn bản cần xử lý.
            chunk_size (int): Kích thước tối đa của một đoạn văn bản.

        Returns:
            list: Danh sách các đoạn văn bản đã được xử lý.
        """
        if not os.path.exists(path_file):
            raise FileNotFoundError(f"Không tìm thấy tệp {path_file}")

        documents = TextLoader(path_file, encoding="utf8")  # Tải nội dung tệp văn bản

        # Tạo bộ chia nhỏ văn bản dựa trên các ký tự phân tách
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",  # Ngắt đoạn theo hai dòng trống
                "\n",  # Ngắt đoạn theo dòng mới
                " ",  # Ngắt đoạn theo khoảng trắng
                ".",  # Ngắt đoạn theo dấu chấm
                ",",  # Ngắt đoạn theo dấu phẩy
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",  # Ký tự rỗng
            ],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,  # Không có độ chồng chéo giữa các đoạn văn bản
        )

        # Chia nhỏ văn bản thành các đoạn
        docs = documents.load_and_split(text_splitter=text_splitter)

        if not docs:
            print(f"Cảnh báo: Không tìm thấy nội dung trong tệp {path_file}")
            return []

        # Thêm thông tin metadata cho từng đoạn văn bản
        for idx, text in enumerate(docs):
            docs[idx].metadata["file_name"] = os.path.basename(path_file)  # Lưu tên tệp nguồn
            docs[idx].metadata["chunk_size"] = chunk_size  # Lưu kích thước đoạn văn bản

        return docs
