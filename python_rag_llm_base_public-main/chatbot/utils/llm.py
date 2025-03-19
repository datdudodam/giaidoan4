from langchain_openai import ChatOpenAI  # Import API của OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Import API của Google Gemini
from app.config import settings  # Import cấu hình API từ file settings


class LLM:
    """
    Lớp LLM (Large Language Model) hỗ trợ gọi API của OpenAI và Google Gemini.

    Attributes:
        temperature (float): Độ sáng tạo của mô hình.
        max_tokens (int): Số token tối đa trong một lần gọi API.
        n_ctx (int): Ngữ cảnh tối đa trong một lần gọi API.
    """

    def __init__(self, temperature: float = 0.01, max_tokens: int = 4096, n_ctx: int = 4096) -> None:
        """
        Khởi tạo lớp LLM với các tham số điều chỉnh mô hình.

        Args:
            temperature (float, optional): Độ sáng tạo của mô hình. Mặc định là 0.01.
            max_tokens (int, optional): Số lượng token tối đa. Mặc định là 4096.
            n_ctx (int, optional): Ngữ cảnh tối đa. Mặc định là 4096.
        """
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.model = ""  # Biến model để lưu mô hình đang sử dụng (nếu cần)

    def open_ai(self):
        """
        Khởi tạo mô hình OpenAI sử dụng API Key từ settings.

        Returns:
            ChatOpenAI: Đối tượng mô hình OpenAI.
        """
        llm = ChatOpenAI(
            openai_api_key=settings.KEY_API_GPT,  # API Key OpenAI
            model=settings.OPENAI_LLM,  # Mô hình OpenAI (ví dụ: 'gpt-4')
            temperature=self.temperature,
        )
        return llm

    def gemini(self):
        """
        Khởi tạo mô hình Google Gemini sử dụng API Key từ settings.

        Returns:
            ChatGoogleGenerativeAI: Đối tượng mô hình Google Gemini.
        """
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.KEY_API,  # API Key Google Gemini
            model=settings.GOOGLE_LLM,  # Mô hình Google Gemini (ví dụ: 'gemini-pro')
            temperature=self.temperature,
        )
        return llm

    def get_llm(self, llm_name: str):
        """
        Trả về mô hình LLM tương ứng dựa trên tên được cung cấp.

        Args:
            llm_name (str): Tên mô hình ('openai' hoặc 'gemini').

        Returns:
            ChatOpenAI hoặc ChatGoogleGenerativeAI: Đối tượng mô hình tương ứng.
        """
        if llm_name == "openai":
            return self.open_ai()
        elif llm_name == "gemini":
            return self.gemini()
        else:
            return self.open_ai()  # Mặc định sử dụng OpenAI nếu không có tên hợp lệ
