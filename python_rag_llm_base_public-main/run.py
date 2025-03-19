import os

#chuẩn bị dữ liệu
from ingestion.ingestion import Ingestion

# Lấy đường dẫn tuyệt đối của thư mục chứa script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_in_path = os.path.join(current_dir, "demo", "data_in")
data_vector_path = os.path.join(current_dir, "demo", "data_vector")

Ingestion("openai").ingestion_folder(
    path_input_folder=data_in_path,
    path_vector_store=data_vector_path,
)

# chatbot
from chatbot.services.files_chat_agent import FilesChatAgent  # noqa: E402
from app.config import settings

settings.LLM_NAME = "openai"

_question = "Nhu cầu tuyển dụng trong nhóm kinh doanh/bán hàng có đang giữ vững vị trí dẫn đầu và gia tăng không?"
chat = FilesChatAgent(data_vector_path).get_workflow().compile().invoke(
    input={
        "question": _question,
    }
)

print(chat)

print("generation", chat["generation"])