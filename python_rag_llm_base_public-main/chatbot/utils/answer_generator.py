from chatbot.utils.custom_prompt import CustomPrompt  # noqa: I001
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


class AnswerGenerator:
    """
    Lớp AnswerGenerator chịu trách nhiệm tạo câu trả lời dựa trên câu hỏi của người dùng
    và ngữ cảnh được cung cấp.
    """

    def __init__(self, llm) -> None:
        """
        Khởi tạo AnswerGenerator với mô hình ngôn ngữ (LLM).

        Args:
            llm: Mô hình ngôn ngữ được sử dụng để tạo câu trả lời.
        """

        # Xây dựng prompt cho chatbot với ngữ cảnh và câu hỏi của người dùng
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.GENERATE_ANSWER_PROMPT),
                ("human", " User question: {question} \n\n Context: {context}"),
            ]
        )

        # Xây dựng pipeline xử lý: nhận prompt -> xử lý với LLM -> trích xuất kết quả dạng chuỗi
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        """
        Trả về chuỗi pipeline xử lý để tạo câu trả lời.

        Returns:
            RunnableSequence: Chuỗi thực thi pipeline xử lý câu hỏi.
        """
        return self.chain
