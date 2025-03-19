from chatbot.utils.custom_prompt import CustomPrompt  # noqa: I001
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


class NoAnswerHandler:
    """
    -Lớp trả về yêu cầu người dùng nhập lại câu đâu vào-.\n
    """
    def __init__(self, llm) -> None:

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.HANDLE_NO_ANSWER),
                ("human", " User question: {question}"),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        return self.chain
