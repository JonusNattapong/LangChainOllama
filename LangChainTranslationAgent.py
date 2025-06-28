from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class TranslationAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["text", "target_language"],
            template="""แปลข้อความต่อไปนี้เป็นภาษา {target_language}:
{text}
คำแปล:"""
        )

    def translate(self, text, target_language="en"):
        try:
            chain = self.prompt | self.llm
            return chain.invoke({"text": text, "target_language": target_language})
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {e}"

if __name__ == "__main__":
    agent = TranslationAgent()
    print(agent.translate("สวัสดีครับ ยินดีต้อนรับ!", target_language="en"))
