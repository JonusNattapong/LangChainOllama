from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging

class TextClassificationAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""จัดประเภทข้อความต่อไปนี้ (เช่น เทคโนโลยี, สุขภาพ, การเงิน, กีฬา ฯลฯ) พร้อมเหตุผล:
{text}
หมวดหมู่:"""
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def classify(self, text):
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"text": text})
            self.logger.info(f"Text classification completed for text: {text}")
            return result
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}")
            return f"เกิดข้อผิดพลาด: {e}"

if __name__ == "__main__":
    agent = TextClassificationAgent()
    print(agent.classify("บทความนี้เกี่ยวกับเทคโนโลยี AI และการประยุกต์ใช้"))
