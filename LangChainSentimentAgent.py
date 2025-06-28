from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging

class SentimentAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""วิเคราะห์อารมณ์ของข้อความต่อไปนี้ (เชิงบวก/ลบ/กลาง พร้อมเหตุผล):
{text}
ผลวิเคราะห์:"""
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def analyze(self, text):
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"text": text})
            self.logger.info(f"Sentiment analysis completed for text: {text}")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return f"เกิดข้อผิดพลาด: {e}"

if __name__ == "__main__":
    agent = SentimentAgent()
    print(agent.analyze("วันนี้อากาศดีมากและฉันมีความสุข"))
