from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class DataExtractionAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""ดึงข้อมูลสำคัญ (ชื่อบริษัท, ปีที่ก่อตั้ง, ผู้ก่อตั้ง, ที่ตั้ง) จากข้อความต่อไปนี้:
{text}
ข้อมูลที่สกัดได้:"""
        )

    def extract(self, text):
        try:
            chain = self.prompt | self.llm
            return chain.invoke({"text": text})
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {e}"

if __name__ == "__main__":
    agent = DataExtractionAgent()
    print(agent.extract("บริษัท ABC จำกัด ก่อตั้งเมื่อปี 2540 โดยนายสมชาย ตั้งอยู่ที่กรุงเทพฯ"))
