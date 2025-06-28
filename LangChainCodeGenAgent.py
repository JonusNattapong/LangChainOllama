from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class CodeGenAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["description"],
            template="""เขียนโค้ด Python ตามคำอธิบายต่อไปนี้:
{description}
โค้ด:"""
        )

    def generate_code(self, description):
        try:
            chain = self.prompt | self.llm
            return chain.invoke({"description": description})
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {e}"

import argparse

def main():
    """
    ฟังก์ชันหลักสำหรับทดสอบ CodeGenAgent
    สามารถรับ description จาก command line ได้
    """
    parser = argparse.ArgumentParser(description="CodeGenAgent CLI")
    parser.add_argument("--desc", type=str, help="คำอธิบายสำหรับสร้างโค้ด Python")
    args = parser.parse_args()

    agent = CodeGenAgent()
    if args.desc:
        print(agent.generate_code(args.desc))
    else:
        print(agent.generate_code("เขียนฟังก์ชัน Python หาค่าเฉลี่ยของ list"))

if __name__ == "__main__":
    main()
