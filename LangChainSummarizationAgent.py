# LangChain Workflow: Summarization Agent
# สรุปเนื้อหาจากไฟล์หรือข้อความยาว ๆ ด้วย LLM + Prompt

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os
import logging

class SummarizationAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสรุปเนื้อหา

โปรดสรุปเนื้อหาต่อไปนี้ให้กระชับและเข้าใจง่าย:
{content}

สรุป:"""
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def summarize(self, content: str) -> str:
        try:
            if not isinstance(content, str) or not content.strip():
                self.logger.error("เนื้อหาว่างหรือไม่ถูกต้อง")
                return "เนื้อหาว่างหรือไม่ถูกต้อง"
            chain = self.prompt | self.llm
            result = chain.invoke({"content": content})
            return result
        except Exception as e:
            self.logger.exception("เกิดข้อผิดพลาดในการสรุปเนื้อหา")
            return f"เกิดข้อผิดพลาด: {e}"

    def summarize_from_file(self, file_path: str, encoding="utf-8") -> str:
        try:
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                self.logger.error(f"ไม่พบไฟล์: {file_path}")
                return f"ไม่พบไฟล์: {file_path}"
            with open(file_path, encoding=encoding) as f:
                content = f.read()
            return self.summarize(content)
        except Exception as e:
            self.logger.exception("เกิดข้อผิดพลาดในการอ่านไฟล์")
            return f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}"

    def summarize_batch(self, contents: list) -> list:
        results = []
        if not isinstance(contents, list) or not contents:
            self.logger.warning("ไม่ได้รับ list ของเนื้อหา")
            return results
        for content in contents:
            results.append(self.summarize(content))
        return results

    def summarize_and_save(self, content: str, output_path: str, encoding="utf-8") -> str:
        summary = self.summarize(content)
        try:
            with open(output_path, "w", encoding=encoding) as f:
                f.write(str(summary))
            self.logger.info(f"บันทึกสรุปผลที่: {output_path}")
            return output_path
        except Exception as e:
            self.logger.exception("เกิดข้อผิดพลาดในการบันทึกไฟล์")
            return f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}"

if __name__ == "__main__":
    agent = SummarizationAgent()
    # ตัวอย่าง workflow 1: สรุปข้อความเดี่ยว
    text = "AI (Artificial Intelligence) หรือปัญญาประดิษฐ์ ... (เนื้อหายาว)"
    print(agent.summarize(text))
    # ตัวอย่าง workflow 2: สรุปจากไฟล์
    print(agent.summarize_from_file("docs/mydoc.txt"))
    # ตัวอย่าง workflow 3: สรุปหลายรายการ
    batch = [
        "AI คือเทคโนโลยีที่ช่วยให้คอมพิวเตอร์คิดได้เหมือนมนุษย์",
        "AI มีข้อดีคือช่วยลดข้อผิดพลาดและเพิ่มประสิทธิภาพ"
    ]
    print(agent.summarize_batch(batch))
    # ตัวอย่าง workflow 4: สรุปและบันทึกไฟล์
    agent.summarize_and_save(text, "summary.txt")
