from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicLLMPipeline:
    """คลาสสำหรับจัดการ LLM Pipeline พื้นฐาน (Product-ready)"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.llm = None
        self.prompt = None
        self.chain = None
        self.setup_pipeline()
    
    def setup_pipeline(self) -> None:
        try:
            self.llm = OllamaLLM(model=self.model_name)
            logger.info(f"โหลด LLM สำเร็จ: {self.model_name}")
            self.prompt = PromptTemplate.from_template(
                """คุณเป็นผู้เชี่ยวชาญที่สามารถอธิบายหัวข้อซับซ้อนให้เข้าใจง่าย

หัวข้อ: {topic}

โปรดอธิบายหัวข้อนี้ในลักษณะต่อไปนี้:
1. ความหมายพื้นฐาน
2. หลักการสำคัญ
3. ตัวอย่างในชีวิตประจำวัน
4. ประโยชน์หรือการประยุกต์ใช้

คำอธิบาย:"""
            )
            self.chain = self.prompt | self.llm
            logger.info("สร้าง Pipeline สำเร็จ")
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการตั้งค่า LLM Pipeline")
            raise

    def explain_topic(self, topic: str) -> str:
        """อธิบายหัวข้อที่กำหนด (ตรวจสอบ input และ handle error)"""
        try:
            if not isinstance(topic, str) or not topic.strip():
                logger.warning("หัวข้อว่างหรือไม่ถูกต้อง")
                return "หัวข้อว่างหรือไม่ถูกต้อง"
            if not self.chain:
                logger.error("Chain ยังไม่ถูกตั้งค่า")
                return "ระบบยังไม่พร้อมใช้งาน"
            result = self.chain.invoke({"topic": topic})
            return result
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการอธิบายหัวข้อ")
            return f"ไม่สามารถอธิบายหัวข้อได้: {e}"

    def custom_query(self, custom_prompt: str, **kwargs) -> str:
        """ใช้ prompt แบบกำหนดเอง (ตรวจสอบ input และ handle error)"""
        try:
            if not isinstance(custom_prompt, str) or not custom_prompt.strip():
                logger.warning("Custom prompt ว่างหรือไม่ถูกต้อง")
                return "Custom prompt ว่างหรือไม่ถูกต้อง"
            temp_prompt = PromptTemplate.from_template(custom_prompt)
            temp_chain = temp_prompt | self.llm
            result = temp_chain.invoke(kwargs)
            return result
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการใช้ custom prompt")
            return f"ไม่สามารถประมวลผล custom prompt ได้: {e}"

    def batch_explain(self, topics: list) -> dict:
        """อธิบายหัวข้อหลายๆ หัวข้อพร้อมกัน (ตรวจสอบ input)"""
        results = {}
        if not isinstance(topics, list) or not topics:
            logger.warning("ไม่ได้รับ list ของหัวข้อ")
            return results
        for topic in topics:
            logger.info(f"กำลังอธิบายหัวข้อ: {topic}")
            try:
                results[topic] = self.explain_topic(topic)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาด: {e}")
                results[topic] = f"เกิดข้อผิดพลาด: {e}"
        return results

import argparse

def main():
    """
    ฟังก์ชันหลักสำหรับทดสอบ BasicLLMPipeline
    สามารถรับหัวข้อหรือ custom prompt จาก command line ได้
    """
    parser = argparse.ArgumentParser(description="Basic LLM Pipeline CLI")
    parser.add_argument("--topic", type=str, help="หัวข้อเดียวที่ต้องการอธิบาย")
    parser.add_argument("--custom_prompt", type=str, help="Custom prompt (ต้องใช้ร่วมกับ --subject)")
    parser.add_argument("--subject", type=str, help="subject สำหรับ custom prompt")
    parser.add_argument("--batch", nargs="+", help="หัวข้อหลายหัวข้อ (batch)")
    args = parser.parse_args()

    pipeline = BasicLLMPipeline()

    if args.topic:
        print("=== การอธิบายหัวข้อเดี่ยว ===")
        result = pipeline.explain_topic(args.topic)
        print(f"หัวข้อ: {args.topic}")
        print(f"คำอธิบาย: {result}\n")
    if args.custom_prompt and args.subject:
        print("=== การใช้ Custom Prompt ===")
        custom_result = pipeline.custom_query(args.custom_prompt, subject=args.subject)
        print(f"Custom Prompt: {args.custom_prompt}")
        print(f"Subject: {args.subject}")
        print(f"ผลลัพธ์: {custom_result}\n")
    if args.batch:
        print("=== การประมวลผลหลายหัวข้อ ===")
        batch_results = pipeline.batch_explain(args.batch)
        for topic, explanation in batch_results.items():
            print(f"หัวข้อ: {topic}")
            print(f"คำอธิบาย: {explanation[:200]}...")
            print("-" * 40)
    if not (args.topic or (args.custom_prompt and args.subject) or args.batch):
        # Default demo
        topics = [
            "กลศาสตร์ควอนตัม",
            "ปัญญาประดิษฐ์",
            "เทคโนโลยี Blockchain"
        ]
        print("=== การทดสอบ Basic LLM Pipeline ===\n")
        print("1. การอธิบายหัวข้อเดี่ยว:")
        print("-" * 40)
        result = pipeline.explain_topic("กลศาสตร์ควอนตัม")
        print(f"หัวข้อ: กลศาสตร์ควอนตัม")
        print(f"คำอธิบาย: {result}\n")
        print("2. การใช้ Custom Prompt:")
        print("-" * 40)
        custom_result = pipeline.custom_query(
            "เขียนบทกวีสั้นๆ เกี่ยวกับ {subject}",
            subject="ธรรมชาติ"
        )
        print(f"บทกวีเกี่ยวกับธรรมชาติ: {custom_result}\n")
        print("3. การประมวลผลหลายหัวข้อ:")
        print("-" * 40)
        batch_results = pipeline.batch_explain(topics[:2])
        for topic, explanation in batch_results.items():
            print(f"หัวข้อ: {topic}")
            print(f"คำอธิบาย: {explanation[:200]}...")
            print("-" * 40)

if __name__ == "__main__":
    main()