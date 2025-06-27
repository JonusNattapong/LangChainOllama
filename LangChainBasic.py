from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicLLMPipeline:
    """คลาสสำหรับจัดการ LLM Pipeline พื้นฐาน"""
    
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.llm = None
        self.prompt = None
        self.chain = None
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """ตั้งค่า LLM และ Pipeline"""
        try:
            # สร้าง LLM
            self.llm = OllamaLLM(model=self.model_name)
            logger.info(f"โหลด LLM สำเร็จ: {self.model_name}")
            
            # สร้าง Prompt Template
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
            
            # สร้าง Chain
            self.chain = self.prompt | self.llm
            logger.info("สร้าง Pipeline สำเร็จ")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า: {e}")
    
    def explain_topic(self, topic):
        """อธิบายหัวข้อที่กำหนด"""
        try:
            if not self.chain:
                return "ระบบยังไม่พร้อมใช้งาน"
            
            result = self.chain.invoke({"topic": topic})
            return result
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอธิบาย: {e}")
            return f"ไม่สามารถอธิบายหัวข้อได้: {e}"
    
    def custom_query(self, custom_prompt, **kwargs):
        """ใช้ prompt แบบกำหนดเอง"""
        try:
            # สร้าง prompt ชั่วคราว
            temp_prompt = PromptTemplate.from_template(custom_prompt)
            temp_chain = temp_prompt | self.llm
            
            result = temp_chain.invoke(kwargs)
            return result
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการใช้ custom prompt: {e}")
            return f"ไม่สามารถประมวลผล custom prompt ได้: {e}"
    
    def batch_explain(self, topics):
        """อธิบายหัวข้อหลายๆ หัวข้อพร้อมกัน"""
        results = {}
        
        for topic in topics:
            logger.info(f"กำลังอธิบายหัวข้อ: {topic}")
            results[topic] = self.explain_topic(topic)
        
        return results

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    # สร้าง LLM Pipeline
    pipeline = BasicLLMPipeline()
    
    # ตัวอย่างหัวข้อสำหรับทดสอบ
    topics = [
        "กลศาสตร์ควอนตัม",
        "ปัญญาประดิษฐ์",
        "เทคโนโลยี Blockchain"
    ]
    
    print("=== การทดสอบ Basic LLM Pipeline ===\n")
    
    # ทดสอบการอธิบายหัวข้อเดี่ยว
    print("1. การอธิบายหัวข้อเดี่ยว:")
    print("-" * 40)
    result = pipeline.explain_topic("กลศาสตร์ควอนตัม")
    print(f"หัวข้อ: กลศาสตร์ควอนตัม")
    print(f"คำอธิบาย: {result}\n")
    
    # ทดสอบ custom prompt
    print("2. การใช้ Custom Prompt:")
    print("-" * 40)
    custom_result = pipeline.custom_query(
        "เขียนบทกวีสั้นๆ เกี่ยวกับ {subject}",
        subject="ธรรมชาติ"
    )
    print(f"บทกวีเกี่ยวกับธรรมชาติ: {custom_result}\n")
    
    # ทดสอบ batch processing (แสดงเฉพาะหัวข้อแรก)
    print("3. การประมวลผลหลายหัวข้อ:")
    print("-" * 40)
    batch_results = pipeline.batch_explain(topics[:2])  # ทดสอบแค่ 2 หัวข้อ
    
    for topic, explanation in batch_results.items():
        print(f"หัวข้อ: {topic}")
        print(f"คำอธิบาย: {explanation[:200]}...")  # แสดงแค่ 200 ตัวอักษรแรก
        print("-" * 40)

if __name__ == "__main__":
    main()