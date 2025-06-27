# LangChain Creative Agent (Writing + Story + Comic)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreativeAgent:
    """คลาสสำหรับ Creative Writing Agent"""
    
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.story_chain = None
        self.style_chain = None
        self.creative_chain = None
        self.setup_chains()
    
    def setup_chains(self):
        """ตั้งค่า chains สำหรับการสร้างสรรค์"""
        try:
            # Prompt สำหรับแต่งเรื่องสั้น
            story_prompt = PromptTemplate(
                input_variables=["idea"],
                template="""คุณเป็นนักเขียนมืออาชีพที่เชี่ยวชาญเรื่องแฟนตาซี

จงแต่งเรื่องสั้นแนวแฟนตาซีจากไอเดียต่อไปนี้:
"{idea}"

เงื่อนไข:
- ความยาวประมาณ 3-4 ย่อหน้า
- มีจุดหักมุมที่น่าสนใจ
- ตัวละครมีมิติและน่าติดตาม
- จบแบบเปิด ให้ผู้อ่านคิดต่อได้

เรื่องสั้น:"""
            )
            
            # Prompt สำหรับแปลงสไตล์
            style_prompt = PromptTemplate(
                input_variables=["story"],
                template="""คุณเป็นกวีที่เชี่ยวชาญการแต่งกลอนสุภาพ

จงแปลงเรื่องสั้นต่อไปนี้ให้เป็นกลอนสุภาพ 4 บท (16 บรรทัด):
{story}

เงื่อนไข:
- ใช้คำสุภาพและสละสลวย
- สัมผัสและจังหวะที่ไพเราะ
- สื่อความหมายครบถ้วน
- เก็บแก่นของเรื่องไว้

กลอนสุภาพ 4 บท:"""
            )
            
            # สร้าง chains
            self.story_chain = LLMChain(llm=self.llm, prompt=story_prompt, verbose=True)
            self.style_chain = LLMChain(llm=self.llm, prompt=style_prompt, verbose=True)
            
            # สร้าง sequential chain
            self.creative_chain = SimpleSequentialChain(
                chains=[self.story_chain, self.style_chain],
                verbose=True
            )
            
            logger.info("ตั้งค่า Creative Chains สำเร็จ")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า: {e}")
    
    def create_story_only(self, idea):
        """สร้างเฉพาะเรื่องสั้น"""
        try:
            result = self.story_chain.invoke({"idea": idea})
            return result.get("text", result)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างเรื่อง: {e}")
            return f"ไม่สามารถสร้างเรื่องได้: {e}"
    
    def create_poem_from_story(self, story):
        """แปลงเรื่องเป็นกลอน"""
        try:
            result = self.style_chain.invoke({"story": story})
            return result.get("text", result)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการแปลงเป็นกลอน: {e}")
            return f"ไม่สามารถแปลงเป็นกลอนได้: {e}"
    
    def create_full_work(self, idea):
        """สร้างผลงานครบชุด (เรื่อง + กลอน)"""
        try:
            result = self.creative_chain.invoke(idea)
            return result
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างผลงาน: {e}")
            return f"ไม่สามารถสร้างผลงานได้: {e}"
    
    def create_custom_content(self, idea, content_type="story"):
        """สร้างเนื้อหาตามประเภทที่กำหนด"""
        custom_prompts = {
            "story": "แต่งเรื่องสั้นแฟนตาซี",
            "poem": "แต่งกลอนสุภาพ",
            "dialog": "เขียนบทสนทนาที่น่าสนใจ",
            "description": "เขียนบรรยายฉากที่สวยงาม"
        }
        
        prompt_text = custom_prompts.get(content_type, "สร้างเนื้อหาสร้างสรรค์")
        
        custom_prompt = PromptTemplate(
            input_variables=["idea"],
            template=f"{prompt_text} จากไอเดีย: {{idea}}"
        )
        
        try:
            custom_chain = LLMChain(llm=self.llm, prompt=custom_prompt)
            result = custom_chain.invoke({"idea": idea})
            return result.get("text", result)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างเนื้อหา: {e}")
            return f"ไม่สามารถสร้างเนื้อหาได้: {e}"

# ฟังก์ชันสำหรับใช้งาน
def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    # สร้าง creative agent
    creative_agent = CreativeAgent()
    
    # ตัวอย่างไอเดีย
    ideas = [
        "เด็กชายคนหนึ่งพบไข่มังกรในป่าและต้องปกป้องมันจากผู้ล่า",
        "นักผจญภัยสาวที่ค้นพบเมืองใต้น้ำที่สูญหายไปนานหลายศตวรรษ",
        "เด็กหญิงที่มีพลังสื่อสารกับต้นไม้และต้องช่วยป่าจากการทำลาย"
    ]
    
    for i, idea in enumerate(ideas, 1):
        print(f"\n{'='*60}")
        print(f"ผลงานสร้างสรรค์ที่ {i}")
        print(f"{'='*60}")
        print(f"ไอเดีย: {idea}")
        print("\n" + "="*60)
        
        # สร้างผลงานครบชุด
        result = creative_agent.create_full_work(idea)
        print("ผลงานสร้างสรรค์:")
        print(result)
        
        if i < len(ideas):  # ไม่แสดงสำหรับรายการสุดท้าย
            input("\nกด Enter เพื่อดูผลงานถัดไป...")

if __name__ == "__main__":
    main()
