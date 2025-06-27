# LangChain Fact-checking / Claim Verification Agent (RAG + Critic)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactCheckAgent:
    """คลาสสำหรับ Fact-checking Agent"""
    
    def __init__(self, main_model="llama3.2:3b", critic_model="llama3.2:3b"):
        self.main_model = main_model
        self.critic_model = critic_model
        self.search_agent = None
        self.critic_chain = None
        self.setup_agents()
    
    def create_search_tool(self):
        """สร้าง search tool พร้อม fallback"""
        def search_with_fallback(query):
            try:
                # ลอง API ก่อน
                search_tool = DuckDuckGoSearchRun(backend="api")
                return search_tool.run(query)
            except Exception as e:
                logger.warning(f"API search ล้มเหลว: {e}, ลอง HTML backend")
                try:
                    # fallback ไป HTML
                    search_tool_html = DuckDuckGoSearchRun(backend="html")
                    return search_tool_html.run(query)
                except Exception as e2:
                    logger.error(f"HTML search ล้มเหลว: {e2}")
                    return f"ไม่สามารถค้นหาข้อมูลได้: {e2}"
        
        return Tool(
            name="search",
            func=search_with_fallback,
            description="ใช้ค้นหาข้อมูลจากอินเทอร์เน็ตเพื่อตรวจสอบความถูกต้องของข้ออ้าง"
        )
    
    def setup_agents(self):
        """ตั้งค่า agents และ chains"""
        try:
            # สร้าง search tool
            search_tool = self.create_search_tool()
            
            # สร้าง main LLM สำหรับค้นหา
            main_llm = OllamaLLM(model=self.main_model)
            
            # สร้าง search agent
            self.search_agent = initialize_agent(
                tools=[search_tool],
                llm=main_llm,
                agent="zero-shot-react-description",
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True
            )
            
            # สร้าง critic LLM
            critic_llm = OllamaLLM(model=self.critic_model)
            
            # สร้าง prompt สำหรับ critic
            critic_prompt = PromptTemplate(
                input_variables=["claim", "evidence"],
                template="""คุณเป็นนักวิเคราะห์ข้อมูลที่เชี่ยวชาญในการตรวจสอบข้อเท็จจริง

ข้ออ้างที่ต้องตรวจสอบ: "{claim}"

หลักฐานที่พบจากการค้นหา:
{evidence}

โปรดวิเคราะห์และประเมินความถูกต้องของข้ออ้างนี้:
1. ข้ออ้างนี้เป็นจริงหรือเท็จ? (จริง/เท็จ/ไม่แน่ใจ)
2. เหตุผลสนับสนุนการประเมิน
3. แหล่งข้อมูลที่เชื่อถือได้
4. ข้อจำกัดในการตรวจสอบ (ถ้ามี)

คำตอบ:"""
            )
            
            # สร้าง critic chain
            self.critic_chain = LLMChain(llm=critic_llm, prompt=critic_prompt)
            
            logger.info("ตั้งค่า Fact-check Agent สำเร็จ")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า: {e}")
    
    def fact_check(self, claim):
        """ตรวจสอบความถูกต้องของข้ออ้าง"""
        try:
            logger.info(f"เริ่มตรวจสอบข้ออ้าง: {claim}")
            
            # 1. ค้นหาข้อมูลสนับสนุน
            search_query = f"ตรวจสอบข้อเท็จจริง: {claim}"
            evidence_result = self.search_agent.invoke(search_query)
            evidence = evidence_result.get("output", "ไม่พบข้อมูล")
            
            # 2. วิเคราะห์ด้วย critic
            analysis = self.critic_chain.invoke({
                "claim": claim,
                "evidence": evidence
            })
            
            return {
                "claim": claim,
                "evidence": evidence,
                "analysis": analysis.get("text", analysis),
                "status": "สำเร็จ"
            }
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบ: {e}")
            return {
                "claim": claim,
                "evidence": "ไม่สามารถค้นหาได้",
                "analysis": f"เกิดข้อผิดพลาด: {e}",
                "status": "ล้มเหลว"
            }

# ฟังก์ชันสำหรับใช้งาน
def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    # สร้าง fact-check agent
    fact_checker = FactCheckAgent()
    
    # ตัวอย่างข้ออ้างสำหรับทดสอบ
    claims = [
        "ประเทศไทยมีประชากรมากกว่า 100 ล้านคน",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "โลกมีทวีปทั้งหมด 7 ทวีป"
    ]
    
    for i, claim in enumerate(claims, 1):
        print(f"\n{'='*60}")
        print(f"การตรวจสอบครั้งที่ {i}")
        print(f"{'='*60}")
        
        result = fact_checker.fact_check(claim)
        
        print(f"ข้ออ้าง: {result['claim']}")
        print(f"สถานะ: {result['status']}")
        print(f"\nการวิเคราะห์:\n{result['analysis']}")

if __name__ == "__main__":
    main()
