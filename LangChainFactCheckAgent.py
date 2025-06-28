# Simplified Robust Fact-Checking Agent
# pip install langchain langchain-community langchain-ollama duckduckgo-search pydantic

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
import logging
import json
import time
from datetime import datetime

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactCheckResult(BaseModel):
    """Model สำหรับผลลัพธ์การตรวจสอบข้อเท็จจริง"""
    claim: str
    verdict: Literal["จริง", "เท็จ", "ไม่แน่ใจ", "ข้อมูลไม่เพียงพอ"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    evidence: List[str]
    sources: List[str]
    reasoning: str
    timestamp: str
    processing_time: float
    status: Literal["สำเร็จ", "ล้มเหลว", "ข้อมูลไม่เพียงพอ"]

class SimplifiedFactCheckAgent:
    """Simplified Fact-checking Agent - More Robust and Reliable"""
    
    def __init__(self, 
                 main_model: str = "llama3.2:3b", 
                 critic_model: str = "llama3.2:3b",
                 max_search_results: int = 3):
        self.main_model = main_model
        self.critic_model = critic_model
        self.max_search_results = max_search_results
        self.main_llm = None
        self.critic_llm = None
        self.search_tool = None
        self.setup_components()
    
    def setup_components(self):
        """ตั้งค่า components ต่างๆ"""
        try:
            # สร้าง LLMs
            self.main_llm = OllamaLLM(model=self.main_model, temperature=0.3)
            self.critic_llm = OllamaLLM(model=self.critic_model, temperature=0.1)
            
            # สร้าง search tool
            self.search_tool = DuckDuckGoSearchRun()
            
            logger.info("Simplified Fact-check Agent setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up components: {e}")
            raise
    
    def search_information(self, query: str) -> Dict:
        """ค้นหาข้อมูลพร้อม error handling"""
        try:
            if not isinstance(query, str) or not query.strip():
                return {"success": False, "error": "คำค้นหาว่าง", "results": []}
            
            # Rate limiting
            time.sleep(1)
            
            # ลอง API backend ก่อน
            try:
                search_tool_api = DuckDuckGoSearchRun(backend="api")
                raw_results = search_tool_api.run(query)
                return {
                    "success": True,
                    "query": query,
                    "results": [raw_results],
                    "method": "api"
                }
            except Exception as e1:
                logger.warning(f"API search failed: {e1}, trying HTML backend")
                
                # ลอง HTML backend
                try:
                    search_tool_html = DuckDuckGoSearchRun(backend="html")
                    raw_results = search_tool_html.run(query)
                    return {
                        "success": True,
                        "query": query,
                        "results": [raw_results],
                        "method": "html"
                    }
                except Exception as e2:
                    logger.error(f"HTML search also failed: {e2}")
                    return {
                        "success": False,
                        "error": f"การค้นหาล้มเหลว: {e2}",
                        "query": query,
                        "results": []
                    }
                    
        except Exception as e:
            logger.error(f"Search method failed completely: {e}")
            return {
                "success": False,
                "error": f"เกิดข้อผิดพลาดในการค้นหา: {e}",
                "query": query,
                "results": []
            }
    
    def generate_search_queries(self, claim: str) -> List[str]:
        """สร้าง search queries หลายแบบสำหรับข้ออ้าง"""
        base_queries = [
            f"ตรวจสอบข้อเท็จจริง {claim}",
            f"ข้อมูล {claim}",
            f"หลักฐาน {claim}"
        ]
        
        # ถ้าข้ออ้างมีตัวเลข ให้ค้นหาเฉพาะหัวข้อหลักด้วย
        if any(char.isdigit() for char in claim):
            # แยกคำสำคัญจากข้ออ้าง
            key_words = []
            if "ประเทศไทย" in claim or "ไทย" in claim:
                key_words.append("ประชากรไทย")
            if "กรุงเทพ" in claim:
                key_words.append("กรุงเทพมหานคร เมืองหลวง")
            if "ทวีป" in claim:
                key_words.append("จำนวนทวีปโลก")
            
            base_queries.extend(key_words)
        
        return base_queries[:self.max_search_results]
    
    def collect_evidence(self, claim: str) -> Dict:
        """รวบรวมหลักฐานจากการค้นหาหลายรอบ"""
        search_queries = self.generate_search_queries(claim)
        all_evidence = []
        successful_searches = 0
        
        logger.info(f"Searching with {len(search_queries)} queries")
        
        for i, query in enumerate(search_queries, 1):
            logger.info(f"Search {i}/{len(search_queries)}: {query}")
            
            result = self.search_information(query)
            
            if result["success"]:
                successful_searches += 1
                evidence_text = "\n".join(result["results"])
                if evidence_text.strip():
                    all_evidence.append({
                        "query": query,
                        "evidence": evidence_text,
                        "method": result.get("method", "unknown")
                    })
            else:
                logger.warning(f"Search failed for: {query}")
        
        combined_evidence = "\n\n--- ผลการค้นหาต่อไปนี้ ---\n\n".join([
            f"คำค้นหา: {item['query']}\nผลลัพธ์: {item['evidence']}" 
            for item in all_evidence
        ])
        
        return {
            "evidence": combined_evidence or "ไม่พบข้อมูลสนับสนุน",
            "search_count": len(search_queries),
            "successful_count": successful_searches,
            "success_rate": successful_searches / len(search_queries) if search_queries else 0
        }
    
    def analyze_claim(self, claim: str, evidence: str) -> Dict:
        """วิเคราะห์ข้ออ้างด้วย LLM"""
        
        prompt_template = """คุณเป็นผู้เชี่ยวชาญด้านการตรวจสอบข้อเท็จจริงที่มีความแม่นยำสูง

ข้ออ้างที่ต้องตรวจสอบ: "{claim}"

หลักฐานจากการค้นหา:
{evidence}

โปรดวิเคราะห์อย่างละเอียดและตอบในรูปแบบต่อไปนี้:

VERDICT: [จริง/เท็จ/ไม่แน่ใจ/ข้อมูลไม่เพียงพอ]
CONFIDENCE: [0.0-1.0]
REASONING: [เหตุผลการวิเคราะห์อย่างละเอียด]
KEY_EVIDENCE: [หลักฐานสำคัญที่พบ]
LIMITATIONS: [ข้อจำกัดในการตรวจสอบ]

เกณฑ์การประเมิน:
- จริง: หลักฐานสนับสนุนอย่างชัดเจน (confidence > 0.8)
- เท็จ: หลักฐานขัดแย้งอย่างชัดเจน (confidence > 0.8)  
- ไม่แน่ใจ: หลักฐานผสมผสานหรือไม่ชัดเจน (confidence 0.4-0.8)
- ข้อมูลไม่เพียงพอ: หลักฐานน้อยเกินไป (confidence < 0.4)

การวิเคราะห์:"""

        try:
            prompt = prompt_template.format(claim=claim, evidence=evidence)
            response = self.critic_llm.invoke(prompt)
            
            # Parse response
            parsed = self._parse_analysis_response(response)
            return parsed
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "verdict": "ข้อมูลไม่เพียงพอ",
                "confidence_score": 0.0,
                "reasoning": f"เกิดข้อผิดพลาดในการวิเคราะห์: {e}",
                "key_evidence": [],
                "limitations": "ระบบวิเคราะห์ขัดข้อง"
            }
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """แยกวิเคราะห์การตอบกลับจาก LLM"""
        try:
            # Initialize default values
            result = {
                "verdict": "ไม่แน่ใจ",
                "confidence_score": 0.5,
                "reasoning": response,
                "key_evidence": [],
                "limitations": "การวิเคราะห์อัตโนมัติ"
            }
            
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('VERDICT:'):
                    verdict_text = line.replace('VERDICT:', '').strip()
                    if any(word in verdict_text for word in ['จริง', 'ถูก', 'correct', 'true']):
                        result["verdict"] = "จริง"
                    elif any(word in verdict_text for word in ['เท็จ', 'ผิด', 'false', 'incorrect']):
                        result["verdict"] = "เท็จ"
                    elif any(word in verdict_text for word in ['ไม่แน่ใจ', 'uncertain']):
                        result["verdict"] = "ไม่แน่ใจ"
                    elif any(word in verdict_text for word in ['ไม่เพียงพอ', 'insufficient']):
                        result["verdict"] = "ข้อมูลไม่เพียงพอ"
                
                elif line.startswith('CONFIDENCE:'):
                    try:
                        conf_text = line.replace('CONFIDENCE:', '').strip()
                        # Extract number from text
                        import re
                        numbers = re.findall(r'0\.\d+|\d+\.\d+', conf_text)
                        if numbers:
                            result["confidence_score"] = float(numbers[0])
                    except:
                        pass
                
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                    if reasoning:
                        result["reasoning"] = reasoning
                
                elif line.startswith('KEY_EVIDENCE:'):
                    evidence = line.replace('KEY_EVIDENCE:', '').strip()
                    if evidence:
                        result["key_evidence"] = [evidence]
                
                elif line.startswith('LIMITATIONS:'):
                    limitations = line.replace('LIMITATIONS:', '').strip()
                    if limitations:
                        result["limitations"] = limitations
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            return {
                "verdict": "ไม่แน่ใจ",
                "confidence_score": 0.5,
                "reasoning": response,
                "key_evidence": [response[:200]],
                "limitations": "การแยกวิเคราะห์ล้มเหลว"
            }
    
    def fact_check(self, claim: str) -> FactCheckResult:
        """ตรวจสอบความถูกต้องของข้ออ้าง - วิธีง่าย ๆ แต่มีประสิทธิภาพ"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Validate input
            if not isinstance(claim, str) or not claim.strip():
                return FactCheckResult(
                    claim=claim,
                    verdict="ข้อมูลไม่เพียงพอ",
                    confidence_score=0.0,
                    evidence=[],
                    sources=[],
                    reasoning="ข้ออ้างว่างหรือไม่ถูกต้อง",
                    timestamp=timestamp,
                    processing_time=time.time() - start_time,
                    status="ล้มเหลว"
                )
            
            logger.info(f"Starting fact-check for: {claim[:100]}...")
            
            # 1. Collect evidence
            evidence_result = self.collect_evidence(claim)
            
            # 2. Analyze the claim
            analysis = self.analyze_claim(claim, evidence_result["evidence"])
            
            processing_time = time.time() - start_time
            
            # 3. Determine status
            status = "สำเร็จ"
            if evidence_result["successful_count"] == 0:
                status = "ข้อมูลไม่เพียงพอ"
            elif analysis["confidence_score"] < 0.3:
                status = "ข้อมูลไม่เพียงพอ"
            
            return FactCheckResult(
                claim=claim,
                verdict=analysis["verdict"],
                confidence_score=analysis["confidence_score"],
                evidence=[evidence_result["evidence"][:1000]],  # Limit evidence length
                sources=[f"ค้นหาสำเร็จ {evidence_result['successful_count']}/{evidence_result['search_count']} ครั้ง"],
                reasoning=analysis["reasoning"],
                timestamp=timestamp,
                processing_time=processing_time,
                status=status
            )
            
        except Exception as e:
            logger.exception("Error during fact-checking process")
            processing_time = time.time() - start_time
            
            return FactCheckResult(
                claim=claim,
                verdict="ข้อมูลไม่เพียงพอ",
                confidence_score=0.0,
                evidence=[],
                sources=[],
                reasoning=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}",
                timestamp=timestamp,
                processing_time=processing_time,
                status="ล้มเหลว"
            )

# Simplified demo function
def run_simplified_demo():
    """ฟังก์ชันสำหรับทดสอบ Simplified Agent"""
    print("🔍 Simplified Fact-Checking Agent Demo")
    print("=" * 60)
    
    # สร้าง simplified fact-checker
    fact_checker = SimplifiedFactCheckAgent(max_search_results=2)
    
    # ตัวอย่างข้ออ้างสำหรับทดสอบ
    test_claims = [
        "ประเทศไทยมีประชากรประมาณ 70 ล้านคน",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย", 
        "โลกมีทวีปทั้งหมด 7 ทวีป"
    ]
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n📋 การตรวจสอบข้ออ้างที่ {i}")
        print("-" * 50)
        print(f"ข้ออ้าง: {claim}")
        
        result = fact_checker.fact_check(claim)
        
        print(f"ผลการตรวจสอบ: {result.verdict}")
        print(f"ความมั่นใจ: {result.confidence_score:.2f}")
        print(f"สถานะ: {result.status}")
        print(f"เวลาประมวลผล: {result.processing_time:.2f} วินาที")
        print(f"เหตุผล: {result.reasoning[:300]}...")
        
        if result.sources:
            print(f"แหล่งข้อมูล: {', '.join(result.sources)}")
        
        # พักระหว่างการประมวลผล
        if i < len(test_claims):
            print("\n⏳ รอ 3 วินาทีก่อนตรวจสอบข้ออ้างต่อไป...")
            time.sleep(3)

if __name__ == "__main__":
    run_simplified_demo()