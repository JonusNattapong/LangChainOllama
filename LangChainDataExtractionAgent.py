from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompanyInfo(BaseModel):
    """Data model for company information"""
    company_name: Optional[str] = Field(description="ชื่อบริษัท")
    founding_year: Optional[str] = Field(description="ปีที่ก่อตั้ง")
    founder: Optional[str] = Field(description="ผู้ก่อตั้ง")
    location: Optional[str] = Field(description="ที่ตั้ง")
    confidence_score: Optional[float] = Field(description="คะแนนความเชื่อมั่น (0-1)")

class DataExtractionAgent:
    def __init__(self, model_name="llama3.2:3b", temperature=0.1):
        """
        Initialize the data extraction agent
        
        Args:
            model_name: Ollama model name
            temperature: Model temperature (lower = more deterministic)
        """
        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=CompanyInfo)
        
        # Enhanced prompt with better structure and examples
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสกัดข้อมูลบริษัทจากข้อความภาษาไทย

ข้อความที่ต้องวิเคราะห์:
{text}

งาน: ดึงข้อมูลสำคัญเกี่ยวกับบริษัท ได้แก่:
- ชื่อบริษัท (รวมทั้งชื่อย่อและชื่อเต็ม)
- ปีที่ก่อตั้ง (ใช้ปี พ.ศ. หรือ ค.ศ. ตามที่ระบุ)
- ผู้ก่อตั้ง (ชื่อคนหรือกลุ่มคน)
- ที่ตั้ง (จังหวัด เขต หรือประเทศ)

ตัวอย่าง:
ข้อความ: "บริษัท ซีพี ออลล์ จำกัด (มหาชน) ก่อตั้งเมื่อปี 2511 โดยตระกูลเจียรวนนท์ มีสำนักงานใหญ่ที่กรุงเทพมหานคร"
ผลลัพธ์:
- ชื่อบริษัท: บริษัท ซีพี ออลล์ จำกัด (มหาชน)
- ปีที่ก่อตั้ง: 2511
- ผู้ก่อตั้ง: ตระกูลเจียรวนนท์
- ที่ตั้ง: กรุงเทพมหานคร

หากไม่พบข้อมูลใดๆ ให้ระบุว่า "ไม่พบข้อมูل"

ข้อมูลที่สกัดได้:
{format_instructions}""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def extract(self, text: str) -> dict:
        """
        Extract company information from text
        
        Args:
            text: Input text to extract information from
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            logger.info(f"Extracting data from text: {text[:100]}...")
            
            # Create chain
            chain = self.prompt | self.llm
            
            # Get response
            response = chain.invoke({"text": text})
            logger.info(f"LLM Response: {response}")
            
            # Try to parse structured output
            try:
                parsed_data = self.parser.parse(response)
                return parsed_data.dict()
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured output: {parse_error}")
                # Fallback to simple extraction
                return self._fallback_extraction(response)
                
        except Exception as e:
            logger.error(f"Error in extraction: {e}")
            return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}

    def _fallback_extraction(self, response: str) -> dict:
        """
        Fallback method for extracting information when structured parsing fails
        """
        result = {
            "company_name": None,
            "founding_year": None,
            "founder": None,
            "location": None,
            "raw_response": response
        }
        
        # Simple pattern matching for common formats
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if 'ชื่อบริษัท' in line or 'บริษัท' in line:
                result["company_name"] = line.split(':')[-1].strip() if ':' in line else line
            elif 'ปี' in line and ('ก่อตั้ง' in line or 'สร้าง' in line):
                result["founding_year"] = line.split(':')[-1].strip() if ':' in line else line
            elif 'ผู้ก่อตั้ง' in line or 'ผู้สร้าง' in line:
                result["founder"] = line.split(':')[-1].strip() if ':' in line else line
            elif 'ที่ตั้ง' in line or 'สถานที่' in line:
                result["location"] = line.split(':')[-1].strip() if ':' in line else line
        
        return result

    def extract_batch(self, texts: List[str]) -> List[dict]:
        """
        Extract information from multiple texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of extraction results
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.extract(text)
            result["text_index"] = i
            results.append(result)
        return results

    def export_results(self, results: List[dict], filename: str = "extraction_results.json"):
        """
        Export results to JSON file
        
        Args:
            results: List of extraction results
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

def main():
    """Main function with examples"""
    agent = DataExtractionAgent()
    
    # Test cases
    test_texts = [
        "บริษัท ABC จำกัด ก่อตั้งเมื่อปี 2540 โดยนายสมชาย ตั้งอยู่ที่กรุงเทพฯ",
        "บริษัท ไทยออยล์ จำกัด (มหาชน) หรือ PTT ก่อตั้งขึ้นในปี 2544 โดยรัฐบาลไทย มีสำนักงานใหญ่อยู่ที่จังหวัดระยอง",
        "เซเว่น อีเลฟเว่น หรือ 7-Eleven เปิดร้านแรกในประเทศไทยเมื่อปี 2532 โดยบริษัท ซีพี ออลล์ ตั้งอยู่ที่กรุงเทพมหานคร",
        "ข้อความที่ไม่มีข้อมูลบริษัทเลย"
    ]
    
    print("=== การทดสอบการสกัดข้อมูลบริษัท ===\n")
    
    # Single extraction
    for i, text in enumerate(test_texts, 1):
        print(f"ทดสอบที่ {i}:")
        print(f"ข้อความ: {text}")
        result = agent.extract(text)
        print(f"ผลลัพธ์: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 50)
    
    # Batch extraction
    print("\n=== การสกัดข้อมูลแบบกลุ่ม ===")
    batch_results = agent.extract_batch(test_texts)
    
    # Export results
    agent.export_results(batch_results)
    print(f"ส่งออกผลลัพธ์ {len(batch_results)} รายการ")

if __name__ == "__main__":
    main()