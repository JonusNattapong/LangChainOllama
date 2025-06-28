from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging
from typing import Dict, List, Optional
import json
import re

class TextClassificationAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the Thai text classification agent
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.llm = OllamaLLM(model=model_name)
        
        # Enhanced prompt with more specific instructions
        self.prompt = PromptTemplate(
            input_variables=["text", "categories"],
            template="""จัดประเภทข้อความภาษาไทยต่อไปนี้ให้อยู่ในหมวดหมู่ที่เหมาะสม:

หมวดหมู่ที่เป็นไปได้: {categories}

ข้อความ: {text}

กรุณาตอบในรูปแบบ JSON ดังนี้:
{{
    "category": "หมวดหมู่ที่เลือก",
    "confidence": "ระดับความมั่นใจ (สูง/กลาง/ต่ำ)",
    "reasoning": "เหตุผลในการจัดประเภท"
}}

คำตอบ:"""
        )
        
        # Default categories in Thai
        self.default_categories = [
            "เทคโนโลยี", "สุขภาพ", "การเงิน", "กีฬา", "การเมือง", 
            "บันเทิง", "การศึกษา", "ข่าวสาร", "ท่องเที่ยว", "อาหาร",
            "แฟชั่น", "ธุรกิจ", "วิทยาศาสตร์", "ศิลปะ", "อื่นๆ"
        ]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def classify(self, text: str, categories: Optional[List[str]] = None) -> Dict:
        """
        Classify Thai text into categories
        
        Args:
            text: Thai text to classify
            categories: Optional list of categories to choose from
            
        Returns:
            Dictionary with classification results
        """
        if not text or not text.strip():
            return {
                "error": "ข้อความว่างเปล่า",
                "category": None,
                "confidence": None,
                "reasoning": None
            }
        
        # Use provided categories or default ones
        cats = categories if categories else self.default_categories
        categories_str = ", ".join(cats)
        
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({
                "text": text.strip(),
                "categories": categories_str
            })
            
            # Try to parse JSON response
            parsed_result = self._parse_response(result)
            
            self.logger.info(f"Classification completed for text: {text[:50]}...")
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}")
            return {
                "error": f"เกิดข้อผิดพลาด: {str(e)}",
                "category": None,
                "confidence": None,
                "reasoning": None
            }

    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM response and extract structured information
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed dictionary with classification results
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return {
                    "category": result.get("category", "ไม่ระบุ"),
                    "confidence": result.get("confidence", "ไม่ระบุ"),
                    "reasoning": result.get("reasoning", "ไม่ระบุ"),
                    "raw_response": response
                }
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: parse unstructured response
        lines = response.strip().split('\n')
        category = "ไม่ระบุ"
        confidence = "ไม่ระบุ"
        reasoning = response
        
        for line in lines:
            if any(cat in line for cat in self.default_categories):
                category = line.strip()
                break
        
        return {
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_response": response
        }

    def classify_batch(self, texts: List[str], categories: Optional[List[str]] = None) -> List[Dict]:
        """
        Classify multiple texts at once
        
        Args:
            texts: List of Thai texts to classify
            categories: Optional list of categories
            
        Returns:
            List of classification results
        """
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.classify(text, categories)
            results.append(result)
        return results

    def get_category_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """
        Get distribution of categories from classification results
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with category counts
        """
        distribution = {}
        for result in results:
            if result.get("category") and "error" not in result:
                category = result["category"]
                distribution[category] = distribution.get(category, 0) + 1
        return distribution

def main():
    """Example usage of the TextClassificationAgent"""
    agent = TextClassificationAgent()
    
    # Test texts in Thai
    test_texts = [
        "บทความนี้เกี่ยวกับเทคโนโลยี AI และการประยุกต์ใช้",
        "การออกกำลังกายเป็นประจำช่วยให้สุขภาพแข็งแรง",
        "ราคาหุ้นในตลาดหลักทรัพย์ปรับตัวเพิ่มขึ้น",
        "ทีมฟุตบอลไทยเอาชนะคู่แข่งได้อย่างสวยงาม"
    ]
    
    print("=== การจัดประเภทข้อความภาษาไทย ===\n")
    
    # Single text classification
    for i, text in enumerate(test_texts, 1):
        print(f"ข้อความที่ {i}: {text}")
        result = agent.classify(text)
        
        if "error" in result:
            print(f"ข้อผิดพลาด: {result['error']}")
        else:
            print(f"หมวดหมู่: {result['category']}")
            print(f"ความมั่นใจ: {result['confidence']}")
            print(f"เหตุผล: {result['reasoning']}")
        print("-" * 50)
    
    # Batch classification
    print("\n=== การจัดประเภทแบบกลุ่ม ===")
    batch_results = agent.classify_batch(test_texts)
    distribution = agent.get_category_distribution(batch_results)
    
    print("การกระจายของหมวดหมู่:")
    for category, count in distribution.items():
        print(f"  {category}: {count} ข้อความ")

if __name__ == "__main__":
    main()