from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional, Tuple
import logging
import re
import json
from dataclasses import dataclass

@dataclass
class TranslationResult:
    """Data class for translation results"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: str
    detected_language: Optional[str] = None
    error: Optional[str] = None

class EnhancedTranslationAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize Enhanced Translation Agent
        
        Args:
            model_name: Ollama model name to use
        """
        self.llm = OllamaLLM(model=model_name)
        
        # Language mappings for better handling
        self.language_codes = {
            'th': 'ไทย', 'thai': 'ไทย',
            'en': 'อังกฤษ', 'english': 'อังกฤษ',
            'zh': 'จีน', 'chinese': 'จีน',
            'ja': 'ญี่ปุ่น', 'japanese': 'ญี่ปุ่น',
            'ko': 'เกาหลี', 'korean': 'เกาหลี',
            'fr': 'ฝรั่งเศส', 'french': 'ฝรั่งเศส',
            'de': 'เยอรมัน', 'german': 'เยอรมัน',
            'es': 'สเปน', 'spanish': 'สเปน',
            'it': 'อิตาลี', 'italian': 'อิตาลี',
            'pt': 'โปรตุเกส', 'portuguese': 'โปรตุเกส',
            'ru': 'รัสเซีย', 'russian': 'รัสเซีย',
            'ar': 'อาหรับ', 'arabic': 'อาหรับ',
            'hi': 'ฮินดี', 'hindi': 'ฮินดี',
            'vi': 'เวียดนาม', 'vietnamese': 'เวียดนาม'
        }
        
        # Setup prompts
        self._setup_prompts()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_prompts(self):
        """Setup various prompt templates"""
        
        # Main translation prompt
        self.translation_prompt = PromptTemplate(
            input_variables=["text", "source_lang", "target_lang"],
            template="""คุณเป็นนักแปลมืออาชีพ กรุณาแปลข้อความต่อไปนี้อย่างแม่นยำและเป็นธรรมชาติ

ข้อความต้นฉบับ ({source_lang}): {text}
แปลเป็น: {target_lang}

กรุณาตอบในรูปแบบ JSON:
{{
    "translated_text": "คำแปลที่แม่นยำ",
    "confidence": "สูง/กลาง/ต่ำ",
    "notes": "หมายเหตุเพิ่มเติม (ถ้ามี)"
}}

คำตอบ:"""
        )
        
        # Language detection prompt
        self.detection_prompt = PromptTemplate(
            input_variables=["text"],
            template="""ระบุภาษาของข้อความต่อไปนี้:
{text}

กรุณาตอบในรูปแบบ JSON:
{{
    "language": "ชื่อภาษา",
    "language_code": "รหัสภาษา (เช่น th, en, zh)",
    "confidence": "สูง/กลาง/ต่ำ"
}}

คำตอบ:"""
        )
        
        # Context-aware translation prompt
        self.context_prompt = PromptTemplate(
            input_variables=["text", "target_lang", "context"],
            template="""แปลข้อความต่อไปนี้เป็นภาษา{target_lang} โดยคำนึงถึงบริบท:

ข้อความ: {text}
บริบท: {context}

กรุณาแปลให้เหมาะสมกับบริบทและใช้ภาษาที่เป็นธรรมชาติ

คำแปล:"""
        )
    
    def detect_language(self, text: str) -> Dict:
        """
        Detect the language of input text
        
        Args:
            text: Text to detect language for
            
        Returns:
            Dictionary with detection results
        """
        try:
            chain = self.detection_prompt | self.llm
            result = chain.invoke({"text": text.strip()})
            
            # Try to parse JSON response
            parsed = self._parse_json_response(result)
            if parsed:
                return parsed
            
            # Fallback: simple detection based on character patterns
            return self._simple_language_detection(text)
            
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            return {
                "language": "ไม่ระบุ",
                "language_code": "unknown",
                "confidence": "ต่ำ",
                "error": str(e)
            }
    
    def _simple_language_detection(self, text: str) -> Dict:
        """Simple rule-based language detection fallback"""
        text = text.strip()
        
        # Thai characters
        if re.search(r'[ก-๙]', text):
            return {"language": "ไทย", "language_code": "th", "confidence": "สูง"}
        
        # Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return {"language": "จีน", "language_code": "zh", "confidence": "กลาง"}
        
        # Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return {"language": "ญี่ปุ่น", "language_code": "ja", "confidence": "กลาง"}
        
        # Korean characters
        if re.search(r'[\uac00-\ud7af]', text):
            return {"language": "เกาหลี", "language_code": "ko", "confidence": "กลาง"}
        
        # Arabic characters
        if re.search(r'[\u0600-\u06ff]', text):
            return {"language": "อาหรับ", "language_code": "ar", "confidence": "กลาง"}
        
        # Default to English for Latin characters
        if re.search(r'[a-zA-Z]', text):
            return {"language": "อังกฤษ", "language_code": "en", "confidence": "กลาง"}
        
        return {"language": "ไม่ระบุ", "language_code": "unknown", "confidence": "ต่ำ"}
    
    def translate(self, 
                 text: str, 
                 target_language: str = "en", 
                 source_language: Optional[str] = None,
                 auto_detect: bool = True) -> TranslationResult:
        """
        Translate text with enhanced features
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            source_language: Source language (optional, will auto-detect if None)
            auto_detect: Whether to auto-detect source language
            
        Returns:
            TranslationResult object
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language="",
                target_language=target_language,
                confidence="",
                error="ข้อความว่างเปล่า"
            )
        
        try:
            # Auto-detect source language if not provided
            detected_lang = None
            if auto_detect and not source_language:
                detection = self.detect_language(text)
                detected_lang = detection.get('language', 'ไม่ระบุ')
                source_language = detected_lang
            
            # Normalize language names
            source_lang = self._normalize_language(source_language) if source_language else "ไม่ระบุ"
            target_lang = self._normalize_language(target_language)
            
            # Perform translation
            chain = self.translation_prompt | self.llm
            result = chain.invoke({
                "text": text.strip(),
                "source_lang": source_lang,
                "target_lang": target_lang
            })
            
            # Parse result
            parsed = self._parse_json_response(result)
            if parsed and 'translated_text' in parsed:
                return TranslationResult(
                    original_text=text,
                    translated_text=parsed['translated_text'],
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence=parsed.get('confidence', 'กลาง'),
                    detected_language=detected_lang
                )
            else:
                # Fallback: use raw result
                return TranslationResult(
                    original_text=text,
                    translated_text=result.strip(),
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence='กลาง',
                    detected_language=detected_lang
                )
                
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language=source_language or "",
                target_language=target_language,
                confidence="",
                error=f"เกิดข้อผิดพลาด: {str(e)}"
            )
    
    def translate_with_context(self, text: str, target_language: str, context: str) -> str:
        """
        Translate with additional context for better accuracy
        
        Args:
            text: Text to translate
            target_language: Target language
            context: Additional context information
            
        Returns:
            Translated text
        """
        try:
            target_lang = self._normalize_language(target_language)
            chain = self.context_prompt | self.llm
            result = chain.invoke({
                "text": text.strip(),
                "target_lang": target_lang,
                "context": context.strip()
            })
            return result.strip()
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def batch_translate(self, 
                       texts: List[str], 
                       target_language: str = "en",
                       source_language: Optional[str] = None) -> List[TranslationResult]:
        """
        Translate multiple texts at once
        
        Args:
            texts: List of texts to translate
            target_language: Target language
            source_language: Source language (optional)
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Translating text {i+1}/{len(texts)}")
            result = self.translate(text, target_language, source_language)
            results.append(result)
        return results
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code/name to Thai name"""
        if not language:
            return "ไม่ระบุ"
        
        lang_lower = language.lower().strip()
        return self.language_codes.get(lang_lower, language)
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.language_codes.copy()
    
    def translation_stats(self, results: List[TranslationResult]) -> Dict:
        """Get statistics from translation results"""
        if not results:
            return {}
        
        stats = {
            "total_translations": len(results),
            "successful": len([r for r in results if not r.error]),
            "failed": len([r for r in results if r.error]),
            "source_languages": {},
            "target_languages": {},
            "confidence_levels": {}
        }
        
        for result in results:
            if not result.error:
                # Count source languages
                src_lang = result.source_language
                stats["source_languages"][src_lang] = stats["source_languages"].get(src_lang, 0) + 1
                
                # Count target languages
                tgt_lang = result.target_language
                stats["target_languages"][tgt_lang] = stats["target_languages"].get(tgt_lang, 0) + 1
                
                # Count confidence levels
                conf = result.confidence
                stats["confidence_levels"][conf] = stats["confidence_levels"].get(conf, 0) + 1
        
        return stats

def demo_translation():
    """Comprehensive demonstration of the translation agent"""
    print("=== Enhanced Translation Agent Demo ===\n")
    
    agent = EnhancedTranslationAgent()
    
    # Test texts in various languages
    test_texts = [
        ("สวัสดีครับ ยินดีต้อนรับเข้าสู่ระบบแปลภาษา", "en"),
        ("Hello, welcome to our translation system", "th"),
        ("Bonjour, comment allez-vous?", "th"),
        ("こんにちは、元気ですか？", "th"),
        ("안녕하세요, 어떻게 지내세요?", "en"),
        ("Hola, ¿cómo estás?", "th")
    ]
    
    print("--- การทดสอบการแปลพื้นฐาน ---")
    results = []
    
    for i, (text, target_lang) in enumerate(test_texts, 1):
        print(f"\nข้อความที่ {i}:")
        print(f"ต้นฉบับ: {text}")
        
        # Detect language first
        detection = agent.detect_language(text)
        print(f"ภาษาที่ตรวจพบ: {detection.get('language', 'ไม่ระบุ')} (ความมั่นใจ: {detection.get('confidence', 'ไม่ระบุ')})")
        
        # Translate
        result = agent.translate(text, target_lang)
        results.append(result)
        
        if result.error:
            print(f"ข้อผิดพลาด: {result.error}")
        else:
            print(f"คำแปล ({result.target_language}): {result.translated_text}")
            print(f"ความมั่นใจ: {result.confidence}")
        
        print("-" * 50)
    
    # Show statistics
    print("\n--- สถิติการแปล ---")
    stats = agent.translation_stats(results)
    print(f"แปลทั้งหมด: {stats.get('total_translations', 0)}")
    print(f"สำเร็จ: {stats.get('successful', 0)}")
    print(f"ล้มเหลว: {stats.get('failed', 0)}")
    
    if stats.get('source_languages'):
        print("ภาษาต้นทาง:", stats['source_languages'])
    
    # Test context-aware translation
    print("\n--- การแปลตามบริบท ---")
    context_text = "Bank"
    contexts = [
        "ธนาคารและการเงิน",
        "ริมฝั่งแม่น้ำ",
        "การเล่นเกม"
    ]
    
    for context in contexts:
        translation = agent.translate_with_context(context_text, "th", context)
        print(f"'{context_text}' ในบริบท '{context}': {translation}")
    
    # Show supported languages
    print(f"\n--- ภาษาที่รองรับ ({len(agent.get_supported_languages())} ภาษา) ---")
    langs = agent.get_supported_languages()
    for code, name in list(langs.items())[:10]:  # Show first 10
        print(f"{code}: {name}")
    print("...")

if __name__ == "__main__":
    demo_translation()