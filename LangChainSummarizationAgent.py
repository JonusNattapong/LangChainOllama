# Enhanced LangChain Workflow: Advanced Summarization Agent
# สรุปเนื้อหาจากไฟล์หรือข้อความยาว ๆ ด้วย LLM + Prompt Engineering

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
import time
from typing import Dict, List, Any, Optional
from enum import Enum
import json

class SummaryType(Enum):
    """ประเภทการสรุป"""
    BRIEF = "brief"           # สรุปสั้น
    DETAILED = "detailed"     # สรุปละเอียด
    BULLET_POINTS = "bullets" # สรุปแบบหัวข้อ
    KEYWORDS = "keywords"     # คำสำคัญ
    ABSTRACT = "abstract"     # บทคัดย่อ

class SummarizationAgent:
    def __init__(self, model_name: str = "llama3.2:3b", chunk_size: int = 2000):
        """
        Initialize Enhanced Summarization Agent
        
        Args:
            model_name: Ollama model name
            chunk_size: Size of text chunks for long documents
        """
        self.llm = OllamaLLM(model=model_name, temperature=0.3)
        self.chunk_size = chunk_size
        
        # Text splitter for long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize prompts for different summary types
        self.prompts = self._create_prompts()

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

    def _create_prompts(self) -> Dict[SummaryType, PromptTemplate]:
        """Create different prompt templates for various summary types"""
        prompts = {}
        
        # Brief summary
        prompts[SummaryType.BRIEF] = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสรุปเนื้อหา

โปรดสรุปเนื้อหาต่อไปนี้ให้กระชับในย่อหน้าเดียว (ไม่เกิน 100 คำ):

{content}

สรุปสั้น:"""
        )
        
        # Detailed summary
        prompts[SummaryType.DETAILED] = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสรุปเนื้อหา

โปรดสรุปเนื้อหาต่อไปนี้อย่างละเอียด โดยรวมประเด็นสำคัญทั้งหมด:

{content}

สรุปละเอียด:"""
        )
        
        # Bullet points summary
        prompts[SummaryType.BULLET_POINTS] = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสรุปเนื้อหา

โปรดสรุปเนื้อหาต่อไปนี้ในรูปแบบหัวข้อย่อย:

{content}

สรุปแบบหัวข้อ:
•"""
        )
        
        # Keywords extraction
        prompts[SummaryType.KEYWORDS] = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการสกัดคำสำคัญ

โปรดหาคำสำคัญจากเนื้อหาต่อไปนี้ (ไม่เกิน 10 คำ):

{content}

คำสำคัญ:"""
        )
        
        # Abstract
        prompts[SummaryType.ABSTRACT] = PromptTemplate(
            input_variables=["content"],
            template="""คุณเป็นผู้เชี่ยวชาญในการเขียนบทคัดย่อ

โปรดเขียนบทคัดย่อจากเนื้อหาต่อไปนี้ในรูปแบบที่เป็นทางการ:

{content}

บทคัดย่อ:"""
        )
        
        return prompts

    def summarize(self, content: str, summary_type: SummaryType = SummaryType.BRIEF) -> Dict[str, Any]:
        """
        Summarize content with specified type
        
        Args:
            content: Text content to summarize
            summary_type: Type of summary to generate
            
        Returns:
            Dictionary containing summary result and metadata
        """
        if not content or not content.strip():
            return {
                "success": False,
                "error": "เนื้อหาว่างหรือไม่ถูกต้อง",
                "content_length": 0,
                "processing_time": 0
            }
        
        start_time = time.time()
        
        try:
            # Check if content is too long and needs chunking
            if len(content) > self.chunk_size:
                return self._summarize_long_content(content, summary_type)
            
            # Use appropriate prompt for summary type
            prompt = self.prompts.get(summary_type, self.prompts[SummaryType.BRIEF])
            chain = prompt | self.llm
            
            result = chain.invoke({"content": content})
            processing_time = time.time() - start_time
            
            self.logger.info(f"Summary completed: {len(content)} chars -> {len(result)} chars in {processing_time:.2f}s")
            
            return {
                "success": True,
                "summary": result,
                "summary_type": summary_type.value,
                "content_length": len(content),
                "summary_length": len(result),
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in summarization: {e}")
            
            return {
                "success": False,
                "error": f"เกิดข้อผิดพลาด: {str(e)}",
                "content_length": len(content),
                "processing_time": processing_time
            }

    def _summarize_long_content(self, content: str, summary_type: SummaryType) -> Dict[str, Any]:
        """Handle summarization of long content using chunking"""
        self.logger.info(f"Processing long content ({len(content)} chars) with chunking")
        
        # Split content into chunks
        chunks = self.text_splitter.split_text(content)
        chunk_summaries = []
        
        # Summarize each chunk
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_result = self.summarize(chunk, summary_type)
            
            if chunk_result['success']:
                chunk_summaries.append(chunk_result['summary'])
            else:
                self.logger.warning(f"Failed to summarize chunk {i+1}")
        
        # Combine chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        
        # Final summarization of combined summaries
        if len(combined_summary) > self.chunk_size:
            final_result = self.summarize(combined_summary, summary_type)
            final_summary = final_result.get('summary', combined_summary)
        else:
            final_summary = combined_summary
        
        return {
            "success": True,
            "summary": final_summary,
            "summary_type": summary_type.value,
            "content_length": len(content),
            "summary_length": len(final_summary),
            "chunks_processed": len(chunks),
            "processing_time": time.time(),
            "timestamp": time.time()
        }

    def summarize_from_file(self, file_path: str, summary_type: SummaryType = SummaryType.BRIEF, encoding: str = "utf-8") -> Dict[str, Any]:
        """Summarize content from file"""
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"ไม่พบไฟล์: {file_path}",
                    "file_path": file_path
                }
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            result = self.summarize(content, summary_type)
            result["file_path"] = file_path
            result["file_size"] = os.path.getsize(file_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return {
                "success": False,
                "error": f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}",
                "file_path": file_path
            }

    def batch_summarize(self, contents: List[str], summary_type: SummaryType = SummaryType.BRIEF) -> List[Dict[str, Any]]:
        """Summarize multiple contents"""
        if not isinstance(contents, list) or not contents:
            return []
        
        self.logger.info(f"Starting batch summarization for {len(contents)} items")
        results = []
        
        for i, content in enumerate(contents):
            self.logger.info(f"Processing item {i+1}/{len(contents)}")
            result = self.summarize(content, summary_type)
            result["batch_index"] = i
            results.append(result)
        
        return results

    def multi_type_summary(self, content: str) -> Dict[SummaryType, Dict[str, Any]]:
        """Generate multiple types of summaries for the same content"""
        results = {}
        
        for summary_type in SummaryType:
            self.logger.info(f"Generating {summary_type.value} summary")
            results[summary_type] = self.summarize(content, summary_type)
        
        return results

    def save_summary(self, result: Dict[str, Any], output_path: str, include_metadata: bool = True, encoding: str = "utf-8") -> bool:
        """Save summary result to file"""
        try:
            output_content = result.get('summary', '')
            
            if include_metadata and result.get('success'):
                metadata = {
                    "summary_type": result.get('summary_type'),
                    "content_length": result.get('content_length'),
                    "summary_length": result.get('summary_length'),
                    "processing_time": result.get('processing_time'),
                    "timestamp": result.get('timestamp')
                }
                
                output_content = f"=== METADATA ===\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n\n=== SUMMARY ===\n{output_content}"
            
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(output_content)
            
            self.logger.info(f"Summary saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
            return False

def interactive_demo():
    """Interactive demonstration"""
    print("=== Enhanced Summarization Agent ===")
    print("ตัวเลือก:")
    print("1. สรุปข้อความ")
    print("2. สรุปจากไฟล์")
    print("3. สรุปหลายรูปแบบ")
    print("4. ออก")
    
    agent = SummarizationAgent()
    
    while True:
        choice = input("\nเลือกตัวเลือก (1-4): ").strip()
        
        if choice == "1":
            content = input("ใส่ข้อความที่ต้องการสรุป: ")
            if content:
                print("\nประเภทการสรุป:")
                for i, summary_type in enumerate(SummaryType, 1):
                    print(f"{i}. {summary_type.value}")
                
                type_choice = input("เลือกประเภท (1-5): ").strip()
                try:
                    summary_type = list(SummaryType)[int(type_choice) - 1]
                    result = agent.summarize(content, summary_type)
                    
                    if result['success']:
                        print(f"\n📝 สรุป ({summary_type.value}):")
                        print(result['summary'])
                        print(f"\n📊 สถิติ: {result['content_length']} -> {result['summary_length']} ตัวอักษร")
                    else:
                        print(f"❌ {result['error']}")
                        
                except (ValueError, IndexError):
                    print("ตัวเลือกไม่ถูกต้อง")
        
        elif choice == "2":
            file_path = input("ระบุ path ของไฟล์: ")
            if file_path:
                result = agent.summarize_from_file(file_path)
                
                if result['success']:
                    print(f"\n📝 สรุปจากไฟล์:")
                    print(result['summary'])
                    
                    save_choice = input("\nต้องการบันทึกสรุปไหม? (y/n): ")
                    if save_choice.lower() == 'y':
                        output_path = input("ระบุ path สำหรับบันทึก: ")
                        if agent.save_summary(result, output_path):
                            print("✅ บันทึกเรียบร้อย")
                else:
                    print(f"❌ {result['error']}")
        
        elif choice == "3":
            content = input("ใส่ข้อความที่ต้องการสรุปหลายรูปแบบ: ")
            if content:
                print("\n🔄 กำลังสร้างสรุปหลายรูปแบบ...")
                results = agent.multi_type_summary(content)
                
                for summary_type, result in results.items():
                    if result['success']:
                        print(f"\n📝 {summary_type.value.upper()}:")
                        print(result['summary'])
                        print("-" * 50)
        
        elif choice == "4":
            print("ขอบคุณที่ใช้งาน!")
            break
        
        else:
            print("ตัวเลือกไม่ถูกต้อง")

def demo_with_examples():
    """Demonstration with example content"""
    agent = SummarizationAgent()
    
    # Example long content
    sample_content = """
    ปัญญาประดิษฐ์ (Artificial Intelligence หรือ AI) เป็นเทคโนโลยีที่ได้รับความสนใจอย่างมากในยุคปัจจุบัน 
    เนื่องจากมีศักยภาพในการเปลี่ยนแปลงวิธีการทำงานและดำเนินชีวิตของมนุษย์อย่างมากมาย
    
    AI สามารถจำแนกออกได้เป็นหลายประเภท เช่น Machine Learning, Deep Learning, และ Natural Language Processing 
    แต่ละประเภทมีความสามารถและการใช้งานที่แตกต่างกันไป
    
    ในด้านการประยุกต์ใช้ AI ได้ถูกนำมาใช้ในหลายอุตสาหกรรม เช่น การแพทย์ การเงิน การขนส่ง และการศึกษา 
    ซึ่งช่วยเพิ่มประสิทธิภาพและลดข้อผิดพลาดจากมนุษย์
    
    อย่างไรก็ตาม AI ยังมีความท้าทายและข้อจำกัดที่ต้องแก้ไข เช่น ปัญหาด้านจริยธรรม ความเป็นส่วนตัว 
    และผลกระทบต่อการจ้างงาน ซึ่งต้องมีการพัฒนาและควบคุมอย่างรอบคอบ
    """
    
    print("=== Demo: Enhanced Summarization Agent ===")
    print("ตัวอย่างเนื้อหา:")
    print(sample_content[:200] + "...")
    print("\n" + "="*60)
    
    # Generate different types of summaries
    for summary_type in SummaryType:
        print(f"\n📝 {summary_type.value.upper()}:")
        result = agent.summarize(sample_content, summary_type)
        
        if result['success']:
            print(result['summary'])
            print(f"📊 {result['content_length']} -> {result['summary_length']} ตัวอักษร | เวลา: {result['processing_time']:.2f}s")
        else:
            print(f"❌ {result['error']}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("เลือกโหมด:")
    print("1. Demo ตัวอย่าง")
    print("2. โหมดโต้ตอบ")
    
    choice = input("เลือก (1-2): ").strip()
    
    if choice == "1":
        demo_with_examples()
    else:
        interactive_demo()