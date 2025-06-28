from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging

class SentimentAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""วิเคราะห์อารมณ์ของข้อความต่อไปนี้ (เชิงบวก/ลบ/กลาง พร้อมเหตุผล):
{text}
ผลวิเคราะห์:"""
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def analyze(self, text):
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"text": text})
            self.logger.info(f"Sentiment analysis completed for text: {text}")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return f"เกิดข้อผิดพลาด: {e}"

def interactive_mode():
    """โหมดโต้ตอบให้ผู้ใช้ใส่ข้อความเอง"""
    print("=== Thai Sentiment Analyzer ===")
    print("พิมพ์ 'quit' หรือ 'exit' เพื่อออก")
    print("-" * 40)
    
    agent = SentimentAgent()
    
    while True:
        # รับข้อความจากผู้ใช้
        user_input = input("\nใส่ข้อความที่ต้องการวิเคราะห์: ").strip()
        
        # ตรวจสอบการออกจากโปรแกรม
        if user_input.lower() in ['quit', 'exit', 'ออก', 'หยุด']:
            print("ขอบคุณที่ใช้งาน!")
            break
        
        # ตรวจสอบข้อความว่าง
        if not user_input:
            print("กรุณาใส่ข้อความที่ต้องการวิเคราะห์")
            continue
        
        # วิเคราะห์อารมณ์
        print("\n🔄 กำลังวิเคราะห์...")
        result = agent.analyze(user_input)
        
        print("\n📊 ผลการวิเคราะห์:")
        print("-" * 30)
        print(result)
        print("-" * 30)

def batch_from_file(filename):
    """อ่านข้อความจากไฟล์และวิเคราะห์ทั้งหมด"""
    try:
        agent = SentimentAgent()
        
        with open(filename, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        
        print(f"พบ {len(texts)} ข้อความในไฟล์")
        print("=" * 50)
        
        for i, text in enumerate(texts, 1):
            text = text.strip()
            if text:  # ข้ามบรรทัดว่าง
                print(f"\n{i}. ข้อความ: {text}")
                result = agent.analyze(text)
                print(f"   ผลลัพธ์: {result}")
                print("-" * 30)
                
    except FileNotFoundError:
        print(f"ไม่พบไฟล์ {filename}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")

def predefined_examples():
    """ตัวอย่างข้อความที่กำหนดไว้แล้ว"""
    agent = SentimentAgent()
    
    examples = [
        "วันนี้อากาศดีมากและฉันมีความสุข",
        "ฉันเศร้ามากวันนี้ เพราะสอบตก", 
        "อาหารนี้รสชาติธรรมดา ไม่ดีไม่เลว",
        "ขอบคุณมากครับ คุณช่วยฉันได้มาก",
        "ฉันโกรธมากที่ถูกหลอก",
        "ร้านนี้บริการแย่มาก ไม่แนะนำ",
        "ภาพยนตร์เรื่องนี้สนุกดี น่าดู",
        "งานนี้น่าเบื่อ ทำไม่เสร็จซักที"
    ]
    
    print("=== ตัวอย่างการวิเคราะห์อารมณ์ ===")
    
    for i, text in enumerate(examples, 1):
        print(f"\n{i}. ข้อความ: {text}")
        result = agent.analyze(text)
        print(f"   ผลลัพธ์: {result}")
        print("-" * 50)

if __name__ == "__main__":
    print("เลือกโหมดการใช้งาน:")
    print("1. โหมดโต้ตอบ (พิมพ์ข้อความเอง)")
    print("2. ตัวอย่างที่กำหนดไว้")
    print("3. อ่านจากไฟล์ (batch_texts.txt)")
    
    choice = input("\nเลือก (1-3): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        predefined_examples()
    elif choice == "3":
        filename = input("ชื่อไฟล์ (default: batch_texts.txt): ").strip()
        if not filename:
            filename = "batch_texts.txt"
        batch_from_file(filename)
    else:
        print("ตัวเลือกไม่ถูกต้อง กำลังใช้โหมดโต้ตอบ...")
        interactive_mode()