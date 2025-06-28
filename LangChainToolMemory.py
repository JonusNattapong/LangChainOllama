from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

def get_exchange_rate(_: str) -> str:
    return "อัตราแลกเปลี่ยนวันนี้คือ 36.50 บาท/ดอลลาร์"

# สร้าง Tool เดียวที่จัดการได้หลายชื่อ
exchange_tool = Tool(
    name="exchange_rate",
    func=get_exchange_rate,
    description="ใช้ดูอัตราแลกเปลี่ยนเงินบาทกับดอลลาร์ สามารถเรียกใช้ได้หลายชื่อ เช่น exchange_rate, อัตราแลกเปลี่ยน, ตรวจสอบอัตราแลกเปลี่ยน"
)

tools = [exchange_tool]

# สร้าง LLM และ Memory
llm = OllamaLLM(model="llama3.2:3b")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# สร้าง Agent ที่มีประสิทธิภาพดีขึ้น
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    max_iterations=3,  # จำกัดรอบการคิด
    handle_parsing_errors=True,  # จัดการ parsing error
    early_stopping_method="generate"  # หยุดเร็วขึ้นเมื่อได้คำตอบ
)

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    try:
        response = agent_executor.invoke("ตอนนี้อัตราแลกเปลี่ยนเป็นเท่าไร?")
        print("ผลลัพธ์:", response.get("output", response))
        
        # ทดสอบ Memory
        response2 = agent_executor.invoke("ขอบคุณสำหรับข้อมูล")
        print("ผลลัพธ์ที่ 2:", response2.get("output", response2))
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).exception("เกิดข้อผิดพลาด")