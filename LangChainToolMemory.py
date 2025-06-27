from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

def get_exchange_rate(_):
    return "อัตราแลกเปลี่ยนวันนี้คือ 36.50 บาท/ดอลลาร์"

tools = [
    Tool(
        name="exchange_rate",  # ✅ ใช้ชื่อให้สั้นและตรง
        func=get_exchange_rate,
        description="ใช้ดูอัตราแลกเปลี่ยนเงินบาทกับดอลลาร์ สามารถเรียกใช้ด้วยคำว่า 'exchange_rate' หรือ 'ตรวจสอบอัตราแลกเปลี่ยน' หรือ 'อัตราแลกเปลี่ยน' หรือ 'ดูอัตราแลกเปลี่ยน'"
    ),
    Tool(
        name="ตรวจสอบอัตราแลกเปลี่ยน",
        func=get_exchange_rate,
        description="ใช้ดูอัตราแลกเปลี่ยนเงินบาทกับดอลลาร์ (ชื่อภาษาไทย)"
    ),
    Tool(
        name="อัตราแลกเปลี่ยน",
        func=get_exchange_rate,
        description="ใช้ดูอัตราแลกเปลี่ยนเงินบาทกับดอลลาร์ (ชื่อภาษาไทย)"
    )
]

llm = OllamaLLM(model="llama3.2:3b")

memory = ConversationBufferMemory(memory_key="chat_history")

# สร้าง AgentExecutor แบบควบคุมได้
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

response = agent_executor.invoke("ตอนนี้อัตราแลกเปลี่ยนเป็นเท่าไร?")
print(response)