# ตัวอย่าง LangChain Agent + Web Tool (Search API)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM

# 1. สร้าง Search Tool
search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="ใช้ค้นหาข้อมูลล่าสุดจากอินเทอร์เน็ตด้วย DuckDuckGo (ผลลัพธ์จะสรุปเป็น Final Answer อัตโนมัติ)"
)

# 2. สร้าง LLM (Ollama)
llm = OllamaLLM(model="llama3.2:3b")

# 3. สร้าง Agent ที่มี Web Search Tool
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=3  # จำกัดรอบการคิด ไม่ให้วน loop
)

# 4. ตัวอย่างการถามข้อมูล
query = "ข่าวเทคโนโลยีล่าสุดวันนี้คืออะไร?"
response = agent.invoke(query)
print(response)

def search_with_final_answer(query):
    result = search_tool.run(query)
    return f"Final Answer: {result}"

# ปรับ Tool ให้ใช้ฟังก์ชันใหม่
search_tool = Tool(
    name="search",
    func=search_with_final_answer,
    description="ใช้ค้นหาข้อมูลล่าสุดจากอินเทอร์เน็ตด้วย DuckDuckGo (ผลลัพธ์จะสรุปเป็น Final Answer อัตโนมัติ)"
)
