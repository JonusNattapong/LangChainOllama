# ตัวอย่าง LangChain Agent + Web Tool (Search API)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM

def search_with_final_answer(query):
    """ฟังก์ชันค้นหาที่จัดรูปแบบผลลัพธ์เป็น Final Answer"""
    try:
        search_tool = DuckDuckGoSearchRun(backend="api")
        result = search_tool.run(query)
        return f"Final Answer: {result}"
    except Exception as e:
        # Fallback ไปใช้ backend="html" ถ้า api ไม่ได้
        try:
            search_tool_html = DuckDuckGoSearchRun(backend="html")
            result = search_tool_html.run(query)
            return f"Final Answer: {result}"
        except Exception as e2:
            return f"Final Answer: ไม่สามารถค้นหาข้อมูลได้ในขณะนี้ เนื่องจาก {str(e2)}"

# 1. สร้าง Search Tool พร้อม error handling
search_tool = Tool(
    name="search",
    func=search_with_final_answer,
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
    max_iterations=3,  # จำกัดรอบการคิด ไม่ให้วน loop
    handle_parsing_errors=True  # จัดการ parsing error
)

# 4. ตัวอย่างการใช้งาน
if __name__ == "__main__":
    query = "ข่าวเทคโนโลยีล่าสุดวันนี้คืออะไร?"
    try:
        response = agent.invoke(query)
        print("ผลลัพธ์:", response.get("output", response))
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
