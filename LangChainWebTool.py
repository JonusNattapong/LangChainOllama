# ตัวอย่าง LangChain Agent + Web Tool (Search API)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM

# 1. สร้าง Search Tool
search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="ใช้ค้นหาข้อมูลล่าสุดจากอินเทอร์เน็ตด้วย DuckDuckGo"
)

# 2. สร้าง LLM (Ollama)
llm = OllamaLLM(model="llama3.2:3b")

# 3. สร้าง Agent ที่มี Web Search Tool
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 4. ตัวอย่างการถามข้อมูล
query = "ข่าวเทคโนโลยีล่าสุดวันนี้คืออะไร?"
response = agent.invoke(query)
print(response)
