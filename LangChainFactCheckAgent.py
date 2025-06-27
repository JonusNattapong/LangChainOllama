# LangChain Fact-checking / Claim Verification Agent (RAG + Critic)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. สร้าง Search Tool สำหรับค้นหาข้อมูลจริง
search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="ใช้ค้นหาข้อมูลจากอินเทอร์เน็ตเพื่อ cross-check ความถูกต้องของข้ออ้าง/คำกล่าวอ้าง"
)

# 2. LLM หลักและ LLM นักวิจารณ์ (Critic)
llm = OllamaLLM(model="llama3.2:3b")
critic_llm = OllamaLLM(model="qwen3:1.7b")  # สามารถใช้ model เดียวกันหรือแยกก็ได้

# 3. Prompt สำหรับ Critic Agent
critic_prompt = PromptTemplate(
    input_variables=["claim", "evidence"],
    template="""
คุณได้รับข้ออ้างดังนี้: "{claim}"
และข้อมูลอ้างอิงจากแหล่งค้นหาดังนี้:
{evidence}
โปรดตรวจสอบว่าข้ออ้างนี้เป็นจริงหรือเท็จ (Fact-check) พร้อมอธิบายเหตุผลและอ้างอิงแหล่งข้อมูล
ตอบเป็นภาษาไทย
"""
)
critic_chain = LLMChain(llm=critic_llm, prompt=critic_prompt)

# 4. Agent สำหรับรับ claim และค้นหาหลักฐาน
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

def fact_checking_agent(claim):
    # 1. ใช้ agent ค้นหาหลักฐานจากเว็บ
    evidence = agent.invoke(f"ค้นหาข้อมูลเกี่ยวกับ: {claim}")
    # 2. ส่ง claim + evidence ให้ Critic LLM วิเคราะห์
    result = critic_chain.invoke({"claim": claim, "evidence": evidence})
    return result["text"]

# ตัวอย่างการใช้งาน
claim = "ประเทศไทยมีประชากรมากกว่า 100 ล้านคน"
result = fact_checking_agent(claim)
print("ผลการตรวจสอบ:")
print(result)
