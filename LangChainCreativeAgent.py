# LangChain Creative Agent (Writing + Story + Comic)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# 1. LLM สำหรับสร้างสรรค์
llm = OllamaLLM(model="llama3.2:3b")

# 2. Prompt สำหรับแต่งเรื่องสั้น/นิยาย
story_prompt = PromptTemplate(
    input_variables=["idea"],
    template="""
จงแต่งเรื่องสั้นแนวแฟนตาซีจากไอเดียต่อไปนี้:
"{idea}"
ขอให้เนื้อเรื่องมีจุดหักมุมและน่าติดตาม
"""
)

# 3. Prompt สำหรับแปลงสไตล์ (Style Transfer) เช่น ให้เป็นกลอน หรือบทพูดตลก
style_prompt = PromptTemplate(
    input_variables=["story"],
    template="""
จงแปลงเรื่องสั้นต่อไปนี้ให้เป็นกลอนสุภาพ 4 บท:
{story}
"""
)

# 4. สร้าง Chain
story_chain = LLMChain(llm=llm, prompt=story_prompt)
style_chain = LLMChain(llm=llm, prompt=style_prompt)
creative_chain = SimpleSequentialChain(chains=[story_chain, style_chain])

# 5. ตัวอย่างการใช้งาน
idea = "เด็กชายคนหนึ่งพบไข่มังกรในป่าและต้องปกป้องมันจากผู้ล่า"
result = creative_chain.invoke(idea)
print("ผลงานสร้างสรรค์:\n")
print(result)
