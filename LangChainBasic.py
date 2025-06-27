from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # ← ใช้จากแพ็กเกจใหม่
from langchain_core.runnables import RunnableSequence

# โมเดลจาก Ollama
llm = OllamaLLM(model="llama3.2:3b")  # หรือ "llama3.2:3b" ได้เช่นกัน

# Prompt แบบใหม่
prompt = PromptTemplate.from_template(
    "อธิบายเกี่ยวกับหัวข้อ: {topic} ให้เข้าใจง่าย"
)

# สร้าง pipeline แบบใหม่ (prompt | llm)
chain = prompt | llm

# เรียกใช้งานด้วย invoke แทน run
result = chain.invoke({"topic": "กลศาสตร์ควอนตัม"})
print(result)