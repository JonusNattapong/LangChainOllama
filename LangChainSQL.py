# ตัวอย่าง LangChain Agent + SQL (SQLite)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama langchain-sqlite sqlite-utils

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
import sqlite3
import os

def setup_database():
    try:
        # ลบไฟล์ฐานข้อมูลเดิม (ถ้ามี) เพื่อป้องกัน schema เก่า
        if os.path.exists('example.db'):
            os.remove('example.db')
        conn = sqlite3.connect('example.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT UNIQUE
        )''')
        sample_data = [
            ('Alice', 30, 'alice@example.com'),
            ('Bob', 25, 'bob@example.com'),
            ('Charlie', 35, 'charlie@example.com')
        ]
        cursor.executemany("INSERT INTO users (name, age, email) VALUES (?, ?, ?)", sample_data)
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการตั้งค่าฐานข้อมูล")

# 1. ตั้งค่าฐานข้อมูล
setup_database()

# 2. สร้าง SQLDatabase object (เพิ่ม top_k และ include_tables)
sql_db = SQLDatabase.from_uri('sqlite:///example.db', include_tables=['users'], sample_rows_in_table_info=3)

# 3. สร้าง LLM (Ollama) พร้อม system prompt อธิบาย schema และข้อกำหนด output
llm = OllamaLLM(
    model="llama3.2:3b",
    system=(
        "คุณคือผู้ช่วย AI ที่ตอบคำถามเกี่ยวกับฐานข้อมูล SQLite ตาราง users (id, name, age, email) "
        "เวลาตอบ ให้ปฏิบัติตามรูปแบบของ LangChain Agent: ถ้าจะสรุป ให้ตอบ Final Answer เท่านั้น ไม่ต้องมี Action หรือ Thought ในรอบเดียวกัน "
        "ห้ามตอบ Final Answer พร้อมกับ Action หรือ Thought ในรอบเดียวกันเด็ดขาด ให้เลือกตอบอย่างใดอย่างหนึ่งเท่านั้นในแต่ละรอบ "
        "ตอบสั้น กระชับ และตรงประเด็น"
    )
)

# 4. สร้าง SQL Agent ที่มีประสิทธิภาพดีขึ้น
agent_executor = create_sql_agent(
    llm=llm,
    db=sql_db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # เพิ่มจำนวนรอบสำหรับ SQL ที่ซับซ้อน
    max_execution_time=60,  # หมดเวลา 60 วินาที
    return_intermediate_steps=True
)

# 5. ฟังก์ชันสำหรับถามข้อมูล
def query_database(question: str) -> str:
    try:
        if not isinstance(question, str) or not question.strip():
            return "คำถามว่างหรือไม่ถูกต้อง"
        response = agent_executor.invoke({"input": question})
        return response.get("output", "ไม่สามารถหาคำตอบได้")
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการ query database")
        return f"เกิดข้อผิดพลาด: {e}"

# 6. ตัวอย่างการใช้งาน
if __name__ == "__main__":
    questions = [
        "มีใครบ้างในฐานข้อมูล และอายุเท่าไร?",
        "ใครอายุมากที่สุด?",
        "มีกี่คนที่อายุมากกว่า 30 ปี?"
    ]
    for i, question in enumerate(questions, 1):
        print(f"\n=== คำถามที่ {i}: {question} ===")
        answer = query_database(question)
        print(f"คำตอบ: {answer}")
        print(f"คำตอบ: {answer}")
    
    for i, question in enumerate(questions, 1):
        print(f"\n=== คำถามที่ {i}: {question} ===")
        answer = query_database(question)
        print(f"คำตอบ: {answer}")
