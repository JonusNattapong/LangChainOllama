# ตัวอย่าง LangChain Agent + SQL (SQLite)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama langchain-sqlite sqlite-utils

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
import sqlite3
import os

def setup_database():
    try:
        conn = sqlite3.connect('example.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            email TEXT UNIQUE
        )''')
        cursor.execute("DELETE FROM users")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")
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

# 2. สร้าง SQLDatabase object
sql_db = SQLDatabase.from_uri('sqlite:///example.db')

# 3. สร้าง LLM (Ollama)
llm = OllamaLLM(model="llama3.2:3b")

# 4. สร้าง SQL Agent ที่มีประสิทธิภาพดีขึ้น
agent_executor = create_sql_agent(
    llm=llm,
    db=sql_db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,  # เพิ่มจำนวนรอบสำหรับ SQL ที่ซับซ้อน
    max_execution_time=30,  # หมดเวลา 30 วินาที
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
