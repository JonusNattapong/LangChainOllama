# ตัวอย่าง LangChain Agent + SQL (SQLite)
# ติดตั้ง: pip install langchain langchain-community langchain-ollama langchain-sqlite sqlite-utils

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
import sqlite3
import os

# 1. สร้าง/เชื่อมต่อฐานข้อมูล SQLite และสร้างตารางตัวอย่าง
conn = sqlite3.connect('example.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER
)''')
cursor.execute("DELETE FROM users")  # ลบข้อมูลเดิมออกก่อน
cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")  # รีเซ็ต id ให้นับใหม่
cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")
cursor.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)")
conn.commit()

# 2. สร้าง SQLDatabase object
sql_db = SQLDatabase.from_uri('sqlite:///example.db')

# 3. สร้าง LLM (Ollama)
llm = OllamaLLM(model="llama3.2:3b")

# 4. สร้าง Agent ที่เชื่อมต่อ SQL
agent_executor = create_sql_agent(
    llm=llm,
    db=sql_db,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True  # เพิ่มบรรทัดนี้เพื่อให้ agent ไม่ล้มเมื่อเจอ output parsing error
)

# 5. ตัวอย่างการถามข้อมูล
query = "มีใครบ้างในฐานข้อมูล และอายุเท่าไร?"
response = agent_executor.invoke({"input": query})
print(response["output"])
