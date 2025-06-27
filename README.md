# LangChainOllama

ชุดตัวอย่าง LangChain + Ollama สำหรับงาน AI Agent ภาษาไทย/อังกฤษ

## Features
- รองรับ LLM จาก Ollama (เช่น llama3.2:3b, qwen3:1.7b)
- ตัวอย่าง Agent หลากหลายรูปแบบ:
  - Basic Prompt/LLM Pipeline (`LangChainBasic.py`)
  - RAG (Retrieval Augmented Generation) + Embedding (`LangChainRAG.py`)
  - SQL Agent (ถาม-ตอบกับฐานข้อมูล SQLite) (`LangChainSQL.py`)
  - Tool Agent + Memory (เช่น อัตราแลกเปลี่ยน) (`LangChainToolMemory.py`)
  - Web Tool Agent (ค้นหาข้อมูลสดจากอินเทอร์เน็ต) (`LangChainWebTool.py`)
  - Fact-checking/Claim Verification Agent (RAG + Critic) (`LangChainFactCheckAgent.py`)
  - Creative Agent (แต่งนิยาย/กลอน/บทพูด) (`LangChainCreativeAgent.py`)

## โครงสร้างไฟล์หลัก
- `LangChainBasic.py` : ตัวอย่าง LLM + Prompt แบบง่าย
- `LangChainRAG.py` : ตัวอย่าง RAG + FAISS + HuggingFace Embedding
- `LangChainSQL.py` : Agent เชื่อมต่อฐานข้อมูล SQL (SQLite)
- `LangChainToolMemory.py` : Agent ที่ใช้ Tool หลายภาษา + Memory
- `LangChainWebTool.py` : Agent ที่ใช้ DuckDuckGo Search Tool
- `LangChainFactCheckAgent.py` : Agent ตรวจสอบข้อเท็จจริง (Fact-check)
- `LangChainCreativeAgent.py` : Agent สร้างสรรค์ (นิยาย/กลอน/บทพูด)
- `docs/mydoc.txt` : ตัวอย่างเอกสารสำหรับ RAG

## วิธีติดตั้ง
1. ติดตั้ง Python >= 3.10
2. ติดตั้งแพ็กเกจที่จำเป็น:
   ```sh
   pip install langchain langchain-community langchain-ollama langchain-huggingface duckduckgo-search sqlite-utils python-dotenv
   ```
3. ติดตั้ง Ollama และดาวน์โหลดโมเดลที่ต้องการ (เช่น llama3.2:3b)
4. สร้างไฟล์ `.env` (ถ้าใช้ HuggingFace Embedding):
   ```env
   HF_TOKEN=your_huggingface_token
   ```

## เทคนิค/รายละเอียดสำคัญ
- ใช้ LLM จาก Ollama ผ่าน `langchain_ollama.OllamaLLM` (รองรับ local LLM)
- ตัวอย่าง RAG ใช้ FAISS + HuggingFaceEmbeddings (รองรับ multilingual)
- SQL Agent ใช้ `langchain_community.agent_toolkits.create_sql_agent` เชื่อมต่อ SQLite
- Web Tool ใช้ DuckDuckGoSearchRun (DuckDuckGo อาจ rate limit ถ้าเรียกบ่อย)
- Fact-check Agent ใช้ 2 LLM: ตัวแรกค้นหา, ตัวที่สองวิจารณ์/ตรวจสอบ
- Creative Agent ใช้ Prompt Chaining (สร้างเรื่อง + แปลง style)
- ตัวอย่างทั้งหมดรองรับภาษาไทยและอังกฤษ

## หมายเหตุ
- หาก DuckDuckGo Search โดน rate limit ให้รอหรือเปลี่ยน API
- สามารถปรับแต่ง prompt, chain, หรือ agent ได้ตามต้องการ

---

> ตัวอย่างนี้เหมาะสำหรับนักพัฒนา AI, นักวิจัย, นักเขียน, นักวิเคราะห์ข่าว และผู้สนใจ LangChain/Ollama