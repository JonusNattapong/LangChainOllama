# LangChainOllama

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<div align="center">
  <img src="asset/images/logo1.png" alt="LangChainOllama Logo" width="500" />
</div>


ชุดตัวอย่าง LangChain + Ollama สำหรับงาน AI Agent ภาษาไทย/อังกฤษ

## Table of Contents
- [Features](#features)
- [ติดตั้ง](#วิธีติดตั้ง)
- [การใช้งานเบื้องต้น](#วิธีใช้งานเบื้องต้น)
- [ภาพรวมของระบบ](#ภาพรวมของระบบ)
- [Workflow ใหม่](#workflow-ใหม่)
- [Best Practices](#best-practices)

## Features
- รองรับ LLM จาก Ollama (เช่น llama3.2:3b, qwen3:1.7b)
- ตัวอย่าง Agent หลากหลายรูปแบบ (Basic, RAG, SQL, Tool+Memory, Web, Fact-check, Creative, Summarization, PDF Q&A, Multi-Document RAG, **Translation, Web Search**, **Knowledge Graph**, **Sentiment Analysis**, **Text Classification**, **Data Extraction**, **Code Generation**)
- ใช้ FAISS + HuggingFace Embeddings สำหรับ RAG
- Agent พร้อม memory, error handling, logging และ input validation
- รองรับภาษาไทยและอังกฤษ (Native)
- รองรับ batch/bulk operation ในหลาย agent (เช่น การแปล, การสรุป, การจัดประเภท, การสกัดข้อมูล)
- Knowledge Graph Triplet Extraction (สร้างกราฟความรู้จากข้อความ)
- Sentiment Analysis ภาษาไทย/อังกฤษ
- Text Classification ภาษาไทย/อังกฤษ
- Data Extraction (สกัดข้อมูลบริษัท/ข้อมูลสำคัญจากข้อความ)
- Code Generation Agent (สร้างโค้ด Python พร้อมอธิบายและรันได้)

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

## วิธีใช้งานเบื้องต้น
- ใช้ LLM จาก Ollama ผ่าน `langchain_ollama.OllamaLLM` (รองรับ local LLM)
- ตัวอย่าง RAG ใช้ FAISS + HuggingFaceEmbeddings (รองรับ multilingual)
- SQL Agent ใช้ `langchain_community.agent_toolkits.create_sql_agent` เชื่อมต่อ SQLite
- Web Tool ใช้ DuckDuckGoSearchRun (DuckDuckGo อาจ rate limit ถ้าเรียกบ่อย)
- Fact-check Agent ใช้ 2 LLM: ตัวแรกค้นหา, ตัวที่สองวิจารณ์/ตรวจสอบ
- Creative Agent ใช้ Prompt Chaining (สร้างเรื่อง + แปลง style)
- ตัวอย่างทั้งหมดรองรับภาษาไทยและอังกฤษ

## ภาพรวมของระบบ
<details>
<summary>คลิกเพื่อดูภาพรวม</summary>

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LangChain + Ollama Agent System                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │ Basic Agent │    │  RAG Agent  │    │ SQL Agent   │                  │
│  │             │    │             │    │             │                  │
│  │ • Chat      │    │ • Vector DB │    │ • SQLite    │                  │
│  │ • Q&A       │    │ • Documents │    │ • Queries   │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │ Web Agent   │    │Creative     │    │Fact-Check   │                  │
│  │             │    │Agent        │    │Agent        │                  │
│  │ • Search    │    │ • Stories   │    │ • Verify    │                  │
│  │ • Real-time │    │ • Poems     │    │ • Critics   │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐                                     │
│  │ Translation │    │ Web Tool    │                                     │
│  │ Agent       │    │ Agent       │                                     │
│  │ • Multilang │    │ • DuckDuckGo│                                     │
│  │ • Detect    │    │ • Fallback  │                                     │
│  └─────────────┘    └─────────────┘                                     │
│                                                                         │
│                         ┌─────────────────┐                             │
│                         │  Ollama LLM     │                             │
│                         │                 │                             │
│                         │ • llama3.2:3b   │                             │
│                         │ • qwen3:1.7b    │                             │
│                         │ • Local Models  │                             │
│                         └─────────────────┘                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┤
│  │                        Tools & Integration                          │
│  ├─────────────────────────────────────────────────────────────────────┤
│  │ Memory System │ Search Tools │ Database Tools │ File Processing     │
│  │               │              │                │                     │
│  │ • Chat History│ • DuckDuckGo │ • SQLite       │ • PDF               │
│  │ • Context     │ • Bing API   │ • PostgreSQL   │ • TXT               │
│  │ • Sessions    │ • Web Scrape │ • MySQL        │ • Markdown          │
│  └─────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

</details>

ระบบนี้แสดงให้เห็นถึงความสามารถหลากหลายของ LangChain เมื่อผสานกับ Ollama:

### 🤖 **Multi-Agent Architecture**
- **Basic Agent**: LLM พื้นฐานสำหรับการสนทนาและตอบคำถาม
- **RAG Agent**: ระบบค้นหาข้อมูลจากเอกสารด้วย Vector Database
- **SQL Agent**: Agent ที่สามารถสื่อสารกับฐานข้อมูลได้โดยตรง
- **Web Agent**: ค้นหาข้อมูลสดจากอินเทอร์เน็ตแบบ Real-time
- **Creative Agent**: สร้างสรรค์เนื้อหา เรื่องสั้น กลอน และบทความ
- **Fact-Check Agent**: ตรวจสอบความถูกต้องของข้อมูล
- **Summarization Agent**: สรุปเนื้อหายาว ๆ ให้กระชับ เข้าใจง่าย
- **PDF Q&A Agent**: ถามตอบข้อมูลจากไฟล์ PDF โดยตรง
- **Multi-Document RAG Agent**: ถามตอบจากหลายไฟล์/หลายแหล่งข้อมูล (TXT, PDF)
- **Translation Agent**: แปลข้อความและเอกสารระหว่างภาษาไทย อังกฤษ จีน ญี่ปุ่น ฯลฯ พร้อมตรวจจับภาษาอัตโนมัติ
- **Web Tool Agent**: ค้นหาข้อมูลสดจากอินเทอร์เน็ตแบบ Real-time ด้วย DuckDuckGo Search (รองรับ fallback หลายกลยุทธ์)
- **Knowledge Graph Agent**: สร้างกราฟความรู้จากข้อความ
- **Sentiment Analysis Agent**: วิเคราะห์อารมณ์ของข้อความ
- **Text Classification Agent**: จัดประเภทข้อความ
- **Data Extraction Agent**: สกัดข้อมูลสำคัญจากข้อความ
- **Code Generation Agent**: สร้างโค้ด Python พร้อมอธิบายและรันได้

### 🔧 **Tool Integration**
- **Memory System**: จดจำบริบทการสนทนาแบบต่อเนื่อง
- **Search Tools**: DuckDuckGo, Bing Search API integration
- **Database Tools**: SQLite, PostgreSQL, MySQL support
- **File Processing**: PDF, TXT, Markdown document processing

### 🌐 **Multi-Language Support**
- รองรับภาษาไทยและอังกฤษแบบ Native
- ใช้ Multilingual Embedding Models
- Prompt Engineering สำหรับภาษาไทยโดยเฉพาะ

---

> ตัวอย่างนี้เหมาะสำหรับนักพัฒนา AI, นักวิจัย, นักเขียน, นักวิเคราะห์ข่าว และผู้สนใจ LangChain/Ollama

<div align="center">
  <img src="asset/images/logo.png" alt="LangChainOllama Logo" width="200" />
</div>

## ตัวอย่างผลลัพธ์

### 📊 **SQL Agent ในการใช้งานจริง**
```python
# ตัวอย่างการถามข้อมูลจากฐานข้อมูล
query = "มีใครบ้างในฐานข้อมูล และอายุเท่าไร?"
response = agent_executor.invoke({"input": query})
# ผลลัพธ์: "ในฐานข้อมูลมี Alice อายุ 30 ปี และ Bob อายุ 25 ปี"
```

ตัวอย่างผลลัพธ์: SQL Agent สามารถตอบคำถามเกี่ยวกับข้อมูลในฐานข้อมูล SQLite ได้โดยตรง เช่น รายชื่อและอายุของบุคคลในฐานข้อมูล

### 🎨 **Creative Writing Agent**
```python
# สร้างเรื่องสั้นและแปลงเป็นกลอน
idea = "เด็กชายคนหนึ่งพบไข่มังกรในป่า"
result = creative_chain.invoke(idea)
# ผลลัพธ์: เรื่องสั้นแฟนตาซีและกลอนสุภาพ 4 บท
```

ตัวอย่างผลลัพธ์: Creative Agent สามารถสร้างเรื่องสั้นและแปลงเป็นกลอนหรือเนื้อหาสร้างสรรค์อื่น ๆ ได้ตามโจทย์ที่กำหนด

### 📚 **RAG System**
```python
# ตอบคำถามจากเอกสาร
query = "AI ช่วยงานอะไรได้บ้าง?"
response = rag_chain.invoke({"query": query})
# ผลลัพธ์: คำตอบที่อ้างอิงจากเอกสารในระบบ
```

ตัวอย่างผลลัพธ์: RAG System สามารถตอบคำถามโดยอ้างอิงข้อมูลจากเอกสารที่มีในระบบ

### 🎯 **Basic LLM Pipeline**
```python
# ตัวอย่างการใช้งาน LLM แบบพื้นฐาน
topic = "กลศาสตร์ควอนตัม"
result = chain.invoke({"topic": topic})
# ผลลัพธ์: คำอธิบายที่เข้าใจง่ายเกี่ยวกับหัวข้อที่ถาม
```

ตัวอย่างผลลัพธ์: Basic LLM Pipeline สามารถอธิบายหัวข้อหรือแนวคิดต่าง ๆ ได้อย่างเข้าใจง่าย

### 📖 **Translation Agent**
```python
# ตัวอย่างการแปลข้อความและตรวจจับภาษาอัตโนมัติ
agent = EnhancedTranslationAgent(model_name="llama3.2:3b")
result = agent.translate("สวัสดีครับ วันนี้อากาศดีมาก", target_language="en")
# ผลลัพธ์: TranslationResult(original_text='สวัสดีครับ วันนี้อากาศดีมาก', translated_text='Hello, the weather is very nice today.', ...)

# ตรวจจับภาษา
lang_info = agent.detect_language("こんにちは")
# ผลลัพธ์: {'language': 'ญี่ปุ่น', 'language_code': 'ja', 'confidence': 'กลาง'}
```

ตัวอย่างผลลัพธ์: Translation Agent สามารถแปลข้อความและตรวจจับภาษาได้อัตโนมัติ รองรับหลายภาษา

### 🌐 **Web Tool Agent**
```python
# ตัวอย่างการค้นหาข้อมูลสดจากอินเทอร์เน็ต
agent = create_search_agent(model="qwen3:1.7b")
response = agent.query("ข่าวเทคโนโลยี AI ล่าสุดวันนี้")
# ผลลัพธ์: {"success": True, "output": "Final Answer: ...", ...}

# รองรับ fallback หลายกลยุทธ์ (API, HTML, simplified query)
```

ตัวอย่างผลลัพธ์: Web Tool Agent สามารถค้นหาข้อมูลสดจากอินเทอร์เน็ตและรองรับ fallback หลายกลยุทธ์ เช่น API, HTML, simplified query

## ความพิเศษของ LangChain + Ollama

### 🚀 **Performance Benefits**
- **Local Processing**: ไม่ต้องส่งข้อมูลออกนอกเครื่อง
- **Cost Effective**: ไม่มีค่าใช้จ่าย API calls
- **Customizable**: ปรับแต่งโมเดลได้ตามต้องการ
- **Privacy**: ข้อมูลไม่ออกจากเครื่องของคุณ

### 💡 **Use Cases**
- **ผู้พัฒนา**: สร้าง AI Assistant สำหรับงานเขียนโค้ด
- **นักวิจัย**: วิเคราะห์เอกสารและข้อมูลขนาดใหญ่
- **นักเขียน**: ช่วยสร้างสรรค์เนื้อหาและตรวจสอบข้อเท็จจริง
- **นักธุรกิจ**: วิเคราะห์ข้อมูลและสร้างรายงาน

### 🔧 **Technical Advantages**
- **Modular Design**: แยกส่วนงานได้ชัดเจน
- **Extensible**: เพิ่ม Agent ใหม่ได้ง่าย
- **Multi-Modal**: รองรับข้อมูลหลายรูปแบบ
- **Production Ready**: พร้อมใช้งานจริง

## การติดตั้งและใช้งานขั้นสูง

### 🐳 **Docker Deployment**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "LangChainBasic.py"]
```

### ⚙️ **Environment Configuration**
```bash
# ตั้งค่าตัวแปรสิ่งแวดล้อม
export OLLAMA_HOST=localhost:11434
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
```

### 📊 **Performance Monitoring**
```python
# ตรวจสอบประสิทธิภาพ
from langchain.callbacks import get_openai_callback
with get_openai_callback() as cb:
    result = agent.invoke("your query")
    print(f"Tokens used: {cb.total_tokens}")
```

## 🗺️ ตัวอย่าง Workflow ของแต่ละ Agent (Mermaid Diagram)

> **หมายเหตุ:** Diagram เหล่านี้ช่วยให้มือใหม่เข้าใจลำดับการทำงานของแต่ละสคริปต์/Agent ได้ง่ายขึ้น สามารถนำไปปรับใช้หรือขยายต่อยอดได้

### 1. Basic Agent (`LangChainBasic.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[validate input]
    B --> C[เตรียม prompt]
    C --> D[โหลด LLM]
    D --> E[สร้าง chain]
    E --> F[ส่งเข้า LLM]
    F --> G[รับผลลัพธ์]
    G --> H[แสดงผล]
    B -. batch .-> I[วนลูปหลายหัวข้อ]
    I --> C
    H -. error .-> J[handle error]
```

**อธิบาย:**
- รับข้อความ/หัวข้อจากผู้ใช้ (ผ่าน CLI หรือฟังก์ชัน)
- ตรวจสอบ input, handle error, validate
- เตรียม prompt และ chain
- ส่งเข้า LLM, รับผลลัพธ์, แสดงผล
- รองรับ batch explain และ custom prompt

---

### 2. RAG Agent (`LangChainRAG.py`)

```mermaid
graph TD
    A[รับไฟล์/คำถาม] --> B[โหลดเอกสาร]
    B --> C[แบ่ง chunk]
    C --> D[สร้างเวกเตอร์]
    D --> E[ตั้งค่า QA]
    E --> F[รับคำถาม]
    F --> G[แปลงเป็นเวกเตอร์]
    G --> H[ค้น context]
    H --> I[รวม context]
    I --> J[เข้า LLM]
    J --> K[แสดงคำตอบ]
    B -. error .-> L[handle error]
```

**อธิบาย:**
- โหลดไฟล์, แบ่ง chunk, สร้างเวกเตอร์, ตั้งค่า chain
- ถาม > ค้น context > รวม > ส่งเข้า LLM > ตอบ

---

### 3. Multi-Document RAG Agent (`LangChainMultiDocRAGAgent.py`)

```mermaid
graph TD
    A[รับหลายไฟล์] --> B[โหลด+แบ่งเอกสาร]
    B --> C[สร้างเวกเตอร์]
    C --> D[ตั้งค่า QA]
    D --> E[รับคำถาม]
    E --> F[ค้น context]
    F --> G[รวม context]
    G --> H[เข้า LLM]
    H --> I[แสดงคำตอบ]
    B -. error .-> J[handle error]
```

**อธิบาย:**
- รองรับหลายไฟล์/หลายประเภท, batch, handle error

---

### 4. SQL Agent (`LangChainSQL.py`)

```mermaid
graph TD
    A[รับ SQL] --> B[วิเคราะห์ intent]
    B --> C[สร้าง query]
    C --> D[รัน SQL]
    D --> E[แปลงผลลัพธ์]
    E --> F[แสดงผล]
    F --> G[บันทึก history]
    F -. analytics .-> H[dashboard]
    F -. suggest .-> I[แนะนำ]
    A -. error .-> J[handle error]
```

**อธิบาย:**
- ถาม > วิเคราะห์ > สร้าง query > รัน > ตีความผลลัพธ์ > ตอบ
- รองรับ analytics, suggest, export history

---

### 5. Web Tool Agent (`LangChainWebTool.py`)

```mermaid
graph TD
    A[รับคำถาม] --> B[เลือก search tool]
    B --> C[ค้นเว็บ]
    C --> D[สรุปผล]
    D --> E[เข้า LLM]
    E --> F[แสดงผล]
    C -. fallback .-> G[fallback]
    A -. error .-> H[handle error]
```

**อธิบาย:**
- ถาม > ค้นเว็บ > สรุป > (ส่งเข้า LLM) > ตอบ
- มี fallback หลายกลยุทธ์

---

### 6. Translation Agent (`LangChainTranslationAgent.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[ตรวจจับภาษา]
    B --> C[เลือก LLM]
    C --> D[แปล]
    D --> E[แสดงผล]
    A -. batch .-> F[แปลหลายข้อความ]
    D -. context .-> G[แปลตามบริบท]
    A -. error .-> H[handle error]
```

**อธิบาย:**
- รับข้อความ > ตรวจจับภาษา > แปล > ตอบ
- รองรับ batch, context-aware, export stats

---

### 7. PDF Q&A Agent (`LangChainPDFQAAgent.py`)

```mermaid
graph TD
    A[รับ PDF+คำถาม] --> B[แปลง PDF]
    B --> C[แบ่ง chunk]
    C --> D[สร้างเวกเตอร์]
    D --> E[ตั้งค่า QA]
    E --> F[รับคำถาม]
    F --> G[ค้น context]
    G --> H[รวม context]
    H --> I[เข้า LLM]
    I --> J[แสดงคำตอบ]
    A -. error .-> K[handle error]
```

**อธิบาย:**
- รับ PDF > แปลง > chunk > vector > QA chain > ถาม LLM > ตอบ

---

### 8. Knowledge Graph Agent (`LangChainKnowledgeGraphAgent.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[สกัด triplet]
    B --> C[อัปเดตกราฟ]
    C --> D[รับคำถาม]
    D --> E[ตั้งค่า prompt]
    E --> F[แสดง node]
    A -. error .-> G[handle error]
```

**อธิบาย:**
- รับข้อความ > สกัด triplet > อัปเดตกราฟ > query node > ตอบ

---

### 9. Sentiment Analysis Agent (`LangChainSentimentAgent.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[เข้า LLM วิเคราะห์]
    B --> C[แสดงอารมณ์]
    A -. batch .-> D[วิเคราะห์หลายข้อความ]
    A -. error .-> E[handle error]
```

**อธิบาย:**
- รับข้อความ > วิเคราะห์อารมณ์ > ตอบ (รองรับ batch)

---

### 10. Text Classification Agent (`LangChainTextClassificationAgent.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[เข้า LLM จัดประเภท]
    B --> C[แสดงหมวดหมู่]
    A -. batch .-> D[จัดประเภทหลายข้อความ]
    C -. distribution .-> E[สรุป distribution]
    A -. error .-> F[handle error]
```

**อธิบาย:**
- รับข้อความ > จัดประเภท > ตอบ (รองรับ batch, สรุป distribution)

---

### 11. Data Extraction Agent (`LangChainDataExtractionAgent.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[เข้า LLM สกัดข้อมูล]
    B --> C[แสดง JSON]
    A -. batch .-> D[สกัดหลายข้อความ]
    C -. export .-> E[ส่งออกไฟล์]
    A -. error .-> F[handle error]
```

**อธิบาย:**
- รับข้อความ > สกัดข้อมูล > ตอบ (รองรับ batch, export)

---

### 12. Code Generation Agent (`LangChainCodeGenAgent.py`)

```mermaid
graph TD
    A[รับคำสั่ง] --> B[เข้า LLM สร้างโค้ด]
    B --> C[แยกโค้ด-อธิบาย]
    C --> D[validate syntax]
    D --> E[รันโค้ด]
    E --> F[แสดงผล]
    F -. save .-> G[บันทึกไฟล์]
    F -. test .-> H[รัน test]
    A -. error .-> I[handle error]
```

**อธิบาย:**
- รับคำสั่ง > สร้างโค้ด > อธิบาย/validate > รัน/บันทึก/test > ตอบ

---

### 13. Creative Agent (`LangChainCreativeAgent.py`)

```mermaid
graph TD
    A[รับไอเดีย] --> B[เลือกเนื้อหา]
    B --> C[เตรียม prompt]
    C --> D[เข้า LLM สร้างเนื้อหา]
    D --> E[แสดงผลงาน]
    A -. batch .-> F[สร้างหลายผลงาน]
    A -. error .-> G[handle error]
```

**อธิบาย:**
- รับไอเดีย > เลือกประเภท > เตรียม prompt > สร้างเนื้อหา > ตอบ (รองรับ batch)

---

### 14. Summarization Agent (`LangChainSummarizationAgent.py`)

```mermaid
graph TD
    A[รับข้อความ/ไฟล์] --> B[เลือกสรุป]
    B --> C[แบ่ง chunk]
    C --> D[เข้า LLM สรุป]
    D --> E[รวมผลลัพธ์]
    E --> F[แสดงสรุป]
    F -. save .-> G[บันทึกไฟล์]
    A -. batch .-> H[สรุปหลายข้อความ]
    A -. error .-> I[handle error]
```

**อธิบาย:**
- รับข้อความ/ไฟล์ > เลือกประเภท > chunk > สรุป > รวม > ตอบ (รองรับ batch, save)

---

### 15. Tool+Memory Agent (`LangChainToolMemory.py`)

```mermaid
graph TD
    A[รับข้อความ] --> B[เลือก Tool]
    B --> C[เรียก Tool]
    C --> D[บันทึก memory]
    D --> E[เข้า LLM]
    E --> F[แสดงผล]
    A -. clear .-> G[ล้าง memory]
    A -. error .-> H[handle error]
```

**อธิบาย:**
- รับข้อความ > เลือก tool > เรียกใช้งาน > บันทึก memory > ตอบ (รองรับ clear memory)

---

### 16. Fact-Check Agent (`LangChainFactCheckAgent.py`)

```mermaid
graph TD
    A[รับ claim] --> B[สร้าง search]
    B --> C[ค้นเว็บ]
    C --> D[รวบรวม evidence]
    D --> E[เข้า LLM วิเคราะห์]
    E --> F[แสดง verdict]
    A -. error .-> G[handle error]
```

**อธิบาย:**
- รับ claim > ค้นหา > รวบรวม evidence > วิเคราะห์ > ตอบ verdict

---

## ตัวอย่างการใช้งาน CLI ของแต่ละสคริปต์

> หมายเหตุ: ตัวอย่างนี้สำหรับรันบน Windows (cmd) หรือ Linux/Mac (bash) ได้เหมือนกัน

- **Basic Agent**
  ```sh
  python LangChainBasic.py --input "หัวข้อหรือข้อความที่ต้องการถาม"
  ```
- **RAG Agent**
  ```sh
  python LangChainRAG.py --file "path/to/document.txt" --query "คำถามของคุณ"
  ```
- **Multi-Document RAG Agent**
  ```sh
  python LangChainMultiDocRAGAgent.py --files "doc1.txt,doc2.pdf" --query "คำถามของคุณ"
  ```
- **SQL Agent**
  ```sh
  python LangChainSQL.py --db "example.db" --query "SELECT * FROM table_name;"
  ```
- **Web Tool Agent**
  ```sh
  python LangChainWebTool.py --query "ข่าวเทคโนโลยี AI ล่าสุดวันนี้"
  ```
- **Translation Agent**
  ```sh
  python LangChainTranslationAgent.py --text "สวัสดีครับ" --target_lang "en"
  ```
- **PDF Q&A Agent**
  ```sh
  python LangChainPDFQAAgent.py --file "example.pdf" --query "เนื้อหาเกี่ยวกับอะไร?"
  ```
- **Knowledge Graph Agent**
  ```sh
  python LangChainKnowledgeGraphAgent.py --text "บริษัท A ซื้อบริษัท B"
  ```
- **Sentiment Analysis Agent**
  ```sh
  python LangChainSentimentAgent.py --text "วันนี้รู้สึกดีมาก"
  ```
- **Text Classification Agent**
  ```sh
  python LangChainTextClassificationAgent.py --text "ข่าวเศรษฐกิจวันนี้"
  ```
- **Data Extraction Agent**
  ```sh
  python LangChainDataExtractionAgent.py --text "บริษัท ABC จำกัด ก่อตั้งปี 2020"
  ```
- **Code Generation Agent**
  ```sh
  python LangChainCodeGenAgent.py --task "เขียนฟังก์ชันหาค่าเฉลี่ยของตัวเลขใน list"
  ```
- **Creative Agent**
  ```sh
  python LangChainCreativeAgent.py --idea "เด็กชายคนหนึ่งพบไข่มังกรในป่า"
  ```
- **Summarization Agent**
  ```sh
  python LangChainSummarizationAgent.py --text "เนื้อหายาว ๆ ..."
  ```
- **Tool+Memory Agent**
  ```sh
  python LangChainToolMemory.py --input "อัตราแลกเปลี่ยนวันนี้เท่าไร"
  ```
- **Fact-Check Agent**
  ```sh
  python LangChainFactCheckAgent.py --claim "AI จะมาแทนที่มนุษย์ทุกอาชีพ"
  ```

> สามารถดู options เพิ่มเติมได้โดยรัน `python <script>.py --help` (ถ้าสคริปต์รองรับ)

## Best Practices

- ตรวจสอบ resource consumption ของ LLM/embedding
- DuckDuckGo Search อาจ rate limit หากเรียกบ่อย
- ตรวจสอบ license ของ model/embedding ที่ใช้งาน
- ใช้ reverse proxy และจำกัด access เมื่อ deploy จริง
- ใช้ .env สำหรับ key/token ทุกชนิด
- **ขยายไอเดียใหม่**: เพิ่ม agent สำหรับค้นหาไอเดีย, สร้างเนื้อหาใหม่, หรือวิเคราะห์แนวโน้มเทคโนโลยีได้ง่ายด้วยการเพิ่มไฟล์ agent ใหม่และเชื่อมต่อกับ LLM

## อัปเดตล่าสุด
- ปรับปรุง error handling, logging, input validation
- เพิ่มตัวอย่าง workflow สำหรับ Production-ready
- ลบรายการซ้ำและจัดระเบียบเนื้อหา