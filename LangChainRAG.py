from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGSystem:
    """คลาสสำหรับจัดการระบบ RAG (Product-ready)"""
    
    def __init__(self, model_name="llama3.2:3b", embedding_model="intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_path: str) -> list:
        """โหลดและแบ่งเอกสาร (ตรวจสอบไฟล์และ handle error)"""
        try:
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                logger.error(f"ไม่พบไฟล์: {file_path}")
                return []
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            logger.info(f"โหลดเอกสารสำเร็จ: {len(documents)} เอกสาร")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"แบ่งเอกสารเป็น {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการโหลดเอกสาร")
            return []

    def create_vectorstore(self, chunks: list) -> bool:
        """สร้าง Vector Store (ตรวจสอบ input และ handle error)"""
        if not isinstance(chunks, list) or not chunks:
            logger.error("Chunks ว่างหรือไม่ถูกต้อง")
            return False
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            logger.info("สร้าง Vector Store สำเร็จ")
            return True
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการสร้าง Vector Store")
            return False

    def setup_qa_chain(self) -> bool:
        """ตั้งค่า QA Chain (ตรวจสอบ vectorstore และ handle error)"""
        if not self.vectorstore:
            logger.error("Vectorstore ยังไม่ถูกสร้าง")
            return False
        try:
            llm = OllamaLLM(model=self.model_name)
            prompt_template = """ใช้ข้อมูลต่อไปนี้เพื่อตอบคำถาม หากไม่พบข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่ทราบ

ข้อมูล: {context}

คำถาม: {question}

คำตอบ:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            logger.info("ตั้งค่า QA Chain สำเร็จ")
            return True
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการตั้งค่า QA Chain")
            return False

    def query(self, question: str) -> dict:
        """ถามคำถามกับระบบ RAG (ตรวจสอบ input และ handle error)"""
        try:
            if not isinstance(question, str) or not question.strip():
                return {"answer": "คำถามว่างหรือไม่ถูกต้อง", "sources": []}
            if not self.qa_chain:
                return {"answer": "ระบบยังไม่พร้อมใช้งาน กรุณาตั้งค่าก่อน", "sources": []}
            result = self.qa_chain.invoke(question)
            return {
                "answer": result["result"],
                "sources": [doc.metadata.get("source", "ไม่ทราบแหล่งที่มา") 
                          for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            logger.exception("เกิดข้อผิดพลาดในการถามคำถาม")
            return {"answer": f"เกิดข้อผิดพลาด: {e}", "sources": []}

import argparse

def main():
    """
    ฟังก์ชันหลักสำหรับทดสอบระบบ RAG
    สามารถรับ path ของไฟล์และคำถามจาก command line ได้
    """
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--file", type=str, default="docs/mydoc.txt", help="Path ของไฟล์เอกสาร")
    parser.add_argument("--question", type=str, help="คำถามเดียว (ถ้าไม่ระบุจะใช้ชุดตัวอย่าง)")
    args = parser.parse_args()

    rag = RAGSystem()
    chunks = rag.load_documents(args.file)
    if not chunks:
        print("ไม่สามารถโหลดเอกสารได้")
        return
    if not rag.create_vectorstore(chunks):
        print("ไม่สามารถสร้าง Vector Store ได้")
        return
    if not rag.setup_qa_chain():
        print("ไม่สามารถตั้งค่า QA Chain ได้")
        return

    if args.question:
        questions = [args.question]
    else:
        questions = [
            "บทความนี้พูดถึงอะไรเกี่ยวกับ AI?",
            "AI ช่วยงานอะไรได้บ้าง?",
            "มีข้อควรระวังอะไรเกี่ยวกับ AI?"
        ]
    for i, question in enumerate(questions, 1):
        print(f"\n=== คำถามที่ {i}: {question} ===")
        result = rag.query(question)
        print(f"คำตอบ: {result['answer']}")
        if result['sources']:
            print(f"แหล่งที่อ้างอิง: {', '.join(set(result['sources']))}")

if __name__ == "__main__":
    main()
