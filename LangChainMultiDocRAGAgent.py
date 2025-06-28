# LangChain Workflow: Multi-Document RAG Agent
# ถามตอบจากหลายไฟล์/หลายแหล่งข้อมูล (TXT, PDF, Web)

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class MultiDocRAGAgent:
    """ตัวแทนถามตอบจากหลายเอกสารด้วย RAG (Product-ready)"""

    def __init__(self, model_name="llama3.2:3b", embedding_model="intfloat/multilingual-e5-base"):
        """กำหนดค่าเริ่มต้นสำหรับโมเดลและเวกเตอร์สโตร์"""
        self.llm = OllamaLLM(model=model_name)
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, file_paths: list) -> list:
        """โหลดและแบ่งเอกสารจากหลายไฟล์ (ตรวจสอบ input และ handle error)"""
        try:
            if not isinstance(file_paths, list) or not file_paths:
                return []
            docs = []
            for file_path in file_paths:
                if file_path.endswith(".txt") and os.path.exists(file_path):
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
                elif file_path.endswith(".pdf") and os.path.exists(file_path):
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            return splitter.split_documents(docs)
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการโหลดเอกสารหลายไฟล์")
            return []

    def create_vectorstore(self, chunks: list) -> None:
        """สร้างเวกเตอร์สโตร์จากเอกสารที่แบ่งแล้ว"""
        try:
            if not isinstance(chunks, list) or not chunks:
                return
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการสร้าง vectorstore")

    def setup_qa_chain(self) -> None:
        """ตั้งค่า QA chain สำหรับการถามตอบ (คืน source documents ด้วย)"""
        try:
            if not self.vectorstore:
                return
            retriever = self.vectorstore.as_retriever()
            prompt = PromptTemplate(
                template="""ใช้ข้อมูลต่อไปนี้เพื่อตอบคำถาม หากไม่พบข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่ทราบ\n\nข้อมูล: {context}\n\nคำถาม: {question}\n\nคำตอบ:""",
                input_variables=["context", "question"]
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการตั้งค่า QA chain")

    def query(self, question: str) -> dict:
        """ส่งคำถามไปยัง QA chain และส่งคืนคำตอบ พร้อมเนื้อหาต้นทาง (context)"""
        try:
            if not isinstance(question, str) or not question.strip():
                return {"answer": "คำถามว่างหรือไม่ถูกต้อง", "sources": [], "contexts": []}
            if not self.qa_chain:
                return {"answer": "ระบบยังไม่พร้อมใช้งาน", "sources": [], "contexts": []}
            result = self.qa_chain({"query": question})
            # ดึง context ต้นทาง (เนื้อหาที่ใช้ตอบ)
            contexts = [doc.page_content for doc in result.get("source_documents", [])]
            return {
                "answer": result.get("result", "ไม่พบคำตอบ"),
                "sources": [doc.metadata.get("source", "ไม่ทราบแหล่งที่มา") for doc in result.get("source_documents", [])],
                "contexts": contexts
            }
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการถาม MultiDoc QA")
            return {"answer": f"เกิดข้อผิดพลาด: {e}", "sources": [], "contexts": []}

if __name__ == "__main__":
    agent = MultiDocRAGAgent()
    files = ["docs/mydoc.txt", "example.pdf"]
    chunks = agent.load_documents(files)
    if chunks:
        agent.create_vectorstore(chunks)
        agent.setup_qa_chain()
        print(agent.query("AI คืออะไร?"))
    else:
        print("ไม่พบไฟล์หรือไม่สามารถโหลดได้")
