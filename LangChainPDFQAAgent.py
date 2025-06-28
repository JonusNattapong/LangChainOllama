# LangChain Workflow: PDF Q&A Agent
# ถามตอบข้อมูลจากไฟล์ PDF ด้วย RAG + LLM

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class PDFQAAgent:
    def __init__(self, model_name="llama3.2:3b", embedding_model="intfloat/multilingual-e5-base"):
        self.llm = OllamaLLM(model=model_name)
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_chain = None

    def load_pdf(self, file_path: str) -> list:
        try:
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                return []
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            return splitter.split_documents(documents)
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการโหลด PDF")
            return []

    def create_vectorstore(self, chunks: list) -> None:
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
                chain_type_kwargs={"prompt": prompt}
            )
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการตั้งค่า QA chain")

    def query(self, question: str) -> dict:
        try:
            if not isinstance(question, str) or not question.strip():
                return {"answer": "คำถามว่างหรือไม่ถูกต้อง", "sources": []}
            if not self.qa_chain:
                return {"answer": "ระบบยังไม่พร้อมใช้งาน", "sources": []}
            result = self.qa_chain.invoke(question)
            return {
                "answer": result.get("result", "ไม่พบคำตอบ"),
                "sources": [doc.metadata.get("source", "ไม่ทราบแหล่งที่มา") for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).exception("เกิดข้อผิดพลาดในการถาม PDF QA")
            return {"answer": f"เกิดข้อผิดพลาด: {e}", "sources": []}

if __name__ == "__main__":
    agent = PDFQAAgent()
    chunks = agent.load_pdf("2505.04588v1.pdf")
    if chunks:
        agent.create_vectorstore(chunks)
        agent.setup_qa_chain()
        print(agent.query("ZEROSEARCH คือ งานวิจัยเกี่ยวกับ อะไร? พออธิบายให้เข้าใจง่ายๆ ได้ไหม?"))
    else:
        print("ไม่พบไฟล์ PDF หรือไม่สามารถโหลดได้")
