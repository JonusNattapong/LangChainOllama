from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. โหลดเอกสาร
loader = TextLoader("docs/mydoc.txt", encoding="utf-8")
documents = loader.load()

# 2. แบ่งเอกสารเป็น chunk ย่อย
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# 3. ใช้ HuggingFace Embedding (หรือจะใช้ OpenAI ก็ได้ถ้ามี key)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# 4. สร้าง Vector Store ด้วย FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. LLM จาก Ollama
llm = OllamaLLM(model="llama3.2:3b")

# 6. สร้าง RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", k=3),
    return_source_documents=True
)

# 7. ใช้งานระบบถามตอบ
query = "บทความนี้พูดถึงอะไรเกี่ยวกับ AI?"
result = qa_chain.invoke(query)

print("คำตอบ:", result["result"])
print("\nแหล่งที่อ้างอิง:")
for doc in result["source_documents"]:
    print("-", doc.metadata["source"])
