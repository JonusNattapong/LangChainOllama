# Enhanced LangChain + MCP + RAG + Ollama Example
# This script demonstrates advanced integration of LangChain, MCP, RAG, and Ollama

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler for MCP integration
class MCPCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to log interactions for MCP monitoring"""
    
    def __init__(self):
        self.interactions = []
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        self.current_interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": inputs.get("query"),
            "chain_type": serialized.get("name")
        }
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if hasattr(self, 'current_interaction'):
            self.current_interaction["result"] = outputs.get("result")
            self.current_interaction["source_count"] = len(outputs.get("source_documents", []))
            self.interactions.append(self.current_interaction)
    
    def get_interactions(self) -> List[Dict]:
        return self.interactions

class EnhancedRAGSystem:
    """Enhanced RAG system with MCP integration and better document handling"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 embedding_model: str = "intfloat/e5-base-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        # Initialize components
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # MCP callback handler
        self.mcp_handler = MCPCallbackHandler()
        
        # Initialize with default documents
        self._setup_default_knowledge_base()
        
    def _setup_default_knowledge_base(self):
        """Setup default knowledge base with more comprehensive documents"""
        default_docs = [
            """LangChain is a comprehensive framework for developing applications powered by language models. 
            It provides tools for prompt management, chains, agents, memory, and integrations with various LLMs and data sources.""",
            
            """Ollama is a powerful tool that allows you to run open-source large language models locally on your machine. 
            It supports models like Llama, Code Llama, Mistral, and many others, providing privacy and control over your AI applications.""",
            
            """Model Context Protocol (MCP) is an open protocol that enables secure connections between host applications 
            and AI models. It standardizes how AI assistants can securely access data and tools while maintaining user control.""",
            
            """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. 
            It allows language models to access external knowledge bases, improving accuracy and reducing hallucinations.""",
            
            """Vector databases like FAISS, Pinecone, or Chroma store embeddings of documents and enable semantic search. 
            They are crucial for RAG applications as they allow finding relevant context based on semantic similarity.""",
            
            """HuggingFace provides pre-trained embedding models that convert text into dense vector representations. 
            Models like e5-base-v2, sentence-transformers, and others are commonly used for semantic search applications."""
        ]
        
        # Split documents into chunks
        documents = []
        for i, doc_text in enumerate(default_docs):
            chunks = self.text_splitter.split_text(doc_text)
            for j, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": f"doc_{i}", "chunk": j}
                ))
        
        # Create vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Setup RAG chain with custom prompt
        custom_prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.
            If you cannot answer based on the context, say so clearly.

            Context: {context}

            Question: {question}

            Answer: """,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
            callbacks=[self.mcp_handler]
        )
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add new documents to the knowledge base"""
        docs = []
        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={**doc_metadata, "chunk": j}
                ))
        
        # Add to existing vector store
        self.vector_store.add_documents(docs)
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "timestamp": datetime.now().isoformat()
            }
            
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            
            return response
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_mcp_interactions(self) -> List[Dict]:
        """Get all logged interactions for MCP monitoring"""
        return self.mcp_handler.get_interactions()
    
    def export_mcp_log(self, filename: str = "mcp_interactions.json"):
        """Export MCP interactions to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.get_mcp_interactions(), f, indent=2)
        print(f"MCP interactions exported to {filename}")

# Demonstration function
def main():
    """Main demonstration of the enhanced RAG system"""
    print("🚀 Initializing Enhanced RAG System...")
    rag_system = EnhancedRAGSystem()
    
    # Example queries
    queries = [
        "What is the benefit of using RAG with Ollama and LangChain?",
        "How does MCP help with AI model interoperability?",
        "What are vector databases and why are they important for RAG?",
        "Can you explain how HuggingFace embeddings work?"
    ]
    
    print("\n📊 Running example queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {query}")
        
        result = rag_system.query(query)
        print(f"A: {result['answer']}")
        
        if result.get('sources'):
            print(f"📚 Sources used: {len(result['sources'])} documents")
    
    # Add custom documents
    print("\n📝 Adding custom documents...")
    custom_docs = [
        """Prompt engineering is the practice of designing and optimizing prompts to get better results from language models. 
        It involves techniques like few-shot learning, chain-of-thought prompting, and instruction tuning.""",
        
        """Fine-tuning involves training a pre-trained model on domain-specific data to improve performance for specific tasks. 
        It's more resource-intensive than prompt engineering but can yield better results for specialized applications."""
    ]
    
    rag_system.add_documents(
        custom_docs, 
        metadata=[
            {"source": "prompt_engineering_guide", "topic": "AI"},
            {"source": "fine_tuning_guide", "topic": "ML"}
        ]
    )
    
    # Query with new documents
    new_query = "What's the difference between prompt engineering and fine-tuning?"
    print(f"\n--- New Query ---")
    print(f"Q: {new_query}")
    result = rag_system.query(new_query)
    print(f"A: {result['answer']}")
    
    # Export MCP interactions
    print("\n📤 Exporting MCP interactions...")
    rag_system.export_mcp_log()
    
    # Show MCP statistics
    interactions = rag_system.get_mcp_interactions()
    print(f"\n📈 MCP Statistics:")
    print(f"Total interactions: {len(interactions)}")
    if interactions:
        avg_sources = sum(interaction.get('source_count', 0) for interaction in interactions) / len(interactions)
        print(f"Average sources per query: {avg_sources:.1f}")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install langchain langchain-ollama langchain-community faiss-cpu sentence-transformers")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running and the model is available.")
        print("Run: ollama pull llama3.2:3b")