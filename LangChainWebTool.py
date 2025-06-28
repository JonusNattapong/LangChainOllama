# Enhanced LangChain Agent + Web Tool (Search API)
# Install: pip install langchain langchain-community langchain-ollama duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self, model_name: str = "qwen3:1.7b", max_iterations: int = 3):
        """Initialize the web search agent with configurable parameters."""
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.llm = None
        self.agent = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._setup_agent()
    
    def search_with_fallback(self, query: str) -> str:
        """Enhanced search function with multiple fallback strategies."""
        if not isinstance(query, str) or not query.strip():
            return "Final Answer: คำค้นหาว่างหรือไม่ถูกต้อง"
        
        # Strategy 1: Try API backend first
        try:
            logger.info(f"Searching with API backend: {query}")
            search_tool = DuckDuckGoSearchRun(backend="api", max_results=5)
            result = search_tool.run(query.strip())
            if result and len(result.strip()) > 10:  # Basic quality check
                return f"Final Answer: {result}"
        except Exception as e:
            logger.warning(f"API backend failed: {str(e)}")
        
        # Strategy 2: Try HTML backend
        try:
            logger.info(f"Fallback to HTML backend: {query}")
            search_tool = DuckDuckGoSearchRun(backend="html", max_results=3)
            result = search_tool.run(query.strip())
            if result and len(result.strip()) > 10:
                return f"Final Answer: {result}"
        except Exception as e:
            logger.warning(f"HTML backend failed: {str(e)}")
        
        # Strategy 3: Try with simplified query
        try:
            simplified_query = query.replace("ล่าสุด", "").replace("วันนี้", "").strip()
            if simplified_query and simplified_query != query:
                logger.info(f"Trying simplified query: {simplified_query}")
                search_tool = DuckDuckGoSearchRun(backend="api", max_results=3)
                result = search_tool.run(simplified_query)
                if result and len(result.strip()) > 10:
                    return f"Final Answer: {result}"
        except Exception as e:
            logger.warning(f"Simplified query failed: {str(e)}")
        
        return "Final Answer: ขออภัย ไม่สามารถค้นหาข้อมูลได้ในขณะนี้ กรุณาลองใหม่อีกครั้งหรือเปลี่ยนคำค้นหา"
    
    def _setup_agent(self):
        """Setup the LLM and agent with tools."""
        try:
            # Initialize LLM
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.1,  # Lower temperature for more focused responses
                num_predict=1000,  # Limit response length
            )
            
            # Create enhanced search tool
            search_tool = Tool(
                name="web_search",
                func=self.search_with_fallback,
                description="""Use this tool to search for current information from the internet using DuckDuckGo.
                Best for: latest news, current events, recent technology updates, real-time information.
                Input should be a clear, specific search query in Thai or English.
                The tool automatically returns 'Final Answer:' so don't add it yourself."""
            )
            
            # Initialize agent with memory
            self.agent = initialize_agent(
                tools=[search_tool],
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
                memory=self.memory,
                agent_kwargs={
                    "prefix": """You are a helpful AI assistant that can search the web for current information.
                    When asked about recent events, news, or current information, use the web_search tool.
                    Always provide responses in Thai when the user asks in Thai.
                    Be concise but informative in your responses."""
                }
            )
            logger.info("Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup agent: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return structured response."""
        try:
            logger.info(f"Processing query: {question}")
            response = self.agent.invoke({"input": question})
            
            return {
                "success": True,
                "output": response.get("output", str(response)),
                "chat_history": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output": f"เกิดข้อผิดพลาด: {str(e)}"
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Memory cleared")

# Convenience function for quick usage
def create_search_agent(model: str = "llama3.2:3b") -> WebSearchAgent:
    """Create and return a configured web search agent."""
    return WebSearchAgent(model_name=model)

# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = create_search_agent()
    
    # Test queries
    test_queries = [
        "ข่าวเทคโนโลยี AI ล่าสุดวันนี้",
        "What are the latest developments in renewable energy?",
        "ราคาหุ้นไทยวันนี้",
        "อัตราแลกเปลี่ยนเงินบาทล่าสุด"
    ]
    
    print("🤖 เริ่มทดสอบ LangChain Web Search Agent\n")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 ทดสอบที่ {i}: {query}")
        print("-" * 40)
        
        result = agent.query(query)
        
        if result["success"]:
            print(f"✅ ผลลัพธ์: {result['output']}")
        else:
            print(f"❌ ข้อผิดพลาด: {result['error']}")
        
        print("-" * 40)
    
    print(f"\n🎉 การทดสอบเสร็จสิ้น!")
    
    # Interactive mode
    print(f"\n💬 โหมดโต้ตอบ (พิมพ์ 'quit' เพื่อออก)")
    while True:
        try:
            user_input = input("\n🙋 คำถาม: ").strip()
            if user_input.lower() in ['quit', 'exit', 'ออก']:
                break
            
            if user_input:
                result = agent.query(user_input)
                print(f"\n🤖 ตอบ: {result['output']}")
            else:
                print("กรุณาใส่คำถาม")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {str(e)}")
    
    print("\n👋 ขอบคุณที่ใช้งาน!")