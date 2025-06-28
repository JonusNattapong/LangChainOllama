from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import networkx as nx
import logging

class KnowledgeGraphAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(model=model_name)
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def build_graph_from_text(self, text):
        try:
            if not isinstance(text, str) or not text.strip():
                self.logger.error("ข้อความว่างหรือไม่ถูกต้อง")
                return
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""Extract knowledge graph triplets (subject, relation, object) from the following text:
{text}
Triplets:"""
            )
            chain = prompt | self.llm
            result = chain.invoke({"text": text})
            for line in str(result).splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    self.graph.add_edge(parts[0], parts[2], relation=parts[1])
            self.logger.info("Graph built successfully.")
        except Exception as e:
            self.logger.exception("Error building graph")

    def query(self, question):
        try:
            if not isinstance(question, str) or not question.strip():
                self.logger.error("คำถามว่างหรือไม่ถูกต้อง")
                return "คำถามว่างหรือไม่ถูกต้อง"
            prompt = PromptTemplate(
                input_variables=["question"],
                template="""แปลงคำถามต่อไปนี้เป็น subject หรือ object ที่ต้องการค้นหาใน knowledge graph:
{question}
คำตอบ:"""
            )
            chain = prompt | self.llm
            node = str(chain.invoke({"question": question})).strip()
            answers = []
            for u, v, d in self.graph.edges(data=True):
                if node in [u, v]:
                    answers.append(f"{u} --{d['relation']}--> {v}")
            self.logger.info(f"Query executed successfully: {question}")
            return answers if answers else "ไม่พบข้อมูลในกราฟ"
        except Exception as e:
            self.logger.exception("Error querying graph")
            return "เกิดข้อผิดพลาดในการค้นหา"

if __name__ == "__main__":
    agent = KnowledgeGraphAgent()
    agent.build_graph_from_text("Steve Jobs founded Apple in 1976.")
    print(agent.query("Who founded Apple?"))
if __name__ == "__main__":
    agent = KnowledgeGraphAgent()
    agent.build_graph_from_text("Steve Jobs founded Apple in 1976.")
    print(agent.query("Who founded Apple?"))
