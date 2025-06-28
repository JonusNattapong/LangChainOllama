from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import networkx as nx
import logging
import re

class KnowledgeGraphAgent:
    def __init__(self, model_name="llama3.2:3b"):
        self.llm = OllamaLLM(
            model=model_name,
            system=(
                "You are a knowledge graph triplet extractor. "
                "Given a sentence, extract all (subject, relation, object) triplets in English. "
                "Output only triplets, one per line, comma separated. Do not explain."
            )
        )
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
                template="Extract (subject, relation, object) triplets from: {text}\nTriplets:"
            )
            chain = prompt | self.llm
            result = chain.invoke({"text": text})
            print("LLM Triplet Output:\n", result)
            def clean_triplet_line(line):
                # ตัด bullet, เลขลำดับ, *, และวงเล็บรอบ triplet
                line = re.sub(r'^[\*\-\d\.\s]*', '', line)  # ตัด *, -, 1. 2. 3. หรือช่องว่างนำหน้า
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    line = line[1:-1]
                return line
            # รองรับทั้งแบบคอมม่าและแบบ label
            lines = [l.strip() for l in str(result).splitlines() if l.strip()]
            triplet = []
            for line in lines:
                # แบบคอมม่า (clean ก่อน)
                line_clean = clean_triplet_line(line)
                parts = [p.strip() for p in line_clean.split(",") if p.strip()]
                if len(parts) == 3:
                    print(f"Add edge: {parts[0]} --{parts[1]}--> {parts[2]}")
                    self.graph.add_edge(parts[0], parts[2], relation=parts[1])
                    triplet = []  # reset triplet
                # แบบ label (รองรับเลขลำดับนำหน้า)
                elif re.match(r'^(\d+\.|\*)?\s*subject:', line, re.I):
                    triplet = [line.split(":",1)[1].strip()]
                elif re.match(r'^(\d+\.|\*)?\s*relation:', line, re.I) and triplet:
                    triplet.append(line.split(":",1)[1].strip())
                elif re.match(r'^(\d+\.|\*)?\s*object:', line, re.I) and len(triplet)==2:
                    triplet.append(line.split(":",1)[1].strip())
                    if len(triplet)==3:
                        print(f"Add edge: {triplet[0]} --{triplet[1]}--> {triplet[2]}")
                        self.graph.add_edge(triplet[0], triplet[2], relation=triplet[1])
                    triplet = []
            print("Current graph nodes:", list(self.graph.nodes))
            self.logger.info("Graph built successfully.")
        except Exception as e:
            self.logger.exception("Error building graph")

    def query(self, question):
        try:
            if not isinstance(question, str) or not question.strip():
                self.logger.error("คำถามว่างหรือไม่ถูกต้อง")
                return "คำถามว่างหรือไม่ถูกต้อง"
            prompt = PromptTemplate(
                input_variables=["question", "nodes"],
                template="""จากคำถามต่อไปนี้ ให้ตอบชื่อ node (subject หรือ object) ที่ต้องการค้นหาใน knowledge graph โดยเลือกจาก node ที่มีในกราฟเท่านั้น (node ที่มีในกราฟ: {nodes}) ตอบชื่อ node ตรงๆ เท่านั้น ห้ามอธิบาย\nตัวอย่าง: ถ้าในกราฟมี Steve Jobs, Apple, 1976 ให้ตอบ Steve Jobs หรือ Apple หรือ 1976 เท่านั้น\nคำถาม: {question}\nNode:"""
            )
            nodes_list = list(self.graph.nodes)
            chain = prompt | self.llm
            node_raw = str(chain.invoke({"question": question, "nodes": ', '.join(nodes_list)})).strip()
            node_lines = [l.strip() for l in node_raw.splitlines() if l.strip()]
            # ข้าม reasoning <think> และเลือกบรรทัดแรกหลัง <think> (หรือบรรทัดแรกที่ไม่ใช่ reasoning)
            node = None
            found_think = False
            for l in node_lines:
                if found_think and l:
                    node = l
                    break
                if l.lower().startswith('<think>'):
                    found_think = True
            if not node:
                node = node_raw  # fallback
            # เลือก node ที่ตรงกับในกราฟ (case-insensitive)
            match_node = next((n for n in nodes_list if node.lower() in n.lower()), None)
            print(f"LLM query node: '{node}' | Matched node: '{match_node}'")
            print("Current graph nodes:", nodes_list)
            answers = []
            for u, v, d in self.graph.edges(data=True):
                if match_node and match_node in [u, v]:
                    answers.append(f"{u} --{d['relation']}--> {v}")
            if not answers:
                print("ไม่พบ node ที่ต้องการในกราฟ ลองใช้ node เหล่านี้:", nodes_list)
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
