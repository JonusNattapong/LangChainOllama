# Optimized Enhanced LangChain SQL Agent + Advanced Features
# Install: pip install langchain langchain-community langchain-ollama langchain-sqlite sqlite-utils pandas plotly

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.manager import get_openai_callback
import sqlite3
import os
import logging
import pandas as pd
import json
from typing import Dict, Any, List, Optional, Union
import time
from datetime import datetime, timedelta
import re
import sys

class OptimizedSQLAgent:
    def __init__(self, model_name: str = "llama3.2:3b", db_path: str = "enhanced_example.db"):
        """
        Initialize Optimized SQL Agent with performance improvements
        
        Args:
            model_name: Ollama model name
            db_path: SQLite database path
        """
        self.model_name = model_name
        self.db_path = db_path
        self.logger = self._setup_logging()
        # Use window memory to limit context size
        self.memory = ConversationBufferWindowMemory(
            k=3,  # Keep only last 3 exchanges
            memory_key="chat_history", 
            return_messages=True
        )
        self.query_history = []
        
        # Setup database and agent
        self.setup_advanced_database()
        self.sql_db = self._create_sql_database()
        self.llm = self._create_llm()
        self.agent_executor = self._create_agent()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(f'sql_agent_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)

            # Console handler (force utf-8 if possible)
            try:
                console_handler = logging.StreamHandler()
                import sys
                if hasattr(console_handler, 'setStream') and hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            except Exception:
                pass  # fallback: skip console handler if encoding fails

            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
        return logger

    def setup_advanced_database(self):
        """Setup SQLite database with comprehensive sample data"""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced users table with more fields
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                email TEXT UNIQUE,
                department TEXT,
                salary INTEGER,
                hire_date DATE,
                manager_id INTEGER,
                city TEXT,
                performance_score REAL,
                FOREIGN KEY (manager_id) REFERENCES users (id)
            )''')
            
            # Enhanced products table
            cursor.execute('''CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT,
                stock INTEGER DEFAULT 0,
                supplier TEXT,
                created_date DATE,
                rating REAL,
                description TEXT
            )''')
            
            # Enhanced orders table
            cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                order_date DATE,
                status TEXT DEFAULT 'pending',
                total_amount REAL,
                discount REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )''')
            
            # New table: departments
            cursor.execute('''CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                budget INTEGER,
                manager_id INTEGER,
                location TEXT,
                FOREIGN KEY (manager_id) REFERENCES users (id)
            )''')
            
            # New table: sales_targets
            cursor.execute('''CREATE TABLE IF NOT EXISTS sales_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department TEXT,
                month TEXT,
                target_amount INTEGER,
                actual_amount INTEGER,
                year INTEGER
            )''')
            
            # Sample data for departments
            departments_data = [
                ('Engineering', 500000, None, 'Bangkok'),
                ('Marketing', 300000, None, 'Bangkok'),
                ('Sales', 400000, None, 'Chiang Mai'),
                ('HR', 200000, None, 'Bangkok'),
                ('Finance', 350000, None, 'Bangkok')
            ]
            cursor.executemany("INSERT INTO departments (name, budget, manager_id, location) VALUES (?, ?, ?, ?)", departments_data)
            
            # Enhanced users data
            users_data = [
                ('สมชาย วงศ์ใหญ่', 35, 'somchai@company.com', 'Engineering', 85000, '2020-01-15', None, 'Bangkok', 4.5),
                ('สมหญิง ดีมาก', 28, 'somying@company.com', 'Marketing', 65000, '2021-03-20', 1, 'Bangkok', 4.2),
                ('วิชาญ เก่งมาก', 42, 'wichan@company.com', 'Engineering', 95000, '2018-07-10', None, 'Bangkok', 4.8),
                ('นันทนา สวยงาม', 30, 'nantana@company.com', 'Sales', 70000, '2019-11-05', 3, 'Chiang Mai', 4.3),
                ('ประสิทธิ์ มั่นใจ', 38, 'prasit@company.com', 'HR', 60000, '2019-05-12', None, 'Bangkok', 4.1),
                ('อรุณ ขยัน', 26, 'arun@company.com', 'Marketing', 55000, '2022-01-08', 2, 'Bangkok', 3.9),
                ('สุดา เฉลียว', 33, 'suda@company.com', 'Finance', 75000, '2020-09-15', None, 'Bangkok', 4.4),
                ('พิมพ์ใจ รัก', 29, 'pimjai@company.com', 'Sales', 68000, '2021-06-20', 4, 'Chiang Mai', 4.6)
            ]
            cursor.executemany("INSERT INTO users (name, age, email, department, salary, hire_date, manager_id, city, performance_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", users_data)
            
            # Enhanced products data
            products_data = [
                ('MacBook Pro', 89000, 'Electronics', 25, 'Apple Thailand', '2024-01-01', 4.7, 'High-performance laptop'),
                ('iPhone 15', 35000, 'Electronics', 50, 'Apple Thailand', '2024-01-01', 4.5, 'Latest smartphone'),
                ('Samsung Monitor', 15000, 'Electronics', 80, 'Samsung', '2024-01-01', 4.3, '27-inch 4K monitor'),
                ('Wireless Mouse', 1500, 'Electronics', 200, 'Logitech', '2024-01-01', 4.2, 'Ergonomic wireless mouse'),
                ('Mechanical Keyboard', 3500, 'Electronics', 150, 'Corsair', '2024-01-01', 4.4, 'RGB mechanical keyboard'),
                ('Office Chair', 12000, 'Furniture', 30, 'Herman Miller', '2024-01-01', 4.6, 'Ergonomic office chair'),
                ('Standing Desk', 25000, 'Furniture', 20, 'IKEA', '2024-01-01', 4.1, 'Height-adjustable desk'),
                ('Tablet', 18000, 'Electronics', 60, 'Samsung', '2024-01-01', 4.0, 'Android tablet'),
                ('Headphones', 8000, 'Electronics', 100, 'Sony', '2024-01-01', 4.5, 'Noise-canceling headphones'),
                ('Webcam', 4500, 'Electronics', 75, 'Logitech', '2024-01-01', 4.2, 'HD webcam for meetings')
            ]
            cursor.executemany("INSERT INTO products (name, price, category, stock, supplier, created_date, rating, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", products_data)
            
            # Enhanced orders data with realistic scenarios
            orders_data = [
                (1, 1, 2, '2024-01-15', 'completed', 70000, 5000),
                (2, 1, 1, '2024-01-16', 'completed', 89000, 0),
                (1, 3, 1, '2024-01-17', 'pending', 15000, 0),
                (3, 2, 3, '2024-01-18', 'completed', 105000, 10000),
                (4, 4, 1, '2024-01-19', 'shipped', 1500, 0),
                (5, 5, 2, '2024-01-20', 'completed', 7000, 1000),
                (6, 6, 1, '2024-01-21', 'pending', 12000, 0),
                (7, 7, 3, '2024-01-22', 'completed', 54000, 4000),
                (8, 8, 1, '2024-01-23', 'shipped', 4500, 0),
                (1, 9, 2, '2024-01-24', 'completed', 16000, 0)
            ]
            cursor.executemany("INSERT INTO orders (user_id, product_id, quantity, order_date, status, total_amount, discount) VALUES (?, ?, ?, ?, ?, ?, ?)", orders_data)
            
            # Sales targets data
            sales_targets_data = [
                ('Sales', 'January', 500000, 480000, 2024),
                ('Sales', 'February', 520000, 510000, 2024),
                ('Marketing', 'January', 300000, 290000, 2024),
                ('Marketing', 'February', 310000, 320000, 2024),
            ]
            cursor.executemany("INSERT INTO sales_targets (department, month, target_amount, actual_amount, year) VALUES (?, ?, ?, ?, ?)", sales_targets_data)
            
            # Create useful indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_department ON users(department)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_salary ON users(salary)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_product_id ON orders(product_id)")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Enhanced database setup completed: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise

    def _create_sql_database(self) -> SQLDatabase:
        """Create SQLDatabase object with optimized configuration"""
        return SQLDatabase.from_uri(
            f'sqlite:///{self.db_path}', 
            include_tables=['users', 'products', 'orders', 'departments', 'sales_targets'], 
            sample_rows_in_table_info=3,  # Show more examples
            view_support=True,
            max_string_length=300  # Limit string length in table info
        )

    def _create_llm(self) -> OllamaLLM:
        """Create LLM with optimized Thai-focused system prompt"""
        system_prompt = """คุณคือผู้ช่วย AI เชี่ยวชาญการวิเคราะห์ฐานข้อมูล

ข้อมูลตาราง:
- users: พนักงาน (id, name, age, email, department, salary, hire_date, manager_id, city, performance_score)
- products: สินค้า (id, name, price, category, stock, supplier, created_date, rating, description)  
- orders: คำสั่งซื้อ (id, user_id, product_id, quantity, order_date, status, total_amount, discount)
- departments: แผนก (id, name, budget, manager_id, location)
- sales_targets: เป้าหมาย (id, department, month, target_amount, actual_amount, year)

กฎการทำงาน:
1. เขียน SQL อย่างเดียว ไม่ต้องอธิบายขั้นตอน
2. ใช้ JOIN เมื่อต้องการข้อมูลจากหลายตาราง
3. ใช้ฟังก์ชัน SQL เช่น COUNT, SUM, AVG, MAX, MIN
4. ตอบเป็นภาษาไทยสั้นๆ กระชับ
5. หากไม่พบข้อมูล ให้บอกตรงๆ

ตัวอย่าง:
Q: ใครมีเงินเดือนสูงสุด?
A: SELECT name, salary FROM users ORDER BY salary DESC LIMIT 1

Q: ฝ่าย Engineering มีใครบ้าง?
A: SELECT name, salary FROM users WHERE department = 'Engineering'
"""
        
        return OllamaLLM(
            model=self.model_name,
            system=system_prompt,
            temperature=0,  # More deterministic
            num_predict=200,  # Shorter responses
            repeat_penalty=1.05,
            top_k=10,
            top_p=0.9
        )

    def _create_agent(self):
        """Create SQL agent with optimized configuration"""
        return create_sql_agent(
            llm=self.llm,
            db=self.sql_db,
            agent_type="zero-shot-react-description",
            verbose=True,  # For debugging
            handle_parsing_errors=True,
            max_iterations=5,  # Reduced from 20
            max_execution_time=30,  # Reduced from 120
            return_intermediate_steps=False,  # Disable for performance
            early_stopping_method="force"  # Stop early if needed
        )

    def query_direct_sql(self, question: str) -> Dict[str, Any]:
        """Direct SQL query method for better performance"""
        start_time = time.time()
        
        try:
            # Simple keyword-based SQL generation for common queries
            sql_patterns = {
                'เงินเดือนสูงสุด': "SELECT name, salary FROM users ORDER BY salary DESC LIMIT 1",
                'engineering': "SELECT name, salary FROM users WHERE department = 'Engineering'",
                'electronics': "SELECT name, price FROM products WHERE category = 'Electronics'",
                'อายุมากกว่า 30': "SELECT name, age FROM users WHERE age > 30",
                'สต็อกน้อยที่สุด': "SELECT name, stock FROM products ORDER BY stock ASC LIMIT 1",
                'ฝ่ายไหนมีคนมากที่สุด': "SELECT department, COUNT(*) as count FROM users GROUP BY department ORDER BY count DESC LIMIT 1",
                'เงินเดือนเฉลี่ย': "SELECT AVG(salary) as avg_salary FROM users",
                'alice': """SELECT u.name, p.name as product_name, o.quantity 
                          FROM orders o 
                          JOIN users u ON o.user_id = u.id 
                          JOIN products p ON o.product_id = p.id 
                          WHERE u.name LIKE '%Alice%'"""
            }
            
            # Find matching pattern
            question_lower = question.lower()
            sql_query = None
            
            for pattern, sql in sql_patterns.items():
                if pattern.lower() in question_lower:
                    sql_query = sql
                    break
            
            if sql_query:
                # Execute direct SQL
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                
                # Format answer in Thai
                if df.empty:
                    answer = "ไม่พบข้อมูลที่ต้องการ"
                else:
                    if len(df) == 1 and len(df.columns) == 2:
                        # Single result with name and value
                        row = df.iloc[0]
                        if 'salary' in df.columns:
                            answer = f"{row[0]} มีเงินเดือน {row[1]:,} บาท"
                        elif 'price' in df.columns:
                            answer = f"{row[0]} ราคา {row[1]:,} บาท"
                        elif 'age' in df.columns:
                            answer = f"{row[0]} อายุ {row[1]} ปี"
                        elif 'stock' in df.columns:
                            answer = f"{row[0]} มีสต็อก {row[1]} ชิ้น"
                        elif 'count' in df.columns:
                            answer = f"ฝ่าย {row[0]} มี {row[1]} คน"
                        else:
                            answer = f"{row[0]}: {row[1]}"
                    elif 'avg_salary' in df.columns:
                        answer = f"เงินเดือนเฉลี่ย {df.iloc[0]['avg_salary']:,.0f} บาท"
                    else:
                        # Multiple results
                        results = []
                        for _, row in df.iterrows():
                            if 'salary' in df.columns:
                                results.append(f"{row['name']} ({row['salary']:,} บาท)")
                            elif 'price' in df.columns:
                                results.append(f"{row['name']} ({row['price']:,} บาท)")
                            elif 'age' in df.columns:
                                results.append(f"{row['name']} ({row['age']} ปี)")
                            else:
                                results.append(f"{row[0]}")
                        answer = ", ".join(results)
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "answer": answer,
                    "question": question,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "method": "direct_sql"
                }
            
            # Fallback to agent if no pattern matched
            return self.query_with_agent(question)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in direct SQL query: {e}")
            
            return {
                "success": False,
                "answer": f"เกิดข้อผิดพลาด: {str(e)}",
                "question": question,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "method": "direct_sql"
            }

    def query_with_agent(self, question: str) -> Dict[str, Any]:
        """Use LangChain agent for complex queries"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Using agent for: {question}")
            
            # Shorter context for better performance
            enhanced_question = f"ตอบสั้นๆ: {question}"
            
            response = self.agent_executor.invoke({"input": enhanced_question})
            answer = response.get("output", "ไม่สามารถหาคำตอบได้")
            
            # Clean up answer
            answer = self._clean_answer(answer)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "answer": answer,
                "question": question,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "method": "agent"
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in agent query: {e}")
            
            return {
                "success": False,
                "answer": f"เกิดข้อผิดพลาด: {str(e)}",
                "question": question,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "method": "agent"
            }

    def query(self, question: str, save_history: bool = True) -> Dict[str, Any]:
        """Main query method - tries direct SQL first, then agent"""
        if not question or not question.strip():
            return {
                "success": False,
                "answer": "คำถามว่างหรือไม่ถูกต้อง",
                "question": question,
                "processing_time": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Try direct SQL first for better performance
        result = self.query_direct_sql(question)
        
        # Save to history
        if save_history:
            self.query_history.append(result)
            if result['success']:
                self.memory.save_context({"input": question}, {"output": result['answer']})
        
        return result

    def _clean_answer(self, answer: str) -> str:
        """Clean and format the answer"""
        # Remove SQL queries from answer
        answer = re.sub(r'```sql.*?```', '', answer, flags=re.DOTALL)
        answer = re.sub(r'SELECT.*?;', '', answer, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Format numbers with commas for Thai readability
        def format_number(match):
            number = int(match.group())
            return f"{number:,}"
        
        answer = re.sub(r'\b\d{4,}\b', format_number, answer)
        
        return answer

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Generate analytics dashboard data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            analytics = {}
            
            # Employee analytics
            df_users = pd.read_sql_query("SELECT * FROM users", conn)
            analytics['total_employees'] = len(df_users)
            analytics['avg_salary'] = df_users['salary'].mean()
            analytics['dept_distribution'] = df_users['department'].value_counts().to_dict()
            analytics['avg_performance'] = df_users['performance_score'].mean()
            
            # Product analytics
            df_products = pd.read_sql_query("SELECT * FROM products", conn)
            analytics['total_products'] = len(df_products)
            analytics['avg_price'] = df_products['price'].mean()
            analytics['category_distribution'] = df_products['category'].value_counts().to_dict()
            analytics['low_stock_products'] = df_products[df_products['stock'] < 50]['name'].tolist()
            
            # Order analytics
            df_orders = pd.read_sql_query("""
                SELECT o.*, p.name as product_name, p.price, u.name as user_name
                FROM orders o
                JOIN products p ON o.product_id = p.id
                JOIN users u ON o.user_id = u.id
            """, conn)
            
            analytics['total_orders'] = len(df_orders)
            analytics['total_revenue'] = df_orders['total_amount'].sum()
            analytics['avg_order_value'] = df_orders['total_amount'].mean()
            analytics['order_status_distribution'] = df_orders['status'].value_counts().to_dict()
            
            conn.close()
            
            return {
                "success": True,
                "analytics": analytics,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating analytics: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def export_query_history(self, format: str = 'json') -> str:
        """Export query history to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_history_{timestamp}.{format}"
            
            if format.lower() == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.query_history, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == 'csv':
                df = pd.DataFrame(self.query_history)
                df.to_csv(filename, index=False, encoding='utf-8')
            
            self.logger.info(f"Query history exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting history: {e}")
            return ""

    def suggest_questions(self) -> List[str]:
        """Generate suggested questions based on database content"""
        suggestions = [
            "พนักงานคนไหนมีผลงานดีที่สุด?",
            "แผนกไหนมีพนักงานมากที่สุด?",
            "สินค้าไหนขายดีที่สุด?",
            "ใครสั่งซื้อสินค้ามากที่สุด?",
            "แผนกไหนมีเงินเดือนเฉลี่ยสูงสุด?",
            "สินค้าไหนมีสต็อกน้อยกว่า 50 ชิ้น?",
            "พนักงานที่เข้าทำงานปี 2020 มีใครบ้าง?",
            "ยอดขายรวมในเดือนมกราคมเป็นเท่าไร?",
            "แผนกไหนบรรลุเป้าหมายการขาย?",
            "ผลิตภัณฑ์ Electronics ที่ได้คะแนนรีวิวสูงสุดคืออะไร?"
        ]
        return suggestions

    def test_performance(self) -> Dict[str, Any]:
        """Test performance with common queries"""
        test_questions = [
            "มีใครบ้างในฝ่าย Engineering และเงินเดือนเท่าไร?",
            "ใครมีเงินเดือนสูงที่สุด?",
            "สินค้าในหมวด Electronics มีอะไรบ้าง และราคาเท่าไร?",
            "มีใครอายุมากกว่า 30 ปี?",
            "สินค้าไหนมีสต็อกน้อยที่สุด?",
            "ฝ่ายไหนมีคนมากที่สุด?",
            "เงินเดือนเฉลี่ยของพนักงานทั้งหมดคือเท่าไร?"
        ]
        
        results = []
        total_time = 0
        
        print("🧪 === การทดสอบประสิทธิภาพ ===")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. คำถาม: {question}")
            result = self.query(question, save_history=False)
            
            status = "✅" if result['success'] else "❌"
            print(f"   {status} คำตอบ: {result['answer']}")
            print(f"   ⏱️ เวลา: {result['processing_time']:.2f}s")
            print(f"   🔧 วิธีการ: {result.get('method', 'unknown')}")
            
            results.append(result)
            total_time += result['processing_time']
        
        print(f"\n📊 สรุปผลการทดสอบ:")
        print(f"   ✅ สำเร็จ: {sum(1 for r in results if r['success'])}/{len(results)}")
        print(f"   ⏱️ เวลารวม: {total_time:.2f}s")
        print(f"   ⚡ เวลาเฉลี่ย: {total_time/len(results):.2f}s")
        
        return {
            "total_questions": len(test_questions),
            "successful": sum(1 for r in results if r['success']),
            "total_time": total_time,
            "average_time": total_time / len(results),
            "results": results
        }

def interactive_optimized_demo():
    """Optimized interactive demonstration"""
    print("🚀 === Optimized SQL Agent Demo ===")
    print("คำสั่งพิเศษ:")
    print("  📊 'analytics' - ดูแดชบอร์ดวิเคราะห์ข้อมูล")
    print("  📋 'schema' - ดูโครงสร้างฐานข้อมูล") 
    print("  💡 'suggest' - ดูคำถามแนะนำ")
    print("  📈 'history' - ดูประวัติคำถาม")

def main():
    print("\n=== เมนูหลัก: เลือกโหมดการใช้งาน SQL Agent ===")
    print("1. ทดสอบประสิทธิภาพ (Performance Test): ทดสอบชุดคำถามอัตโนมัติและดูผลลัพธ์ เช่น 'มีใครบ้างในฝ่าย Engineering และเงินเดือนเท่าไร?', 'ใครมีเงินเดือนสูงที่สุด?' ฯลฯ")
    print("2. โหมดโต้ตอบ (Interactive): พิมพ์คำถามเองและรับคำตอบแบบโต้ตอบสด เช่น 'ยอดขายรวมในเดือนมกราคมเป็นเท่าไร?', 'สินค้าไหนมีสต็อกน้อยที่สุด?' ฯลฯ")
    print("3. ดูแดชบอร์ดวิเคราะห์ข้อมูล (Analytics Dashboard): ดูสรุปข้อมูลพนักงาน สินค้า ยอดขาย ฯลฯ ในรูปแบบ JSON")
    print("4. ดูคำถามแนะนำ (Suggest Questions): ดูตัวอย่างคำถามที่ระบบแนะนำ (เหมาะสำหรับผู้เริ่มต้น)")
    print("5. ออก (Exit): ออกจากโปรแกรม\n")
    agent = OptimizedSQLAgent()
    while True:
        choice = input("กรุณาพิมพ์หมายเลข 1-5 เพื่อเลือกเมนู: ").strip()
        if choice == "1":
            agent.test_performance()
        elif choice == "2":
            print("\nพิมพ์ 'exit' เพื่อออกจากโหมดโต้ตอบ\n")
            while True:
                q = input("\nคำถาม: ").strip()
                if q.lower() in ["exit", "quit", "ออก"]:
                    break
                if not q:
                    print("กรุณาใส่คำถาม")
                    continue
                result = agent.query(q)
                status = "✅" if result['success'] else "❌"
                print(f"{status} คำตอบ: {result['answer']}")
                print(f"⏱️ เวลา: {result['processing_time']:.2f}s")
        elif choice == "3":
            dashboard = agent.get_analytics_dashboard()
            if dashboard['success']:
                print(json.dumps(dashboard['analytics'], ensure_ascii=False, indent=2))
            else:
                print(f"❌ {dashboard['error']}")
        elif choice == "4":
            print("\nคำถามแนะนำ:")
            for i, q in enumerate(agent.suggest_questions(), 1):
                print(f"{i}. {q}")
        elif choice == "5":
            print("ขอบคุณที่ใช้งาน!")
            break
        else:
            print("กรุณาเลือก 1-5")

if __name__ == "__main__":
    main()