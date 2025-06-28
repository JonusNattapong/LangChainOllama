from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from typing import Dict, Any, List, Optional
import logging
import json
import datetime
import re
import requests
import pytz
from datetime import timezone

class RealDataThaiAgent:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize Thai Chat Agent with real data APIs
        
        Args:
            model_name: Ollama model name to use
        """
        self.llm = OllamaLLM(model=model_name)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # API endpoints for real data
        self.exchange_api_base = "https://api.exchangerate.host"
        self.backup_exchange_api = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1"
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_tools(self) -> List[Tool]:
        """Create tools with real data sources"""
        tools = [
            Tool(
                name="exchange_rate",
                func=self.get_real_exchange_rate,
                description="ดูอัตราแลกเปลี่ยนเงินบาทกับดอลลาร์สหรัฐแบบเรียลไทม์"
            ),
            Tool(
                name="currency_converter",
                func=self.convert_currency_real,
                description="แปลงเงินระหว่างสกุลต่างๆ แบบเรียลไทม์ รูปแบบ: จำนวน,สกุลเงินต้นทาง,สกุลเงินปลายทาง"
            ),
            Tool(
                name="multiple_currencies",
                func=self.get_multiple_rates,
                description="ดูอัตราแลกเปลี่ยนหลายสกุลเงินพร้อมกัน เช่น USD,EUR,JPY,GBP"
            ),
            Tool(
                name="current_time",
                func=self.get_real_time,
                description="ดูเวลาปัจจุบันในประเทศไทยและเขตเวลาอื่นๆ"
            ),
            Tool(
                name="historical_rate",
                func=self.get_historical_rate,
                description="ดูอัตราแลกเปลี่ยนย้อนหลัง รูปแบบ: วันที่(YYYY-MM-DD),สกุลเงินต้นทาง,สกุลเงินปลายทาง"
            ),
            Tool(
                name="calculator",
                func=self.calculate,
                description="คำนวณทางคณิตศาสตร์พื้นฐาน"
            )
        ]
        return tools
    
    def get_real_exchange_rate(self, query: str) -> str:
        """Get real USD/THB exchange rate from API"""
        try:
            # Try primary API
            url = f"{self.exchange_api_base}/latest?base=USD&symbols=THB"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', True) and 'rates' in data:
                    rate = data['rates']['THB']
                    date = data.get('date', datetime.date.today().isoformat())
                    return f"อัตราแลกเปลี่ยน USD/THB (เรียลไทม์)\nวันที่: {date}\nอัตรา: {rate:.4f} บาท/ดอลลาร์สหรัฐ\nที่มา: exchangerate.host"
            
            # Try backup API
            backup_url = f"{self.backup_exchange_api}/currencies/usd.json"
            backup_response = requests.get(backup_url, timeout=10)
            
            if backup_response.status_code == 200:
                backup_data = backup_response.json()
                if 'usd' in backup_data and 'thb' in backup_data['usd']:
                    rate = backup_data['usd']['thb']
                    return f"อัตราแลกเปลี่ยน USD/THB (เรียลไทม์)\nอัตรา: {rate:.4f} บาท/ดอลลาร์สหรัฐ\nที่มา: fawazahmed0 API"
            
            return "ไม่สามารถเชื่อมต่อ API อัตราแลกเปลี่ยนได้ กรุณาลองใหม่อีกครั้ง"
            
        except requests.RequestException as e:
            return f"ข้อผิดพลาดการเชื่อมต่อ: {str(e)}"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def convert_currency_real(self, query: str) -> str:
        """Convert currency using real exchange rates"""
        try:
            parts = query.strip().split(',')
            if len(parts) != 3:
                return "รูปแบบ: จำนวน,สกุลเงินต้นทาง,สกุลเงินปลายทาง (เช่น 100,USD,THB)"
            
            amount = float(parts[0].strip())
            from_curr = parts[1].strip().upper()
            to_curr = parts[2].strip().upper()
            
            # Get real exchange rate
            url = f"{self.exchange_api_base}/latest?base={from_curr}&symbols={to_curr}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', True) and 'rates' in data and to_curr in data['rates']:
                    rate = data['rates'][to_curr]
                    result = amount * rate
                    date = data.get('date', datetime.date.today().isoformat())
                    
                    return f"การแปลงสกุลเงิน (เรียลไทม์)\n{amount:,.2f} {from_curr} = {result:,.2f} {to_curr}\nอัตรา: 1 {from_curr} = {rate:.4f} {to_curr}\nวันที่: {date}"
            
            # Try backup for USD/THB specifically
            if from_curr == "USD" and to_curr == "THB":
                backup_url = f"{self.backup_exchange_api}/currencies/usd.json"
                backup_response = requests.get(backup_url, timeout=10)
                if backup_response.status_code == 200:
                    backup_data = backup_response.json()
                    if 'usd' in backup_data and 'thb' in backup_data['usd']:
                        rate = backup_data['usd']['thb']
                        result = amount * rate
                        return f"{amount:,.2f} USD = {result:,.2f} THB (อัตรา: {rate:.4f})"
            
            return f"ไม่สามารถแปลง {from_curr} เป็น {to_curr} ได้ กรุณาตรวจสอบรหัสสกุลเงิน"
            
        except ValueError:
            return "จำนวนเงินไม่ถูกต้อง กรุณาใส่ตัวเลข"
        except requests.RequestException as e:
            return f"ข้อผิดพลาดการเชื่อมต่อ: {str(e)}"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_multiple_rates(self, query: str) -> str:
        """Get multiple currency rates against THB"""
        try:
            currencies = [curr.strip().upper() for curr in query.split(',')]
            if not currencies:
                currencies = ['USD', 'EUR', 'JPY', 'GBP']
            
            url = f"{self.exchange_api_base}/latest?base=THB&symbols={','.join(currencies)}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', True) and 'rates' in data:
                    rates = data['rates']
                    date = data.get('date', datetime.date.today().isoformat())
                    
                    result = f"อัตราแลกเปลี่ยนหลายสกุลเงิน (วันที่ {date})\nฐาน: 1 บาทไทย\n\n"
                    for curr, rate in rates.items():
                        # Convert to show how much THB for 1 unit of foreign currency
                        thb_per_unit = 1 / rate if rate != 0 else 0
                        result += f"1 {curr} = {thb_per_unit:.4f} THB\n"
                    
                    return result.strip()
            
            return "ไม่สามารถดึงข้อมูลอัตราแลกเปลี่ยนหลายสกุลเงินได้"
            
        except requests.RequestException as e:
            return f"ข้อผิดพลาดการเชื่อมต่อ: {str(e)}"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_historical_rate(self, query: str) -> str:
        """Get historical exchange rate"""
        try:
            parts = query.strip().split(',')
            if len(parts) != 3:
                return "รูปแบบ: วันที่(YYYY-MM-DD),สกุลเงินต้นทาง,สกุลเงินปลายทาง"
            
            date = parts[0].strip()
            from_curr = parts[1].strip().upper()
            to_curr = parts[2].strip().upper()
            
            # Validate date format
            try:
                datetime.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                return "รูปแบบวันที่ไม่ถูกต้อง ใช้ YYYY-MM-DD"
            
            url = f"{self.exchange_api_base}/{date}?base={from_curr}&symbols={to_curr}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', True) and 'rates' in data and to_curr in data['rates']:
                    rate = data['rates'][to_curr]
                    return f"อัตราแลกเปลี่ยนย้อนหลัง\nวันที่: {date}\n1 {from_curr} = {rate:.4f} {to_curr}"
            
            return f"ไม่พบข้อมูลอัตราแลกเปลี่ยนสำหรับวันที่ {date}"
            
        except requests.RequestException as e:
            return f"ข้อผิดพลาดการเชื่อมต่อ: {str(e)}"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_real_time(self, query: str) -> str:
        """Get real current time in Thailand and other timezones"""
        try:
            # Thailand timezone
            thai_tz = pytz.timezone('Asia/Bangkok')
            thai_time = datetime.datetime.now(thai_tz)
            
            # Other major timezones
            utc_time = datetime.datetime.now(pytz.UTC)
            ny_time = datetime.datetime.now(pytz.timezone('America/New_York'))
            london_time = datetime.datetime.now(pytz.timezone('Europe/London'))
            tokyo_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
            
            thai_day_names = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
            day_name = thai_day_names[thai_time.weekday()]
            
            result = f"เวลาปัจจุบัน (เรียลไทม์)\n\n"
            result += f"🇹🇭 ประเทศไทย: วัน{day_name} {thai_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            result += f"🌍 UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            result += f"🇺🇸 นิวยอร์ก: {ny_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            result += f"🇬🇧 ลอนดอน: {london_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            result += f"🇯🇵 โตเกียว: {tokyo_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            
            return result
            
        except Exception as e:
            return f"ไม่สามารถดึงข้อมูลเวลาได้: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        """Perform basic mathematical calculations"""
        try:
            expression = expression.strip()
            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return "รองรับเฉพาะการคำนวณพื้นฐาน (+, -, *, /, วงเล็บ)"
            
            result = eval(expression)
            return f"{expression} = {result:,.4f}"
        except ZeroDivisionError:
            return "ไม่สามารถหารด้วยศูนย์ได้"
        except Exception as e:
            return f"ไม่สามารถคำนวณได้: {str(e)}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor"""
        try:
            agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent="zero-shot-react-description",
                memory=self.memory,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True,
                early_stopping_method="generate"
            )
            return agent_executor
        except Exception as e:
            self.logger.error(f"Error creating agent: {e}")
            raise
    
    def chat(self, message: str) -> str:
        """Main chat interface"""
        try:
            self.logger.info(f"Processing message: {message}")
            response = self.agent_executor.invoke({"input": message})
            
            if isinstance(response, dict):
                output = response.get("output", str(response))
            else:
                output = str(response)
            
            self.logger.info("Response generated successfully")
            return output
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.logger.info("Memory cleared")
    
    def test_apis(self) -> str:
        """Test all APIs to ensure they're working"""
        results = []
        
        # Test exchange rate API
        try:
            url = f"{self.exchange_api_base}/latest?base=USD&symbols=THB"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results.append("✅ Exchange Rate API: Working")
            else:
                results.append(f"❌ Exchange Rate API: Error {response.status_code}")
        except Exception as e:
            results.append(f"❌ Exchange Rate API: {str(e)}")
        
        # Test backup API
        try:
            backup_url = f"{self.backup_exchange_api}/currencies/usd.json"
            backup_response = requests.get(backup_url, timeout=5)
            if backup_response.status_code == 200:
                results.append("✅ Backup Exchange API: Working")
            else:
                results.append(f"❌ Backup Exchange API: Error {backup_response.status_code}")
        except Exception as e:
            results.append(f"❌ Backup Exchange API: {str(e)}")
        
        return "\n".join(results)

def demo_real_data():
    """Demonstration with real data"""
    print("=== Thai Agent with Real Data Demo ===\n")
    
    agent = RealDataThaiAgent()
    
    # Test API connectivity first
    print("--- ทดสอบการเชื่อมต่อ API ---")
    print(agent.test_apis())
    print("-" * 50)
    
    # Test real data queries
    test_queries = [
        "อัตราแลกเปลี่ยน USD/THB วันนี้เป็นเท่าไร?",
        "แปลง 1000 USD เป็น THB",
        "ดูอัตราแลกเปลี่ยน EUR,JPY,GBP เทียบกับบาท",
        "เวลาปัจจุบันในไทยกี่โมง?",
        "อัตราแลกเปลี่ยน USD/THB วันที่ 2024-01-01",
        "คำนวณ 1500 * 36.5"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- คำถามที่ {i} ---")
        print(f"ผู้ใช้: {query}")
        try:
            response = agent.chat(query)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"ข้อผิดพลาด: {e}")
        print("-" * 50)

if __name__ == "__main__":
    # Install required packages first
    print("Required packages: requests pytz")
    print("Install with: pip install requests pytz\n")
    
    demo_real_data()