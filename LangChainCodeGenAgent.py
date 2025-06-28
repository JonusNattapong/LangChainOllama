from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import ast
import subprocess
import tempfile
import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeResponse(BaseModel):
    """Data model for code generation response"""
    code: str = Field(description="Generated Python code")
    explanation: Optional[str] = Field(description="Explanation of the code")
    dependencies: Optional[List[str]] = Field(description="Required dependencies", default=[])
    complexity: Optional[str] = Field(description="Code complexity level", default="medium")

class CodeGenAgent:
    def __init__(self, model_name="llama3.2:3b", temperature=0.2, validate_syntax=True):
        """
        Initialize the code generation agent
        
        Args:
            model_name: Ollama model name
            temperature: Model temperature (lower = more deterministic)
            validate_syntax: Whether to validate Python syntax
        """
        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        self.validate_syntax = validate_syntax
        self.parser = PydanticOutputParser(pydantic_object=CodeResponse)
        
        # Enhanced prompt with better structure
        self.prompt = PromptTemplate(
            input_variables=["description", "style", "complexity"],
            template="""คุณเป็นผู้เชี่ยวชาญด้านการเขียนโปรแกรม Python ที่มีประสบการณ์มากมาย

คำอธิบายงาน: {description}
สไตล์โค้ด: {style}
ระดับความซับซ้อน: {complexity}

กรุณาเขียนโค้ด Python ที่:
1. ถูกต้องตามหลักไวยากรณ์ (syntax)
2. มีความชัดเจนและอ่านง่าย
3. ใช้ชื่อตัวแปรและฟังก์ชันที่สื่อความหมาย
4. มี docstring และ comments ที่เหมาะสม
5. จัดการ error handling ที่จำเป็น
6. ปฏิบัติตาม PEP 8 style guide

ตัวอย่างรูปแบบการตอบ:
```python
def example_function(param1, param2):
    \"\"\"
    คำอธิบายฟังก์ชัน
    
    Args:
        param1: คำอธิบายพารามิเตอร์ 1
        param2: คำอธิบายพารามิเตอร์ 2
        
    Returns:
        คำอธิบายค่าที่ return
    \"\"\"
    # โค้ดหลัก
    return result

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    result = example_function("test", 123)
    print(result)
```

ข้อมูลเพิ่มเติม:
- อธิบายการทำงานของโค้ดโดยย่อ
- ระบุ dependencies ที่จำเป็น (ถ้ามี)
- ระดับความซับซ้อน: {complexity}

{format_instructions}

โค้ด Python:""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def generate_code(self, description: str, style: str = "functional", complexity: str = "medium") -> Dict:
        """
        Generate Python code based on description
        
        Args:
            description: Description of what the code should do
            style: Code style (functional, oop, procedural)
            complexity: Complexity level (simple, medium, advanced)
            
        Returns:
            Dictionary containing generated code and metadata
        """
        try:
            logger.info(f"Generating code for: {description[:100]}...")
            
            # Create chain and generate code
            chain = self.prompt | self.llm
            response = chain.invoke({
                "description": description,
                "style": style,
                "complexity": complexity
            })
            
            logger.info("Code generated successfully")
            
            # Try to parse structured response
            try:
                parsed_response = self.parser.parse(response)
                result = parsed_response.dict()
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured response: {parse_error}")
                result = self._extract_code_from_response(response)
            
            # Validate syntax if enabled
            if self.validate_syntax and 'code' in result:
                result['syntax_valid'] = self._validate_syntax(result['code'])
                if not result['syntax_valid']:
                    logger.warning("Generated code has syntax errors")
            
            result['generated_at'] = datetime.now().isoformat()
            result['model_used'] = self.llm.model
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}

    def _extract_code_from_response(self, response: str) -> Dict:
        """
        Extract code from raw response when structured parsing fails
        """
        result = {
            "code": "",
            "explanation": "",
            "dependencies": [],
            "complexity": "medium"
        }
        
        # Extract code blocks
        lines = response.split('\n')
        in_code_block = False
        code_lines = []
        explanation_lines = []
        
        for line in lines:
            if '```python' in line or '```' in line:
                in_code_block = not in_code_block
                continue
                
            if in_code_block:
                code_lines.append(line)
            else:
                if line.strip() and not line.startswith('#'):
                    explanation_lines.append(line.strip())
        
        result["code"] = '\n'.join(code_lines) if code_lines else response
        result["explanation"] = ' '.join(explanation_lines)
        
        return result

    def _validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax
        
        Args:
            code: Python code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False

    def execute_code(self, code: str, timeout: int = 10) -> Dict:
        """
        Execute generated code safely in a temporary environment
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_result = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            # Clean up
            os.unlink(temp_file)
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Code execution timed out"}
        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    def generate_and_test(self, description: str, test_cases: List[Dict] = None) -> Dict:
        """
        Generate code and run test cases
        
        Args:
            description: Code description
            test_cases: List of test cases with input/expected output
            
        Returns:
            Complete results including code and test results
        """
        # Generate code
        code_result = self.generate_code(description)
        
        if 'error' in code_result:
            return code_result
        
        result = code_result.copy()
        
        # Execute code
        if 'code' in result:
            execution_result = self.execute_code(result['code'])
            result['execution'] = execution_result
            
            # Run test cases if provided
            if test_cases and execution_result.get('success'):
                result['test_results'] = self._run_test_cases(result['code'], test_cases)
        
        return result

    def _run_test_cases(self, code: str, test_cases: List[Dict]) -> List[Dict]:
        """
        Run test cases against generated code
        """
        test_results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Create test code
                test_code = f"""
{code}

# Test case {i+1}
try:
    result = {test_case.get('call', 'main()')}
    expected = {test_case.get('expected', 'None')}
    print(f"Test {i+1}: {'PASS' if result == expected else 'FAIL'}")
    print(f"Expected: {{expected}}")
    print(f"Got: {{result}}")
except Exception as e:
    print(f"Test {i+1}: ERROR - {{e}}")
"""
                
                execution = self.execute_code(test_code)
                test_results.append({
                    "test_case": i+1,
                    "success": execution.get('success', False),
                    "output": execution.get('stdout', ''),
                    "error": execution.get('stderr', '')
                })
            except Exception as e:
                test_results.append({
                    "test_case": i+1,
                    "success": False,
                    "error": str(e)
                })
        
        return test_results

    def save_code(self, code_result: Dict, filename: str = None) -> str:
        """
        Save generated code to file
        
        Args:
            code_result: Result from generate_code
            filename: Output filename (auto-generated if None)
            
        Returns:
            Filename of saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_code_{timestamp}.py"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write(f"# Generated by CodeGenAgent\n")
                f.write(f"# Generated at: {code_result.get('generated_at', 'unknown')}\n")
                f.write(f"# Description: {code_result.get('explanation', 'No description')}\n")
                
                if code_result.get('dependencies'):
                    f.write(f"# Dependencies: {', '.join(code_result['dependencies'])}\n")
                
                f.write("\n")
                f.write(code_result.get('code', ''))
            
            logger.info(f"Code saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save code: {e}")
            return None

def main():
    """
    Main function with enhanced CLI interface
    """
    parser = argparse.ArgumentParser(
        description="Enhanced CodeGenAgent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codegen_agent.py --desc "เขียนฟังก์ชันหาค่าเฉลี่ย"
  python codegen_agent.py --desc "สร้าง class สำหรับจัดการข้อมูลนักเรียน" --style oop
  python codegen_agent.py --desc "เขียนโปรแกรมคำนวณ fibonacci" --complexity advanced --execute
        """
    )
    
    parser.add_argument("--desc", type=str, help="คำอธิบายสำหรับสร้างโค้ด Python")
    parser.add_argument("--style", choices=["functional", "oop", "procedural"], 
                       default="functional", help="สไตล์การเขียนโค้ด")
    parser.add_argument("--complexity", choices=["simple", "medium", "advanced"], 
                       default="medium", help="ระดับความซับซ้อน")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--save", action="store_true", help="บันทึกโค้ดลงไฟล์")
    parser.add_argument("--execute", action="store_true", help="รันโค้ดที่สร้างขึ้น")
    parser.add_argument("--output", type=str, help="ชื่อไฟล์สำหรับบันทึกผลลัพธ์")
    
    args = parser.parse_args()
    
    # Create agent
    agent = CodeGenAgent(model_name=args.model)
    
    # Default description if not provided
    description = args.desc or "เขียนฟังก์ชัน Python หาค่าเฉลี่ยของ list"
    
    print(f"🤖 กำลังสร้างโค้ด Python...")
    print(f"📝 คำอธิบาย: {description}")
    print(f"🎨 สไตล์: {args.style}")
    print(f"⚡ ความซับซ้อน: {args.complexity}")
    print("=" * 60)
    
    # Generate code
    if args.execute:
        result = agent.generate_and_test(description)
    else:
        result = agent.generate_code(description, args.style, args.complexity)
    
    # Display results
    if 'error' in result:
        print(f"❌ เกิดข้อผิดพลาด: {result['error']}")
        return
    
    print("✅ สร้างโค้ดสำเร็จ!")
    print("\n📋 โค้ดที่สร้างขึ้น:")
    print("-" * 40)
    print(result.get('code', 'No code generated'))
    print("-" * 40)
    
    # Show explanation
    if result.get('explanation'):
        print(f"\n📖 คำอธิบาย: {result['explanation']}")
    
    # Show dependencies
    if result.get('dependencies'):
        print(f"\n📦 Dependencies: {', '.join(result['dependencies'])}")
    
    # Show syntax validation
    if 'syntax_valid' in result:
        status = "✅ ถูกต้อง" if result['syntax_valid'] else "❌ มีข้อผิดพลาด"
        print(f"\n🔍 ตรวจสอบ Syntax: {status}")
    
    # Show execution results
    if 'execution' in result:
        exec_result = result['execution']
        if exec_result.get('success'):
            print(f"\n🚀 รันโค้ดสำเร็จ!")
            if exec_result.get('stdout'):
                print("Output:")
                print(exec_result['stdout'])
        else:
            print(f"\n❌ รันโค้ดไม่สำเร็จ:")
            if exec_result.get('stderr'):
                print(exec_result['stderr'])
    
    # Save code if requested
    if args.save:
        filename = agent.save_code(result, args.output)
        if filename:
            print(f"\n💾 บันทึกโค้ดไปที่: {filename}")
    
    # Save full results as JSON
    if args.output and args.output.endswith('.json'):
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n📄 บันทึกผลลัพธ์ทั้งหมดไปที่: {args.output}")
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกผลลัพธ์: {e}")

if __name__ == "__main__":
    main()