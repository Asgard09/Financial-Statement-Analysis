"""
Phase 2: Data Preprocessor
Converts filtered financial reasoning data to instruction/input/response format for model training.
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_table_data(table_data: Any) -> str:
    """
    Convert table data to a readable string format
    """
    if not table_data:
        return ""
    
    # Handle different table formats
    if isinstance(table_data, str):
        # Already a string, just clean it up
        return table_data.strip()
    
    elif isinstance(table_data, dict):
        # Dictionary format - convert to readable table
        if 'header' in table_data and 'rows' in table_data:
            # Structured table format
            header = table_data.get('header', [])
            rows = table_data.get('rows', [])
            
            if not header or not rows:
                return str(table_data)
            
            # Create table string
            table_str = " | ".join(str(h) for h in header) + "\n"
            table_str += " | ".join(["-" * len(str(h)) for h in header]) + "\n"
            
            for row in rows[:10]:  # Limit to 10 rows for readability
                if isinstance(row, list):
                    table_str += " | ".join(str(cell) for cell in row) + "\n"
                else:
                    table_str += str(row) + "\n"
            
            if len(rows) > 10:
                table_str += f"... ({len(rows) - 10} more rows)\n"
            
            return table_str.strip()
        else:
            # Generic dictionary - convert to key-value pairs
            items = []
            for key, value in table_data.items():
                if isinstance(value, (list, dict)):
                    value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                items.append(f"{key}: {value}")
            return "\n".join(items)
    
    elif isinstance(table_data, list):
        # List format
        if not table_data:
            return ""
        
        # Check if it's a list of lists (rows)
        if isinstance(table_data[0], list):
            table_str = ""
            for i, row in enumerate(table_data[:10]):
                table_str += " | ".join(str(cell) for cell in row) + "\n"
            
            if len(table_data) > 10:
                table_str += f"... ({len(table_data) - 10} more rows)\n"
            
            return table_str.strip()
        else:
            # List of values
            return " | ".join(str(item) for item in table_data[:20])
    
    else:
        return str(table_data)

def create_instruction_format(item: Dict, instruction_type: str = "standard") -> Dict:
    """
    Convert filtered data to Instruction/Input/Response format
    """
    # Build context from table and text
    context_parts = []
    
    # Add table data if available
    table_content = format_table_data(item.get('table'))
    if table_content:
        context_parts.append(f"Financial Data Table:\n{table_content}")
    
    # Add text context if available
    text_content = item.get('text', '').strip()
    if text_content:
        # Limit text length to avoid overly long inputs
        if len(text_content) > 1000:
            text_content = text_content[:1000] + "... [truncated]"
        context_parts.append(f"Additional Context:\n{text_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create different instruction variations for diversity
    instructions = {
        "standard": """You are a financial analyst AI. Given the financial data and question below, provide step-by-step reasoning to calculate the answer. Show your mathematical steps clearly and explain your reasoning.""",
        
        "detailed": """You are an expert financial analyst. Analyze the provided financial data and answer the question with detailed step-by-step calculations. Break down each mathematical operation and explain the financial reasoning behind each step.""",
        
        "concise": """Analyze the financial data and answer the question. Show the calculation steps and provide the final numerical answer.""",
        
        "educational": """As a financial tutor, help solve this problem by showing clear step-by-step calculations. Explain each step so that someone learning financial analysis can understand the process."""
    }
    
    # Select instruction based on type
    instruction = instructions.get(instruction_type, instructions["standard"])
    
    # Create the input
    input_parts = [f"Question: {item['question']}"]
    if context:
        input_parts.append(context)
    
    input_text = "\n\n".join(input_parts)
    
    # Create the response with reasoning steps
    reasoning_steps = item.get('reasoning_steps', [])
    if reasoning_steps:
        reasoning_text = "\n".join(reasoning_steps)
    else:
        # Fallback to program_re if no reasoning steps
        program = item.get('program_re', '')
        reasoning_text = f"Calculation steps:\n{program}"
    
    # Format the final answer
    answer = item.get('answer', '')
    if isinstance(answer, (int, float)):
        answer_text = f"Final Answer: {answer}"
    elif isinstance(answer, str) and answer.strip():
        answer_text = f"Final Answer: {answer.strip()}"
    else:
        answer_text = "Final Answer: [See calculation above]"
    
    response = f"{reasoning_text}\n\n{answer_text}"
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": response,
        "source": item.get('source', 'unknown'),
        "question_length": len(item.get('question', '')),
        "has_table": bool(item.get('table')),
        "has_text": bool(item.get('text', '').strip()),
        "reasoning_steps_count": len(reasoning_steps)
    }

def validate_training_example(example: Dict) -> bool:
    """
    Validate that a training example meets quality standards
    """
    # Check required fields
    required_fields = ['instruction', 'input', 'output']
    if not all(field in example and example[field].strip() for field in required_fields):
        return False
    
    # Check minimum lengths
    if len(example['input']) < 20:  # Too short input
        return False
    
    if len(example['output']) < 30:  # Too short output
        return False
    
    # Check for mathematical content in output
    math_indicators = ['step', 'calculate', 'divide', 'multiply', 'add', 'subtract', 'answer']
    has_math = any(indicator in example['output'].lower() for indicator in math_indicators)
    
    if not has_math:
        return False
    
    return True

def preprocess_for_training(
    input_dir: str = "filtered_data", 
    output_file: str = "training/finqa_clean.json",
    max_examples: int = None,
    instruction_variety: bool = True
) -> int:
    """
    Convert all filtered data to training format
    """
    logger.info("Converting filtered data to training format...")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} not found!")
        return 0
    
    training_data = []
    file_stats = {}
    
    # Get all filtered files
    filtered_files = [f for f in os.listdir(input_dir) if f.endswith('_filtered.json')]
    
    if not filtered_files:
        logger.error(f"No filtered JSON files found in {input_dir}")
        logger.info("Please run data_filter.py first")
        return 0
    
    logger.info(f"Processing {len(filtered_files)} filtered files...")
    
    # Process each filtered file
    for filename in sorted(filtered_files):
        filepath = os.path.join(input_dir, filename)
        logger.info(f"Processing {filename}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_examples = 0
            
            for item in tqdm(data, desc=f"Converting {filename}"):
                # Randomly vary instruction type for diversity
                if instruction_variety:
                    instruction_types = ["standard", "detailed", "concise", "educational"]
                    instruction_type = random.choice(instruction_types)
                else:
                    instruction_type = "standard"
                
                try:
                    formatted_item = create_instruction_format(item, instruction_type)
                    
                    # Validate the example
                    if validate_training_example(formatted_item):
                        training_data.append(formatted_item)
                        file_examples += 1
                        
                        # Check if we've reached max examples
                        if max_examples and len(training_data) >= max_examples:
                            logger.info(f"Reached maximum examples limit: {max_examples}")
                            break
                    
                except Exception as e:
                        logger.warning(f"Error processing item: {str(e)}")
                        continue
            
            file_stats[filename] = file_examples
            logger.info(f"Converted {file_examples} examples from {filename}")
            
            # Break if we've reached max examples
            if max_examples and len(training_data) >= max_examples:
                break
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    if not training_data:
        logger.error("No valid training examples created!")
        return 0
    
    # Shuffle the training data for better training
    random.shuffle(training_data)
    
    # Save training data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Create summary statistics
    create_preprocessing_summary(training_data, file_stats, output_file)
    
    logger.info(f"✅ Created {len(training_data)} training examples")
    logger.info(f"📁 Saved to {output_file}")
    
    return len(training_data)

def create_preprocessing_summary(training_data: List[Dict], file_stats: Dict, output_file: str):
    """
    Create a summary of the preprocessing results
    """
    summary = {
        "total_examples": len(training_data),
        "source_breakdown": {},
        "quality_metrics": {
            "avg_input_length": 0,
            "avg_output_length": 0,
            "examples_with_tables": 0,
            "examples_with_text": 0,
            "avg_reasoning_steps": 0
        },
        "file_breakdown": file_stats
    }
    
    # Calculate metrics
    if training_data:
        # Source breakdown
        for item in training_data:
            source = item.get('source', 'unknown')
            summary["source_breakdown"][source] = summary["source_breakdown"].get(source, 0) + 1
        
        # Quality metrics
        total_input_len = sum(len(item['input']) for item in training_data)
        total_output_len = sum(len(item['output']) for item in training_data)
        
        summary["quality_metrics"]["avg_input_length"] = int(total_input_len / len(training_data))
        summary["quality_metrics"]["avg_output_length"] = int(total_output_len / len(training_data))
        summary["quality_metrics"]["examples_with_tables"] = sum(1 for item in training_data if item.get('has_table', False))
        summary["quality_metrics"]["examples_with_text"] = sum(1 for item in training_data if item.get('has_text', False))
        
        reasoning_steps = [item.get('reasoning_steps_count', 0) for item in training_data]
        if reasoning_steps:
            summary["quality_metrics"]["avg_reasoning_steps"] = sum(reasoning_steps) / len(reasoning_steps)
    
    # Save summary
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Log summary
    logger.info("\n=== PREPROCESSING SUMMARY ===")
    logger.info(f"Total examples: {summary['total_examples']}")
    logger.info(f"Source breakdown: {summary['source_breakdown']}")
    logger.info(f"Average input length: {summary['quality_metrics']['avg_input_length']} characters")
    logger.info(f"Average output length: {summary['quality_metrics']['avg_output_length']} characters")
    logger.info(f"Examples with tables: {summary['quality_metrics']['examples_with_tables']}")
    logger.info(f"Examples with text context: {summary['quality_metrics']['examples_with_text']}")
    logger.info(f"Average reasoning steps: {summary['quality_metrics']['avg_reasoning_steps']:.1f}")
    
    logger.info(f"📊 Summary saved to: {summary_file}")

def main():
    """Main preprocessing function"""
    logger.info("=== PHASE 2: DATA PREPROCESSING ===")
    
    # Allow customization via command line or defaults
    max_examples = None  # Set to a number like 10000 to limit dataset size
    
    examples_created = preprocess_for_training(
        input_dir="filtered_data",
        output_file="training/finqa_clean.json",
        max_examples=max_examples,
        instruction_variety=True
    )
    
    if examples_created > 0:
        logger.info("✅ Preprocessing complete!")
        logger.info("Next step: Run data_validator.py to validate training data quality")
        
        # Show sample example
        try:
            with open("training/finqa_clean.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data:
                logger.info("\n=== SAMPLE TRAINING EXAMPLE ===")
                sample = data[0]
                logger.info(f"Instruction: {sample['instruction'][:100]}...")
                logger.info(f"Input: {sample['input'][:150]}...")
                logger.info(f"Output: {sample['output'][:150]}...")
                logger.info(f"Source: {sample.get('source', 'unknown')}")
        
        except Exception as e:
            logger.warning(f"Could not load sample: {str(e)}")
        
        return True
    else:
        logger.error("❌ Preprocessing failed!")
        return False

if __name__ == "__main__":
    main()