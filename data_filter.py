"""
Phase 2: Data Filter
Filters FinQA dataset to keep only examples with mathematical reasoning.
This is the core algorithm that ensures quality training data for financial calculations.
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def has_valid_program(item: Dict[Any, Any]) -> Tuple[bool, str]:
    """
    Check if an item has a valid program_re (mathematical reasoning steps)
    Returns (is_valid, reason)
    """
    program = item.get('program_re', '') or item.get('program', '')
    
    if not program or len(program.strip()) == 0:
        return False, "No program found"
    
    # Convert to string if it's a list
    if isinstance(program, list):
        program = '\n'.join(str(p) for p in program)
    
    program_str = str(program).lower()
    
    # Check for mathematical operations - these indicate computational reasoning
    math_patterns = [
        r'divide\s*\(',
        r'multiply\s*\(',
        r'add\s*\(',
        r'subtract\s*\(',
        r'greater\s*\(',
        r'less\s*\(',
        r'exp\s*\(',
        r'table_\w+\(',
        r'#\d+\s*=',  # Variable assignments like #0 = 
        r'\+|\-|\*|\/|\%',  # Basic math operators
    ]
    
    has_math = any(re.search(pattern, program_str, re.IGNORECASE) for pattern in math_patterns)
    
    if not has_math:
        return False, "No mathematical operations found"
    
    # Check for multistep reasoning (more than just a lookup)
    reasoning_indicators = [
        len(program.split('\n')) > 1,  # Multiple lines
        'divide(' in program_str and ('multiply(' in program_str or 'add(' in program_str),  # Multiple operations
        program_str.count('=') > 1,  # Multiple variable assignments
        'table_' in program_str and any(op in program_str for op in ['divide', 'multiply', 'add', 'subtract'])
    ]
    
    has_reasoning = any(reasoning_indicators)
    
    if not has_reasoning:
        return False, "Simple lookup, no multi-step reasoning"
    
    # Avoid examples that are too complex or malformed
    if len(program.split('\n')) > 10:
        return False, "Program too complex (>10 steps)"
    
    return True, "Valid mathematical reasoning"

def extract_reasoning_steps(program_re: str, question: str = "") -> List[str]:
    """
    Parse program_re into human-readable reasoning steps
    """
    if not program_re:
        return []
    
    # Convert to string if it's a list
    if isinstance(program_re, list):
        program_re = '\n'.join(str(p) for p in program_re)
    
    steps = []
    lines = str(program_re).strip().split('\n')
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Convert program operations to natural language
        step_text = ""
        
        if 'table_' in line.lower():
            if 'max(' in line.lower():
                step_text = f"Step {i}: Find the maximum value from the financial table"
            elif 'min(' in line.lower():
                step_text = f"Step {i}: Find the minimum value from the financial table"
            elif 'sum(' in line.lower():
                step_text = f"Step {i}: Sum values from the financial table"
            else:
                step_text = f"Step {i}: Extract data from the financial table"
                
        elif 'divide(' in line.lower():
            step_text = f"Step {i}: Perform division calculation"
        elif 'multiply(' in line.lower():
            step_text = f"Step {i}: Perform multiplication calculation"
        elif 'add(' in line.lower():
            step_text = f"Step {i}: Perform addition calculation"
        elif 'subtract(' in line.lower():
            step_text = f"Step {i}: Perform subtraction calculation"
        elif 'greater(' in line.lower():
            step_text = f"Step {i}: Compare values (greater than)"
        elif 'less(' in line.lower():
            step_text = f"Step {i}: Compare values (less than)"
        elif '#' in line and '=' in line:
            step_text = f"Step {i}: Store intermediate calculation result"
        else:
            # Keep original line but make it more readable
            clean_line = line.replace('(', ' (').replace(')', ') ').replace(',', ', ')
            step_text = f"Step {i}: {clean_line}"
        
        steps.append(step_text)
    
    return steps

def analyze_dataset_structure(filepath: str) -> Dict[str, Any]:
    """
    Analyze the structure of a dataset file to understand its format
    """
    logger.info(f"Analyzing structure of {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            return {"error": "Empty dataset"}
        
        # Analyze first few items
        sample_size = min(5, len(data))
        sample_items = data[:sample_size]
        
        # Get all unique keys
        all_keys = set()
        for item in sample_items:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        # Check for program fields
        program_fields = [key for key in all_keys if 'program' in key.lower()]
        
        # Sample values for key fields
        key_samples = {}
        important_keys = ['question', 'answer'] + program_fields + ['table', 'text']
        
        for key in important_keys:
            if key in all_keys:
                values = [item.get(key) for item in sample_items if item.get(key)]
                if values:
                    key_samples[key] = values[0]
        
        return {
            "total_items": len(data),
            "all_keys": sorted(list(all_keys)),
            "program_fields": program_fields,
            "key_samples": key_samples
        }
    
    except Exception as e:
        return {"error": str(e)}

def filter_financial_reasoning_data(input_file: str, output_file: str) -> int:
    """
    Filter dataset to keep only examples with mathematical reasoning
    """
    logger.info(f"Filtering {input_file}...")
    
    # First analyze the dataset structure
    structure = analyze_dataset_structure(input_file)
    if "error" in structure:
        logger.error(f"Error analyzing {input_file}: {structure['error']}")
        return 0
    
    logger.info(f"Dataset structure: {len(structure['all_keys'])} fields, {structure['total_items']} total items")
    if structure['program_fields']:
        logger.info(f"Program fields found: {structure['program_fields']}")
    else:
        logger.warning("No program fields found - this dataset may not contain reasoning steps")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {input_file}: {str(e)}")
        return 0
    
    filtered_data = []
    filter_stats = {
        "total": len(data),
        "no_program": 0,
        "no_math": 0,
        "simple_lookup": 0,
        "too_complex": 0,
        "valid": 0
    }
    
    for item in tqdm(data, desc="Filtering examples"):
        is_valid, reason = has_valid_program(item)
        
        # Update statistics
        if "No program found" in reason:
            filter_stats["no_program"] += 1
        elif "No mathematical operations found" in reason:
            filter_stats["no_math"] += 1
        elif "Simple lookup" in reason:
            filter_stats["simple_lookup"] += 1
        elif "too complex" in reason:
            filter_stats["too_complex"] += 1
        elif is_valid:
            filter_stats["valid"] += 1
        
        if is_valid:
            # Get the program field (try different possible names)
            program = item.get('program_re', '') or item.get('program', '') or item.get('reasoning', '')
            
            # Create enhanced item with reasoning steps
            enhanced_item = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'program_re': program,
                'reasoning_steps': extract_reasoning_steps(program, item.get('question', '')),
                'table': item.get('table', ''),
                'text': item.get('text', ''),
                'source': 'finqa',
                'original_keys': list(item.keys())  # Keep track of original structure
            }
            
            # Add any additional context fields that might be useful
            for key in ['context', 'gold_inds', 'question_type']:
                if key in item:
                    enhanced_item[key] = item[key]
            
            filtered_data.append(enhanced_item)
    
    # Save filtered data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    # Log detailed statistics
    logger.info(f"\n=== FILTERING RESULTS FOR {os.path.basename(input_file)} ===")
    logger.info(f"Total examples: {filter_stats['total']}")
    logger.info(f"No program field: {filter_stats['no_program']} ({filter_stats['no_program']/filter_stats['total']*100:.1f}%)")
    logger.info(f"No math operations: {filter_stats['no_math']} ({filter_stats['no_math']/filter_stats['total']*100:.1f}%)")
    logger.info(f"Simple lookups: {filter_stats['simple_lookup']} ({filter_stats['simple_lookup']/filter_stats['total']*100:.1f}%)")
    logger.info(f"Too complex: {filter_stats['too_complex']} ({filter_stats['too_complex']/filter_stats['total']*100:.1f}%)")
    logger.info(f"Valid reasoning: {filter_stats['valid']} ({filter_stats['valid']/filter_stats['total']*100:.1f}%)")
    logger.info(f"Saved to: {output_file}")
    
    return len(filtered_data)

def main():
    """Filter FinQA dataset"""
    logger.info("=== PHASE 2: DATA FILTERING ===")
    
    os.makedirs("filtered_data", exist_ok=True)
    
    total_filtered = 0
    datasets_processed = 0
    
    # Process all JSON files in raw_data directory
    raw_data_dir = "raw_data"
    
    if not os.path.exists(raw_data_dir):
        logger.error(f"Raw data directory {raw_data_dir} not found!")
        logger.info("Please run datasets_downloader.py first")
        return False
    
    json_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
    
    if not json_files:
        logger.error("No JSON files found in raw_data directory!")
        return False
    
    logger.info(f"Found {len(json_files)} dataset files to process")
    
    for filename in sorted(json_files):
        input_file = os.path.join(raw_data_dir, filename)
        
        # Create output filename
        base_name = filename.replace('.json', '')
        output_file = f"filtered_data/{base_name}_filtered.json"
        
        logger.info(f"\nProcessing: {filename}")
        
        try:
            count = filter_financial_reasoning_data(input_file, output_file)
            total_filtered += count
            datasets_processed += 1
            
            logger.info(f"✅ Successfully filtered {filename}: {count} examples")
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}")
    
    # Final summary
    logger.info(f"\n=== FILTERING COMPLETE ===")
    logger.info(f"Datasets processed: {datasets_processed}/{len(json_files)}")
    logger.info(f"Total examples with mathematical reasoning: {total_filtered}")
    
    if total_filtered > 0:
        logger.info("✅ Data filtering successful!")
        logger.info("Next step: Run data_preprocessor.py to convert to training format")
        return True
    else:
        logger.error("❌ No valid examples found!")
        return False

if __name__ == "__main__":
    main()