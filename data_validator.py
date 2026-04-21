"""
Phase 2: Data Validator
Validates the quality and completeness of training data for financial reasoning model.
"""

import json
import re
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_training_data(data_file: str = "training/finqa_clean.json") -> Dict[str, Any]:
    """
    Comprehensive validation of training data quality
    """
    logger.info(f"Validating training data: {data_file}")
    
    if not os.path.exists(data_file):
        logger.error(f"Training data file not found: {data_file}")
        return {"error": f"File not found: {data_file}"}
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return {"error": f"Failed to load data: {str(e)}"}
    
    if not data:
        logger.error("Training data is empty!")
        return {"error": "Empty dataset"}
    
    logger.info(f"Loaded {len(data)} training examples")
    
    validation_results = {
        "total_examples": len(data),
        "field_completeness": {},
        "content_quality": {},
        "reasoning_analysis": {},
        "mathematical_operations": {},
        "length_statistics": {},
        "source_distribution": {},
        "quality_issues": [],
        "recommendations": []
    }
    
    # Validate field completeness
    validation_results["field_completeness"] = validate_field_completeness(data)
    
    # Analyze content quality
    validation_results["content_quality"] = analyze_content_quality(data)
    
    # Analyze reasoning patterns
    validation_results["reasoning_analysis"] = analyze_reasoning_patterns(data)
    
    # Check mathematical operations coverage
    validation_results["mathematical_operations"] = analyze_mathematical_operations(data)
    
    # Analyze length statistics
    validation_results["length_statistics"] = analyze_length_statistics(data)
    
    # Check source distribution
    validation_results["source_distribution"] = analyze_source_distribution(data)
    
    # Identify quality issues and recommendations
    validation_results["quality_issues"] = identify_quality_issues(data, validation_results)
    validation_results["recommendations"] = generate_recommendations(validation_results)
    
    return validation_results

def validate_field_completeness(data: List[Dict]) -> Dict[str, Any]:
    """Check completeness of required fields"""
    logger.info("Validating field completeness...")
    
    required_fields = ['instruction', 'input', 'output']
    optional_fields = ['source', 'has_table', 'has_text', 'reasoning_steps_count']
    
    field_stats = {}
    
    for field in required_fields + optional_fields:
        present_count = sum(1 for item in data if field in item and item[field])
        field_stats[field] = {
            "present": present_count,
            "missing": len(data) - present_count,
            "percentage": (present_count / len(data)) * 100
        }
    
    return field_stats

def analyze_content_quality(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the quality of content in each field"""
    logger.info("Analyzing content quality...")
    
    quality_metrics = {
        "empty_instructions": 0,
        "empty_inputs": 0,
        "empty_outputs": 0,
        "very_short_inputs": 0,  # < 50 chars
        "very_short_outputs": 0,  # < 100 chars
        "very_long_inputs": 0,  # > 2000 chars
        "very_long_outputs": 0,  # > 1000 chars
        "malformed_examples": 0
    }
    
    for item in data:
        instruction = item.get('instruction', '').strip()
        input_text = item.get('input', '').strip()
        output_text = item.get('output', '').strip()
        
        # Check for empty fields
        if not instruction:
            quality_metrics["empty_instructions"] += 1
        if not input_text:
            quality_metrics["empty_inputs"] += 1
        if not output_text:
            quality_metrics["empty_outputs"] += 1
        
        # Check lengths
        if len(input_text) < 50:
            quality_metrics["very_short_inputs"] += 1
        if len(output_text) < 100:
            quality_metrics["very_short_outputs"] += 1
        if len(input_text) > 2000:
            quality_metrics["very_long_inputs"] += 1
        if len(output_text) > 1000:
            quality_metrics["very_long_outputs"] += 1
        
        # Check for malformed examples
        if not instruction or not input_text or not output_text:
            quality_metrics["malformed_examples"] += 1
    
    # Convert to percentages
    total = len(data)
    for key in quality_metrics:
        count = quality_metrics[key]
        quality_metrics[key] = {
            "count": count,
            "percentage": (count / total) * 100
        }
    
    return quality_metrics

def analyze_reasoning_patterns(data: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns in reasoning steps"""
    logger.info("Analyzing reasoning patterns...")
    
    step_patterns = []
    reasoning_quality = {
        "has_step_keywords": 0,
        "has_calculations": 0,
        "has_final_answer": 0,
        "multi_step_reasoning": 0
    }
    
    step_keywords = ['step', 'first', 'second', 'next', 'then', 'finally', 'calculate', 'divide', 'multiply']
    calculation_keywords = ['=', '+', '-', '*', '/', 'divide', 'multiply', 'add', 'subtract', 'calculate']
    answer_keywords = ['final answer', 'answer:', 'result:', 'solution:']
    
    for item in data:
        output = item.get('output', '').lower()
        
        # Count reasoning steps
        steps = len(re.findall(r'step \d+', output))
        step_patterns.append(steps)
        
        # Check quality indicators
        if any(keyword in output for keyword in step_keywords):
            reasoning_quality["has_step_keywords"] += 1
        
        if any(keyword in output for keyword in calculation_keywords):
            reasoning_quality["has_calculations"] += 1
        
        if any(keyword in output for keyword in answer_keywords):
            reasoning_quality["has_final_answer"] += 1
        
        if steps > 1 or len(output.split('\n')) > 3:
            reasoning_quality["multi_step_reasoning"] += 1
    
    return {
        "step_distribution": dict(Counter(step_patterns)),
        "avg_steps": sum(step_patterns) / len(step_patterns) if step_patterns else 0,
        "max_steps": max(step_patterns) if step_patterns else 0,
        "quality_indicators": {
            key: {
                "count": value,
                "percentage": (value / len(data)) * 100
            }
            for key, value in reasoning_quality.items()
        }
    }

def analyze_mathematical_operations(data: List[Dict]) -> Dict[str, Any]:
    """Analyze coverage of different mathematical operations"""
    logger.info("Analyzing mathematical operations coverage...")
    
    operations = {
        'addition': ['add', '+', 'plus', 'sum'],
        'subtraction': ['subtract', '-', 'minus', 'difference'],
        'multiplication': ['multiply', '*', 'times', 'product'],
        'division': ['divide', '/', 'divided by', 'ratio'],
        'percentage': ['%', 'percent', 'percentage'],
        'comparison': ['greater', 'less', 'compare', 'higher', 'lower'],
        'average': ['average', 'mean'],
        'growth': ['growth', 'increase', 'decrease', 'change']
    }
    
    operation_counts = {op: 0 for op in operations.keys()}
    
    for item in data:
        output = item.get('output', '').lower()
        input_text = item.get('input', '').lower()
        combined_text = output + " " + input_text
        
        for operation, keywords in operations.items():
            if any(keyword in combined_text for keyword in keywords):
                operation_counts[operation] += 1
    
    # Convert to percentages
    total = len(data)
    operation_coverage = {}
    for operation, count in operation_counts.items():
        operation_coverage[operation] = {
            "count": count,
            "percentage": (count / total) * 100
        }
    
    return operation_coverage

def analyze_length_statistics(data: List[Dict]) -> Dict[str, Any]:
    """Analyze length statistics for inputs and outputs"""
    logger.info("Analyzing length statistics...")
    
    input_lengths = [len(item.get('input', '')) for item in data]
    output_lengths = [len(item.get('output', '')) for item in data]
    
    def get_stats(lengths):
        if not lengths:
            return {}
        return {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths)//2]
        }
    
    return {
        "input_lengths": get_stats(input_lengths),
        "output_lengths": get_stats(output_lengths),
        "length_distribution": {
            "very_short_inputs": sum(1 for l in input_lengths if l < 100),
            "short_inputs": sum(1 for l in input_lengths if 100 <= l < 300),
            "medium_inputs": sum(1 for l in input_lengths if 300 <= l < 800),
            "long_inputs": sum(1 for l in input_lengths if l >= 800),
            "very_short_outputs": sum(1 for l in output_lengths if l < 150),
            "short_outputs": sum(1 for l in output_lengths if 150 <= l < 400),
            "medium_outputs": sum(1 for l in output_lengths if 400 <= l < 800),
            "long_outputs": sum(1 for l in output_lengths if l >= 800),
        }
    }

def analyze_source_distribution(data: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of data sources"""
    logger.info("Analyzing source distribution...")
    
    sources = [item.get('source', 'unknown') for item in data]
    source_counts = Counter(sources)
    
    total = len(data)
    source_distribution = {}
    
    for source, count in source_counts.items():
        source_distribution[source] = {
            "count": count,
            "percentage": (count / total) * 100
        }
    
    return source_distribution

def identify_quality_issues(data: List[Dict], validation_results: Dict) -> List[str]:
    """Identify potential quality issues in the dataset"""
    logger.info("Identifying quality issues...")
    
    issues = []
    
    # Check field completeness issues
    field_stats = validation_results["field_completeness"]
    for field in ['instruction', 'input', 'output']:
        if field_stats[field]["percentage"] < 100:
            issues.append(f"Missing {field} in {field_stats[field]['missing']} examples ({100-field_stats[field]['percentage']:.1f}%)")
    
    # Check content quality issues
    quality = validation_results["content_quality"]
    if quality["very_short_outputs"]["percentage"] > 10:
        issues.append(f"Too many very short outputs ({quality['very_short_outputs']['percentage']:.1f}%)")
    
    if quality["malformed_examples"]["count"] > 0:
        issues.append(f"Found {quality['malformed_examples']['count']} malformed examples")
    
    # Check mathematical operations coverage
    math_ops = validation_results["mathematical_operations"]
    low_coverage_ops = [op for op, stats in math_ops.items() if stats["percentage"] < 20]
    if low_coverage_ops:
        issues.append(f"Low coverage for mathematical operations: {', '.join(low_coverage_ops)}")
    
    # Check reasoning quality
    reasoning = validation_results["reasoning_analysis"]["quality_indicators"]
    if reasoning["has_calculations"]["percentage"] < 80:
        issues.append(f"Only {reasoning['has_calculations']['percentage']:.1f}% of examples contain clear calculations")
    
    if reasoning["has_final_answer"]["percentage"] < 90:
        issues.append(f"Only {reasoning['has_final_answer']['percentage']:.1f}% of examples have clear final answers")
    
    return issues

def generate_recommendations(validation_results: Dict) -> List[str]:
    """Generate recommendations based on validation results"""
    logger.info("Generating recommendations...")
    
    recommendations = []
    
    total_examples = validation_results["total_examples"]
    
    # Dataset size recommendations
    if total_examples < 1000:
        recommendations.append("Consider increasing dataset size (current: {total_examples}). 5,000+ examples recommended for good performance.")
    elif total_examples < 5000:
        recommendations.append("Dataset size is adequate but could be larger for better performance.")
    
    # Mathematical operations recommendations
    math_ops = validation_results["mathematical_operations"]
    for operation, stats in math_ops.items():
        if stats["percentage"] < 15:
            recommendations.append(f"Consider adding more examples with {operation} operations (current: {stats['percentage']:.1f}%)")
    
    # Reasoning quality recommendations
    reasoning = validation_results["reasoning_analysis"]["quality_indicators"]
    if reasoning["multi_step_reasoning"]["percentage"] < 70:
        recommendations.append("Add more examples with multi-step reasoning for better model training")
    
    # Source diversity recommendations
    sources = validation_results["source_distribution"]
    if len(sources) == 1:
        recommendations.append("Consider adding examples from multiple data sources for better diversity")
    
    # Length recommendations
    lengths = validation_results["length_statistics"]
    if lengths["output_lengths"]["avg"] < 200:
        recommendations.append("Consider examples with more detailed explanations (current avg output: {lengths['output_lengths']['avg']:.0f} chars)")
    
    return recommendations

def create_validation_report(validation_results: Dict, output_file: str = None):
    """Create a detailed validation report"""
    if output_file is None:
        output_file = "training/validation_report.json"
    
    # Save detailed results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary
    summary_file = output_file.replace('.json', '_summary.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("FINANCIAL REASONING DATA VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"- Total Examples: {validation_results['total_examples']}\n")
        f.write(f"- Data Sources: {list(validation_results['source_distribution'].keys())}\n\n")
        
        f.write("Field Completeness:\n")
        for field, stats in validation_results['field_completeness'].items():
            f.write(f"- {field}: {stats['percentage']:.1f}% complete\n")
        f.write("\n")
        
        f.write("Mathematical Operations Coverage:\n")
        for op, stats in validation_results['mathematical_operations'].items():
            f.write(f"- {op.title()}: {stats['count']} examples ({stats['percentage']:.1f}%)\n")
        f.write("\n")
        
        if validation_results['quality_issues']:
            f.write("Quality Issues Identified:\n")
            for issue in validation_results['quality_issues']:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        if validation_results['recommendations']:
            f.write("Recommendations:\n")
            for rec in validation_results['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
    
    logger.info(f"📊 Validation report saved to: {output_file}")
    logger.info(f"📝 Summary saved to: {summary_file}")

def main():
    """Main validation function"""
    logger.info("=== PHASE 2: DATA VALIDATION ===")
    
    # Check if training data exists
    training_file = "training/finqa_clean.json"
    
    if not os.path.exists(training_file):
        logger.error(f"Training data not found: {training_file}")
        logger.info("Please run data_preprocessor.py first")
        return False
    
    # Run validation
    results = validate_training_data(training_file)
    
    if "error" in results:
        logger.error(f"Validation failed: {results['error']}")
        return False
    
    # Create report
    create_validation_report(results)
    
    # Print summary
    logger.info("\n=== VALIDATION SUMMARY ===")
    logger.info(f"Total examples: {results['total_examples']}")
    
    # Key quality metrics
    reasoning = results['reasoning_analysis']['quality_indicators']
    logger.info(f"Examples with calculations: {reasoning['has_calculations']['percentage']:.1f}%")
    logger.info(f"Examples with final answers: {reasoning['has_final_answer']['percentage']:.1f}%")
    logger.info(f"Multi-step reasoning: {reasoning['multi_step_reasoning']['percentage']:.1f}%")
    
    # Mathematical coverage
    math_ops = results['mathematical_operations']
    high_coverage = [op for op, stats in math_ops.items() if stats['percentage'] > 30]
    logger.info(f"Well-covered operations: {', '.join(high_coverage)}")
    
    # Issues and recommendations
    if results['quality_issues']:
        logger.warning(f"Quality issues found: {len(results['quality_issues'])}")
        for issue in results['quality_issues'][:3]:  # Show first 3
            logger.warning(f"  - {issue}")
    
    if results['recommendations']:
        logger.info(f"Recommendations: {len(results['recommendations'])}")
        for rec in results['recommendations'][:3]:  # Show first 3
            logger.info(f"  - {rec}")
    
    # Overall assessment
    quality_score = calculate_quality_score(results)
    logger.info(f"\nOverall Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 80:
        logger.info("✅ Dataset quality is excellent! Ready for training.")
    elif quality_score >= 60:
        logger.info("✅ Dataset quality is good. Consider addressing recommendations.")
    else:
        logger.warning("⚠️ Dataset quality needs improvement. Address issues before training.")
    
    return quality_score >= 60

def calculate_quality_score(results: Dict) -> float:
    """Calculate an overall quality score (0-100)"""
    score = 100.0
    
    # Penalize missing fields
    for field in ['instruction', 'input', 'output']:
        missing_pct = 100 - results['field_completeness'][field]['percentage']
        score -= missing_pct * 0.5  # 0.5 points per % missing
    
    # Penalize quality issues
    quality = results['content_quality']
    score -= quality['malformed_examples']['percentage'] * 2
    score -= max(0, quality['very_short_outputs']['percentage'] - 5) * 0.5
    
    # Reward good reasoning
    reasoning = results['reasoning_analysis']['quality_indicators']
    if reasoning['has_calculations']['percentage'] < 70:
        score -= (70 - reasoning['has_calculations']['percentage']) * 0.3
    
    if reasoning['has_final_answer']['percentage'] < 80:
        score -= (80 - reasoning['has_final_answer']['percentage']) * 0.2
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, score))

if __name__ == "__main__":
    main()