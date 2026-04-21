"""
Phase 2: Complete Pipeline Runner
Orchestrates the complete data engineering pipeline for financial reasoning model training.
"""

import subprocess
import sys
import os
import time
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check and install required dependencies"""
    logger.info("Checking dependencies...")
    
    try:
        # Check if requests library is available
        import requests
        logger.info("✅ requests library found")
    except ImportError:
        logger.info("Installing requests library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests>=2.25.0"], check=True)
        logger.info("✅ requests library installed")
    
    try:
        # Check if transformers is available
        import transformers
        logger.info("✅ transformers library found")
    except ImportError:
        logger.info("Installing transformers library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.35.0"], check=True)
        logger.info("✅ transformers library installed")
    
    try:
        # Check if tqdm is available
        import tqdm
        logger.info("✅ tqdm library found")
    except ImportError:
        logger.info("Installing tqdm library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tqdm>=4.65.0"], check=True)
        logger.info("✅ tqdm library installed")

def run_step(step_name: str, script_name: str, required: bool = True) -> bool:
    """Run a pipeline step and handle errors"""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING STEP: {step_name}")
    logger.info(f"{'='*60}")
    
    if not os.path.exists(script_name):
        logger.error(f"Script not found: {script_name}")
        return False
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=1800)  # 30 minute timeout
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {step_name} completed successfully in {elapsed_time:.1f}s")
            
            # Show last few lines of output
            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                logger.info("Output preview:")
                for line in output_lines[-5:]:  # Last 5 lines
                    if line.strip():
                        logger.info(f"  {line}")
            
            return True
        else:
            logger.error(f"❌ {step_name} failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            if result.stdout:
                logger.error(f"Standard output: {result.stdout}")
            
            if required:
                return False
            else:
                logger.warning(f"⚠️ {step_name} failed but is not required, continuing...")
                return True
                
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {step_name} timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} failed with exception: {str(e)}")
        return False

def check_results():
    """Check the results of the complete pipeline"""
    logger.info("\n" + "="*60)
    logger.info("CHECKING PIPELINE RESULTS")
    logger.info("="*60)
    
    results = {
        "raw_data": {"exists": False, "files": 0, "examples": 0},
        "filtered_data": {"exists": False, "files": 0, "examples": 0},
        "training_data": {"exists": False, "examples": 0},
        "validation_report": {"exists": False}
    }
    
    # Check raw data
    if os.path.exists("raw_data"):
        results["raw_data"]["exists"] = True
        json_files = [f for f in os.listdir("raw_data") if f.endswith('.json')]
        results["raw_data"]["files"] = len(json_files)
        
        # Count total raw examples
        for filename in json_files:
            try:
                import json
                with open(os.path.join("raw_data", filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results["raw_data"]["examples"] += len(data)
            except:
                pass
    
    # Check filtered data
    if os.path.exists("filtered_data"):
        results["filtered_data"]["exists"] = True
        filtered_files = [f for f in os.listdir("filtered_data") if f.endswith('_filtered.json')]
        results["filtered_data"]["files"] = len(filtered_files)
        
        # Count total filtered examples
        for filename in filtered_files:
            try:
                import json
                with open(os.path.join("filtered_data", filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results["filtered_data"]["examples"] += len(data)
            except:
                pass
    
    # Check training data
    training_file = "training/finqa_clean.json"
    if os.path.exists(training_file):
        results["training_data"]["exists"] = True
        try:
            import json
            with open(training_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results["training_data"]["examples"] = len(data)
        except:
            pass
    
    # Check validation report
    if os.path.exists("training/validation_report.json"):
        results["validation_report"]["exists"] = True
    
    # Print results summary
    logger.info("\n📊 PIPELINE RESULTS SUMMARY:")
    logger.info(f"Raw Data: {results['raw_data']['files']} files, {results['raw_data']['examples']} total examples")
    logger.info(f"Filtered Data: {results['filtered_data']['files']} files, {results['filtered_data']['examples']} reasoning examples")
    logger.info(f"Training Data: {results['training_data']['examples']} formatted examples")
    logger.info(f"Validation Report: {'✅ Generated' if results['validation_report']['exists'] else '❌ Missing'}")
    
    # Calculate success metrics
    success_metrics = []
    if results['raw_data']['examples'] > 0:
        success_metrics.append("✅ Successfully downloaded datasets")
    if results['filtered_data']['examples'] > 0:
        filter_rate = (results['filtered_data']['examples'] / results['raw_data']['examples']) * 100 if results['raw_data']['examples'] > 0 else 0
        success_metrics.append(f"✅ Successfully filtered data ({filter_rate:.1f}% retention rate)")
    if results['training_data']['examples'] > 0:
        success_metrics.append(f"✅ Successfully created training data")
    if results['validation_report']['exists']:
        success_metrics.append("✅ Successfully validated data quality")
    
    for metric in success_metrics:
        logger.info(metric)
    
    return results

def generate_phase2_summary(results: dict):
    """Generate a summary of Phase 2 completion"""
    summary_file = "training/phase2_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PHASE 2: DATA ENGINEERING COMPLETION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Completion Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Pipeline Results:\n")
        f.write(f"- Raw Dataset Examples: {results['raw_data']['examples']}\n")
        f.write(f"- Filtered Reasoning Examples: {results['filtered_data']['examples']}\n")
        f.write(f"- Training Examples Created: {results['training_data']['examples']}\n")
        f.write(f"- Validation Report: {'Generated' if results['validation_report']['exists'] else 'Missing'}\n\n")
        
        f.write("Key Files Created:\n")
        f.write("- training/finqa_clean.json (Main training data)\n")
        f.write("- training/finqa_clean_summary.json (Dataset statistics)\n")
        f.write("- training/validation_report.json (Quality analysis)\n")
        f.write("- training/validation_report_summary.txt (Human-readable report)\n\n")
        
        f.write("Next Steps:\n")
        f.write("1. Review validation report for data quality\n")
        f.write("2. Proceed to Phase 3: Model Training (train.py)\n")
        f.write("3. Set up training environment with GPU support\n")
        f.write("4. Fine-tune base model with financial reasoning data\n\n")
        
        if results['training_data']['examples'] >= 5000:
            f.write("✅ Dataset size is excellent for training!\n")
        elif results['training_data']['examples'] >= 1000:
            f.write("✅ Dataset size is adequate for training.\n")
        else:
            f.write("⚠️  Dataset size is small - consider gathering more data.\n")
    
    logger.info(f"📋 Phase 2 summary saved to: {summary_file}")

def main():
    """Run the complete Phase 2 data engineering pipeline"""
    start_time = time.time()
    
    logger.info("🚀 STARTING PHASE 2: DATA ENGINEERING PIPELINE")
    logger.info("Building financial reasoning knowledge base from FinQA dataset...")
    logger.info("FinQA provides comprehensive financial Q&A with mathematical reasoning")
    
    # Step 0: Check dependencies
    try:
        check_dependencies()
    except Exception as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        logger.error("Please install dependencies manually: pip install -r requirements.txt")
        return False
    
    # Pipeline steps
    pipeline_steps = [
        ("Dataset Download", "datasets_downloader.py", True),
        ("Data Filtering", "data_filter.py", True),
        ("Data Preprocessing", "data_preprocessor.py", True),
        ("Data Validation", "data_validator.py", False),  # Not required for pipeline to continue
    ]
    
    # Execute pipeline
    success = True
    for step_name, script_name, required in pipeline_steps:
        if not run_step(step_name, script_name, required):
            if required:
                success = False
                break
    
    # Check results regardless of validation step success
    results = check_results()
    
    # Generate summary
    generate_phase2_summary(results)
    
    # Final assessment
    total_time = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    if success and results['training_data']['examples'] > 0:
        logger.info("🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total pipeline time: {total_time:.1f} seconds")
        logger.info(f"📚 Created {results['training_data']['examples']} training examples")
        logger.info("\n🔜 NEXT STEPS:")
        logger.info("1. Review the validation report in training/validation_report_summary.txt")
        logger.info("2. Set up GPU environment for training")
        logger.info("3. Proceed to Phase 3: Model Training")
        logger.info("   - Create train.py script")
        logger.info("   - Configure QLoRA fine-tuning")
        logger.info("   - Train on financial reasoning data")
        
        return True
    else:
        logger.error("❌ PHASE 2 FAILED")
        logger.error("Please check error messages above and retry failed steps")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)