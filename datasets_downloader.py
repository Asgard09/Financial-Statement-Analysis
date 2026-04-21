"""
Phase 2: Dataset Downloader
Downloads FinQA dataset from original GitHub source for financial reasoning training.
Uses the "Source of Truth" data from czyssrs/FinQA with complete program/program_re fields.
FinQA provides comprehensive financial question-answering with step-by-step mathematical reasoning.
"""

import json
import os
import requests
from tqdm import tqdm
import logging
from zipfile import ZipFile
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_finqa():
    """Download FinQA dataset from original GitHub source"""
    logger.info("Starting FinQA dataset download from original GitHub source...")
    logger.info("Source: https://github.com/czyssrs/FinQA")
    
    # GitHub repository ZIP URL
    github_zip_url = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs("raw_data", exist_ok=True)
        
        logger.info("Downloading FinQA repository...")
        
        # Download the ZIP file
        response = requests.get(github_zip_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            # Download with progress bar
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        pbar.update(len(chunk))
            
            tmp_zip_path = tmp_file.name
        
        logger.info("Extracting dataset files...")
        
        # Extract and process JSON files
        total_examples = 0
        
        with ZipFile(tmp_zip_path, 'r') as zip_ref:
            # File mappings: original -> our naming convention
            file_mappings = {
                'FinQA-main/dataset/train.json': 'raw_data/finqa_train.json',
                'FinQA-main/dataset/dev.json': 'raw_data/finqa_validation.json', 
                'FinQA-main/dataset/test.json': 'raw_data/finqa_test.json'
            }
            
            for original_path, output_file in file_mappings.items():
                try:
                    logger.info(f"Processing {original_path}...")
                    
                    # Extract and read the JSON file
                    with zip_ref.open(original_path) as json_file:
                        data = json.load(json_file)
                    
                    # Validate structure
                    if not data:
                        logger.warning(f"Empty data in {original_path}")
                        continue
                    
                    # Check if data contains expected fields
                    sample_item = data[0] if isinstance(data, list) else data
                    if isinstance(sample_item, dict) and 'qa' in sample_item:
                        # Extract qa fields to top level for easier processing
                        processed_data = []
                        
                        for item in tqdm(data, desc=f"Processing {os.path.basename(output_file)}"):
                            if 'qa' in item:
                                qa_data = item['qa']
                                processed_item = {
                                    'id': item.get('id', ''),
                                    'pre_text': item.get('pre_text', []),
                                    'post_text': item.get('post_text', []), 
                                    'table': item.get('table', []),
                                    'question': qa_data.get('question', ''),
                                    'answer': qa_data.get('answer', ''),
                                    'program': qa_data.get('program', []),
                                    'program_re': qa_data.get('program_re', ''),
                                    'gold_inds': qa_data.get('gold_inds', {})
                                }
                                processed_data.append(processed_item)
                        
                        data = processed_data
                    
                    # Save processed data
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"✅ Saved {len(data)} examples to {output_file}")
                    total_examples += len(data)
                    
                    # Show sample fields for verification
                    if data and isinstance(data[0], dict):
                        sample_keys = list(data[0].keys())
                        logger.info(f"  Sample fields: {sample_keys}")
                        
                        # Check for program fields specifically
                        has_program = any(key in sample_keys for key in ['program', 'program_re'])
                        if has_program:
                            logger.info(f"  ✅ Contains reasoning programs!")
                        else:
                            logger.warning(f"  ⚠️ No program fields found")
                
                except KeyError as e:
                    logger.error(f"File {original_path} not found in archive: {e}")
                except Exception as e:
                    logger.error(f"Error processing {original_path}: {e}")
        
        # Clean up temporary file
        os.unlink(tmp_zip_path)
        
        if total_examples > 0:
            logger.info(f"✅ FinQA download complete! Total examples: {total_examples}")
            logger.info("Original FinQA data includes reasoning programs (program/program_re fields)")
            return True
        else:
            logger.error("❌ No data was successfully downloaded")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading FinQA from GitHub: {str(e)}")
        return False


def check_downloaded_data():
    """Check what data has been successfully downloaded"""
    logger.info("Checking downloaded datasets...")
    
    raw_data_dir = "raw_data"
    if not os.path.exists(raw_data_dir):
        logger.warning("No raw_data directory found")
        return
    
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.json')]
    
    if not files:
        logger.warning("No JSON files found in raw_data directory")
        return
    
    total_examples = 0
    
    for filename in sorted(files):
        filepath = os.path.join(raw_data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"{filename}: {len(data)} examples")
            total_examples += len(data)
            
            # Show sample keys from first item
            if data and isinstance(data[0], dict):
                sample_keys = list(data[0].keys())
                logger.info(f"  Sample fields: {sample_keys[:5]}{'...' if len(sample_keys) > 5 else ''}")
        
        except Exception as e:
            logger.error(f"Error reading {filename}: {str(e)}")
    
    logger.info(f"Total examples downloaded: {total_examples}")

def main():
    """Main function to download FinQA dataset"""
    logger.info("=== PHASE 2: DATASET DOWNLOADER ===")
    logger.info("Focusing on FinQA dataset - provides excellent financial reasoning examples")
    
    # Download FinQA
    finqa_success = download_finqa()
    
    # Check results
    check_downloaded_data()
    
    # Summary
    if finqa_success:
        logger.info("✅ FinQA dataset downloaded successfully!")
        logger.info("FinQA provides comprehensive financial reasoning training data")
    else:
        logger.error("❌ FinQA dataset download failed")
        return False
    
    logger.info("Next step: Run data_filter.py to extract reasoning examples")
    return True

if __name__ == "__main__":
    main()