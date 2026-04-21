"""
Test script to verify GitHub FinQA download and check for program fields
"""

import json
import os

def test_downloaded_data():
    """Test if the downloaded FinQA data has the required program fields"""
    
    files_to_check = [
        'raw_data/finqa_train.json',
        'raw_data/finqa_validation.json', 
        'raw_data/finqa_test.json'
    ]
    
    print("Testing downloaded FinQA data structure...")
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            continue
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                print(f"❌ {filename} is empty")
                continue
            
            print(f"\n📁 {filename}")
            print(f"   Examples: {len(data)}")
            
            # Check first item structure
            sample = data[0]
            print(f"   Fields: {list(sample.keys())}")
            
            # Check for critical fields
            has_program = 'program' in sample
            has_program_re = 'program_re' in sample
            has_question = 'question' in sample
            has_answer = 'answer' in sample
            
            print(f"   ✅ Has question: {has_question}")
            print(f"   ✅ Has answer: {has_answer}")
            print(f"   ✅ Has program: {has_program}")
            print(f"   ✅ Has program_re: {has_program_re}")
            
            # Show sample program if available
            if has_program and sample['program']:
                program = sample['program']
                if isinstance(program, list):
                    print(f"   📋 Sample program steps: {len(program)} steps")
                    if len(program) > 0:
                        print(f"      First step: {program[0]}")
                else:
                    print(f"   📋 Sample program: {str(program)[:100]}...")
            
            if has_program_re and sample['program_re']:
                program_re = sample['program_re']
                print(f"   🔄 Sample program_re: {str(program_re)[:100]}...")
            
            # Critical assessment
            if has_program or has_program_re:
                print(f"   ✅ READY FOR FILTERING - Contains reasoning programs!")
            else:
                print(f"   ❌ MISSING PROGRAMS - Filtering will return 0 results")
        
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    print(f"\n{'='*60}")
    print("If you see 'READY FOR FILTERING' above, you can proceed with:")
    print("python data_filter.py")

if __name__ == "__main__":
    test_downloaded_data()