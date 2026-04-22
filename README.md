# Financial Statement Analysis AI - Phase 2: Data Engineering

This project builds a custom AI model for financial statement analysis with step-by-step reasoning capabilities. This README covers **Phase 2: Data Engineering** - the critical foundation that creates the "financial knowledge" for your AI model.

## 🎯 Phase 2 Overview

Phase 2 transforms raw financial datasets into high-quality training data that teaches your AI model mathematical reasoning for financial analysis. 

**What it does:**
- Downloads FinQA dataset (financial question-answering with step-by-step reasoning)
- Filters for examples containing mathematical reasoning steps
- Converts to instruction/input/response format for model training
- Validates data quality and provides detailed analysis

**Output:** Ready-to-use training dataset in `training/finqa_clean.json`

## 📁 Project Structure

```
Financial_Analysis_AI/
├── Build Pipeline.ipynb     # Phase 2 pipeline (download → filter → preprocess → validate)
├── training/                # Training data and reports
│   ├── finqa_clean.json    # Main training dataset
│   ├── finqa_clean_summary.json
│   ├── validation_report.json
│   └── phase2_summary.txt
├── raw_data/               # Downloaded datasets
│   ├── finqa_train.json
│   ├── finqa_validation.json
│   └── finqa_test.json
├── filtered_data/          # Filtered reasoning examples
│   ├── finqa_train_filtered.json
│   └── finqa_validation_filtered.json
├── core/                   # Future: Analysis engine
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

Phase 2 runs entirely from **`Build Pipeline.ipynb`**.

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt` (Jupyter is listed there for running the notebook).
2. Open `Build Pipeline.ipynb` in VS Code, Cursor, or Jupyter.
3. Select the correct Python kernel (your venv if you use one).
4. Use **Run All** to execute the full pipeline, or run each code section in order: downloader → filter → preprocessor → validator.

## 📊 Expected Results

After successful completion, you should have:

- **5,000-8,000 training examples** from FinQA dataset
- **High-quality mathematical reasoning** examples
- **Comprehensive validation report** showing data quality metrics  
- **Ready for Phase 3** model training

### Sample Training Example
```json
{
  "instruction": "You are a financial analyst AI. Given the financial data and question below, provide step-by-step reasoning to calculate the answer.",
  "input": "Question: What is the return on equity for 2023?\n\nFinancial Data Table:\n| Item | 2023 | 2022 |\n| Net Income | 125,000 | 110,000 |\n| Shareholders Equity | 500,000 | 450,000 |",
  "output": "Step 1: Extract Net Income for 2023\nStep 2: Extract Shareholders Equity for 2023\nStep 3: Calculate ROE = Net Income / Shareholders Equity\nStep 4: ROE = 125,000 / 500,000 = 0.25\n\nFinal Answer: 25%"
}
```

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`:

```txt
datasets>=2.14.0          # Hugging Face datasets
transformers>=4.35.0      # Model handling
torch>=2.0.0             # PyTorch
pandas>=1.5.0            # Data manipulation
numpy>=1.24.0            # Numerical computing
tqdm>=4.65.0             # Progress bars
```

Install with:
```bash
pip install -r requirements.txt
```

## 🎛️ Configuration Options

### Custom dataset limits
In **Build Pipeline.ipynb**, in the **Data preprocessor** code cell, adjust `preprocess_for_training` / `main()` (e.g. set `max_examples`, `instruction_variety`).

### Quality thresholds
In the **Data filter** code cell, edit `has_valid_program` and related logic (e.g. math patterns, max program length) to match your data.

## 📈 Quality Metrics

The validation report provides comprehensive quality analysis:

### Key Metrics
- **Field Completeness**: % of examples with required fields
- **Mathematical Operations**: Coverage of different calculation types
- **Reasoning Quality**: % with step-by-step explanations
- **Length Statistics**: Input/output length distributions
- **Source Distribution**: Balance across datasets

### Quality Score
- **80-100**: Excellent quality, ready for training
- **60-79**: Good quality, consider improvements
- **<60**: Needs improvement before training

## 🔍 Troubleshooting

### Common Issues

**1. "No datasets found"**
- Check internet connection
- Ensure Hugging Face datasets library is installed
- FinQA should be publicly accessible

**2. "No reasoning examples found"**
- Datasets may have different field names
- Check the **Data filter** section in `Build Pipeline.ipynb` for your dataset format

**3. "Low quality score"**
- Review validation report details
- Consider adjusting filtering criteria
- May need additional data sources

**4. "Out of memory errors"**
- Reduce `max_examples` in preprocessor
- Process datasets individually
- Use smaller batch sizes

### Debug Mode
Add logging for detailed debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Validation Checklist

Before proceeding to Phase 3, ensure:

- [ ] Training data exists: `training/finqa_clean.json`
- [ ] Dataset size > 1,000 examples (preferably 5,000+)
- [ ] Quality score ≥ 60
- [ ] Mathematical operations well-covered
- [ ] Validation report generated
- [ ] No critical quality issues

## 🔜 Next Steps: Phase 3

Once Phase 2 is complete:

1. **Review** validation report thoroughly
2. **Set up** GPU environment for training
3. **Create** `train.py` script for model fine-tuning
4. **Configure** QLoRA for memory-efficient training
5. **Train** on your financial reasoning dataset

## 🤝 Contributing

This is your graduation project! Key areas for enhancement:

- **Additional datasets**: Financial domain-specific data
- **Better filtering**: More sophisticated reasoning detection
- **Quality improvements**: Enhanced validation metrics
- **Domain expansion**: Support for different financial document types

## 📄 License & Usage

This is an educational project for financial statement analysis. The FinQA dataset has its own license - please review their terms for commercial use.

---

**Next Phase:** [Phase 3: Model Training](../phase3/README.md) (Coming soon!)

**Questions?** Check the validation report first, then review the troubleshooting section above.