# NLP Authorship Attribution

A project for authorship attribution using NLP models (Gemini and RoBERTa).

## Project Overview

This project implements a RoBERTa-based and LLM-based approach to authorship attribution using the Blog Authorship Corpus. Note: I include the training, validation, and testing datasets used for experimentaiton, but not the full original corpus. 

***Citation for Dataset:***

J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs. URL: http://www.cs.biu.ac.il/~schlerj/schler_springsymp06.pdf


## Project Structure

```
├── data/
│   ├── blog_train.csv                            # Training dataset
│   ├── blog_val.csv                              # Validation dataset
│   ├── blog_test.csv                             # Test dataset
│   ├── candidate_sets.json                       # Candidate author sets for evaluation
│   └── llm_results_*.csv                         # LLM prediction results
├── prompts/
│   ├── lip_prompt.txt                            # Linguistically-informed prompt template
├── src/
    |── authorship_attribution_roberta.ipynb      # Notebook for fine-tuning and evaluating RoBERTa
│   ├── prepare_datasets.py                       # Dataset preprocessing script
│   ├── sample_candidate_texts.py                 # Generate candidate author samples
│   ├── run_authorship_attribution_llm.py         # Main LLM inference script
│   └── eval_llm_preds.py                         # Evaluation metrics computation
```

## Setup

### Prerequisites

- Python 3.8+
- Google Gemini API key

###

1. Clone the repository:

   ```bash
   git clone https://github.com/cohajunwa/NLP-Authorship-Attribution.git
   cd NLP-Authorship-Attribution
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API key (See `.env.example`):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
