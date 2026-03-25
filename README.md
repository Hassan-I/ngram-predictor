# N-Gram Language Model
A Python-based N-Gram language model that trains on text data to predict the next word in a sequence. 

## Requirements
* Python 3.14+
* Install dependencies listed in 'requirements.txt'
  
## Setup
* Clone the repo
* Create and activate Anaconda environment
* Install dependencies
* Populate <code style="background-color: #ffffff3a; color: #f5a904;">config/.env</code>.This is the list of required variables:
    * TRAIN_RAW_DIR
    * EVAL_RAW_DIR
    * TRAIN_TOKENS
    * EVAL_TOKENS
    * MODEL
    * VOCAB
    * UNK_THRESHOLD
    * TOP_K
    * NGRAM_ORDER
    * LOG_LEVEL
* Download raw .txt files into <code style="background-color: #ffffff3a; color: #f5a904;">data/raw/train/</code> and <code style="background-color: #ffffff3a; color: #f5a904;">data/raw/eval/</code> folders
  
## Usage

## Project Structure
├── config/ <br>
│   └── .env                     
├── data/<br>
│   ├── raw/<br>
│   │   ├── train/                  
│   │   └── eval/
│   ├── processed/
│   │   ├── train_tokens.txt
│   │   └── eval_tokens.txt
│   └── model/
│       ├── model.json
│       └── vocab.json
├── src/
│   ├── data_prep/
│   │   └── normalizer.py           # Normalizer class
│   ├── model/
│   │   └── ngram_model.py          # NGramModel class
│   ├── inference/
│   │   └── predictor.py            # Predictor class
│   ├── ui/
│   │   └── app.py                  # PredictorUI class — extra credit
│   └── evaluation/
│       └── evaluator.py            # Evaluator class — extra credit
├── main.py                         # Single entry point — CLI and wiring
├── tests/
│   ├── test_data_prep.py           # Extra credit
│   ├── test_model.py               # Extra credit
│   ├── test_inference.py           # Extra credit
│   ├── test_ui.py                  # Extra credit
│   └── test_evaluation.py          # Extra credit
├── .gitignore
├── requirements.txt
└── README.md