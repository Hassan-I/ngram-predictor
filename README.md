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
### Command-Line Interface (CLI)
The main.py script uses the --step argument to execute specific parts of the logic.

**Step A:** Data Preparation Clean and tokenize the raw dataset to prepare it for the modeling phase.  
The command for this step is <code style="background-color: #ffffff3a; color: #f5a904;">python main.py --step dataprep</code>

**Step B:** Build the N-Gram Model Generate the N-Gram counts and probability distributions (from 1-gram up to the limit specified in your .env).  The command for this step is <code style="background-color: #ffffff3a; color: #f5a904;">python main.py --step model</code>

**Step C:** Run Inference Test the prediction algorithm directly in the terminal by providing manual text input.  The command for this step is <code style="background-color: #ffffff3a; color: #f5a904;">python main.py --step inference</code>

### Graphical User Interface (GUI)
To interact with the predictor through a visual web interface, use the following Streamlit command: <code style="background-color: #ffffff3a; color: #f5a904;">streamlit run main.py</code>

## Project Structure
```text
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
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
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   ├── inference/
│   │   └── predictor.py
│   ├── ui/
│   │   └── app.py
│   └── evaluation/
│       └── evaluator.py
├── main.py
├── tests/
│   ├── test_data_prep.py
│   ├── test_model.py
│   ├── test_inference.py
│   ├── test_ui.py
│   └── test_evaluation.py
├── .gitignore
├── requirements.txt
└── README.md