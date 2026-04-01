import os
import argparse
import sys
import logging


from dotenv import load_dotenv
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.ui.app import PredictorUI
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)


# Load .env file
CONFIG_PATH = "config/.env"
if not os.path.exists(CONFIG_PATH):
    print(f"Folder not found: {CONFIG_PATH}. Check the path")
    sys.exit(1)

load_dotenv(CONFIG_PATH)

# Required variables
required_vars = [
    "TRAIN_RAW_DIR", "EVAL_RAW_DIR", "TRAIN_TOKENS", "EVAL_TOKENS",
    "MODEL", "VOCAB", "UNK_THRESHOLD", "TOP_K", "NGRAM_ORDER", "SMOOTHING"
]

# Check if any variable is missing
try:
    for var in required_vars:
        if os.getenv(var) is None:
            raise KeyError(var)
except KeyError as e:
    print(f"Missing variable in config/.env: {e}")
    sys.exit(1)

# Load variables
try:
    TRAIN_RAW_DIR = os.getenv("TRAIN_RAW_DIR")
    EVAL_RAW_DIR = os.getenv("EVAL_RAW_DIR")
    TRAIN_TOKENS = os.getenv("TRAIN_TOKENS")
    EVAL_TOKENS = os.getenv("EVAL_TOKENS")
    #Changed MODEL to NGRAM_MODEL to prevent conficlt with env variable called MODEL in My Machine
    NGRAM_MODEL = os.getenv("NGRAM_MODEL") # MODEL variable name may work with other machines
    VOCAB = os.getenv("VOCAB")
    UNK_THRESHOLD = int(os.getenv("UNK_THRESHOLD"))
    TOP_K = int(os.getenv("TOP_K"))
    NGRAM_ORDER = int(os.getenv("NGRAM_ORDER"))
    SMOOTHING = int(os.getenv("SMOOTHING"))

except ValueError as e:
    print(f"Error: UNK_THRESHOLD, TOP_K, and NGRAM_ORDER must be integers - {e}")
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=str, default="gui", help="Step to run")
args = parser.parse_args()

VALID_STEPS = ["dataprep", "model", "inference", "all", "gui"]

normalize = None
model = None

if args.step not in VALID_STEPS:
    print(f"Invalid step: '{args.step}'")
    print(f"Valid steps are: {', '.join(VALID_STEPS)}")
    sys.exit(1)
else:
    if args.step == "dataprep" or args.step == "all":
        normalize = Normalizer()
        normalize.load(TRAIN_RAW_DIR)
        #print(normalize.text)
        sentences = normalize.sentence_tokenize()
        #print(sentences)
        for i in range(len(sentences)):
            sentences[i] = normalize.normalize(sentences[i])
        normalize.save(sentences, TRAIN_TOKENS)
    if args.step == "model" or args.step == "all":
        if normalize is None:
            normalize = Normalizer()
        model = NGramModel(normalize.word_tokenize, unk_threshold=UNK_THRESHOLD, ngram_order=NGRAM_ORDER, smoothing=SMOOTHING)
        model.build_vocab(TRAIN_TOKENS)
        model.build_counts_and_probabilities(TRAIN_TOKENS)
        model.save_model(NGRAM_MODEL)
        model.save_vocab(VOCAB)
    if args.step == "inference" or args.step == "all" or args.step == "gui":
        if normalize is None:
            normalize = Normalizer()
        if model is None:
            model = NGramModel(normalize.word_tokenize, unk_threshold=UNK_THRESHOLD, ngram_order=NGRAM_ORDER, smoothing=SMOOTHING)
        model.load(NGRAM_MODEL, VOCAB)
        predictor = Predictor(model, normalize)
        print("\nN-Gram Language Model — Next Word Predictor")
        print("Type 'quit' to exit\n")
        if get_script_run_ctx() is not None:
            # If we're running inside Streamlit, we want to launch the UI directly
            predictor = Predictor(model, normalize)
            ui = PredictorUI(predictor, TOP_K)
            ui.run()
        else:
            while True:
                try:
                    text = input("> ")
                    if text.strip().lower() == "quit":
                        print("Goodbye.")
                        break
                    # Skip prediction if input is empty or whitespace
                    if not text.strip():
                        continue
                    predictions = predictor.predict_next(text, TOP_K)
                    print(f"Predictions: {predictions}\n")

                except KeyboardInterrupt:
                    print("\nGoodbye.")
                    break