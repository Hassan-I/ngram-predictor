import os
import argparse
import sys

from dotenv import load_dotenv
from src.data_prep import normalizer
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer


# Load .env file
if not os.path.exists("config/.env"):
    print("Error: config/.env file not found")
    sys.exit(1)

load_dotenv("config/.env")

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
parser.add_argument("--step", type=str, help="Step to run")
args = parser.parse_args()

VALID_STEPS = ["dataprep", "model", "inference", "all"]

normalize = None # Initialize normalize variable to None for later use in model step

if args.step not in VALID_STEPS:
    print(f"Invalid step: '{args.step}'")
    print(f"Valid steps are: {', '.join(VALID_STEPS)}")
    sys.exit(1)
else:
    if args.step == "dataprep" or args.step == "all":
        normalize = Normalizer()
        normalize.load(TRAIN_RAW_DIR)
        normalize.strip_gutenberg()
        sentences = normalize.sentence_tokenize()
        for i in range(len(sentences)):
            sentences[i] = normalize.normalize(sentences[i])
            print(sentences[i])

        normalize.word_tokenize(sentences[0])
        normalize.save(sentences, TRAIN_TOKENS)
    if args.step == "model" or args.step == "all":
        if normalize is None:
            normalize = Normalizer()
        model = NGramModel(normalize.word_tokenize, unk_threshold=UNK_THRESHOLD, ngram_order=NGRAM_ORDER, smoothing=SMOOTHING)
        model.build_vocab(TRAIN_TOKENS)
        model.build_counts_and_probabilities(TRAIN_TOKENS)
        model.save_model(NGRAM_MODEL)
        model.save_vocab(VOCAB)
    if args.step == "inference" or args.step == "all":
        print("Inference not implemented yet.")