from src.data_prep.normalizer import Normalizer
from dotenv import load_dotenv
import os
import argparse



load_dotenv("config/.env")
TRAIN_RAW_DIR = os.getenv("TRAIN_RAW_DIR")
EVAL_RAW_DIR = os.getenv("EVAL_RAW_DIR")
TRAIN_TOKENS = os.getenv("TRAIN_TOKENS")
EVAL_TOKENS = os.getenv("EVAL_TOKENS")
MODEL = os.getenv("MODEL")
VOCAB = os.getenv("VOCAB")
UNK_THRESHOLD = int(os.getenv("UNK_THRESHOLD"))
TOP_K = int(os.getenv("TOP_K"))
NGRAM_ORDER = int(os.getenv("NGRAM_ORDER"))


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=str, help="Step to run")
args = parser.parse_args()

VALID_STEPS = ["dataprep", "model", "inference", "all"]

if args.step not in VALID_STEPS:
    print(f"Invalid step: '{args.step}'")
    print(f"Valid steps are: {', '.join(VALID_STEPS)}")
    exit(1)
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
        print("Model training not implemented yet.")
    if args.step == "inference" or args.step == "all":
        print("Inference not implemented yet.")
