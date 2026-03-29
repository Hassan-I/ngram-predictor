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

if args.step == "dataprep":
    normalize = Normalizer()
    normalize.load(TRAIN_RAW_DIR)
    normalize.strip_gutenberg()
    sentences = normalize.sentence_tokenize()
    for i in range(len(sentences)):
        sentences[i] = normalize.normalize(sentences[i])
        print(sentences[i])

    normalize.word_tokenize(sentences[0])
    normalize.save(sentences, TRAIN_TOKENS)