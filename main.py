from src.data_prep.normalizer import Normalizer


TRAIN_RAW_DIR = r"C:\Users\hhassan4\Documents" + "\\"

normalize = Normalizer()
normalize.load(TRAIN_RAW_DIR)
normalize.strip_gutenberg()
sentences = normalize.sentence_tokenize()
for i in range(len(sentences)):
    sentences[i] = normalize.normalize(sentences[i])
    print(sentences[i])

normalize.word_tokenize(sentences[0])
normalize.save(sentences, r"C:\Users\hhassan4\Documents" + "\\sentences.txt")