import os
import string
import re

class Normalizer:

    """
    Handles loading, cleaning, and tokenizing raw text data.

    Attributes:
        text (str): The raw or processed text content.
        tokens (str): The tokenized output.

    Methods:
        load(folder_path): Loads all .txt files from a folder.
        strip_gutenberg(): Removes Project Gutenberg header and footer.
        normalize(text): Runs all normalization steps in order.
        sentence_tokenize(): Splits text into a list of sentences.
        word_tokenize(sentence): Splits a sentence into a list of words.
        save(sentences, filepath): Saves sentences to a .txt file.

    Example:
        normalizer = Normalizer()
        normalizer.load("data/raw/train")
        normalizer.strip_gutenberg()
        sentences = normalizer.sentence_tokenize()
        normalized = [normalizer.normalize(s) for s in sentences]
        normalizer.save(normalized, "data/processed/train_tokens.txt")
    """

    def __init__(self, text = "", tokens = ""):
        self.text = text
        self.tokens = tokens

    def load(self, folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        self.text = self.text + file.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        return self.text
        

    def strip_gutenberg(self):
        # Remove everything before and including *** START OF ... ***
        self.text = re.sub(r'^[\s\S]+?\*{3} START OF THE PROJECT GUTENBERG EBOOK[\s\S]+?\*{3}', '', self.text, count=1)

        # Remove everything from and including *** END OF ... ***
        self.text = re.sub(r'\*{3} END OF THE PROJECT GUTENBERG EBOOK[\s\S]+?\*{3}[\s\S]*$', '', self.text, count=1)

        self.text = self.text.strip()
        return self.text
    
    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_numbers(self, text):
        return text.translate(str.maketrans("", "", string.digits))

    def remove_whitespace(self, text):
        return " ".join(text.split())

    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text
        

    def sentence_tokenize(self):
        sentences = re.split(r'(?<=[.!?]) +', self.text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences
    

    def word_tokenize(self, sentence):
        return sentence.split()
    
    def save(self, sentences, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))