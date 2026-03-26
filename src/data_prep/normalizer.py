import os
import string
import re

class Normalizer:

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
        sentences_to_remove = ["", ""]
        self.text = self.text.replace(sentences_to_remove[0], "")
        self.text = self.text.replace(sentences_to_remove[1], "")
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
        print(sentence.split())
    
    def save(self, sentences, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
        print(f"Sentences saved to {filepath}")