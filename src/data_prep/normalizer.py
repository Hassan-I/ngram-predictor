import unicodedata
import os
import string
import re
import sys

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
        """
        Initializes the Normalizer with optional text and tokens.

        Args:
            text (str): Initial raw or processed text content. Defaults to "".
            tokens (str): Initial tokenized output. Defaults to "".
        """
        self.text = text
        self.tokens = tokens

    def load(self, folder_path):
        """
        Reads all text files from a directory, cleans them, and appends them to the instance text.

        Args:
            folder_path (str): The path to the directory containing .txt files.

        Returns:
            str: The accumulated text content from all successfully read and cleaned files.
        """
        try:
            # Move the directory listing INSIDE the try block
            files = os.listdir(folder_path)
            
            for file_name in files:
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            raw_file = file.read()
                            self.text = self.text + self.strip_gutenberg(raw_file) + "\n"
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        except FileNotFoundError:
            # Now this will correctly catch the error when the folder is missing
            print(f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env.")
            sys.exit(1)

        return self.text
        

    def strip_gutenberg(self, input_text=""):
        """
        Removes the Project Gutenberg standard header and footer from the input text.

        Args:
            input_text (str): The raw text content of a Project Gutenberg ebook.

        Returns:
            str: The cleaned text containing only the body of the book.
        """
        # Remove everything before and including *** START OF ... ***
        input_text = re.sub(r'^[\s\S]+?\*{3} START OF THE PROJECT GUTENBERG EBOOK[\s\S]+?\*{3}', '', input_text, count=1)

        # Remove everything from and including *** END OF ... ***
        input_text = re.sub(r'\*{3} END OF THE PROJECT GUTENBERG EBOOK[\s\S]+?\*{3}[\s\S]*$', '', input_text, count=1)

        return input_text.strip()
    
    def lowercase(self, text):
        """
        Converts the provided text string to lowercase.

        Args:
            text (str): The string to be converted.

        Returns:
            str: The input text in all lowercase letters.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """
        Removes punctuation from the text while preserving apostrophes and normalizing characters.

        Args:
            text (str): The string from which punctuation and non-ASCII characters should be removed.

        Returns:
            str: The cleaned text containing only alphanumeric characters, spaces, and apostrophes.
        """
        custom_punctuation = string.punctuation.replace("'", "")
        pattern = f"[{re.escape(custom_punctuation)}]"
        text = re.sub(pattern, " ", text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        return text

    def remove_numbers(self, text):
        """
        Removes all digits from the text.

        Args:
            text (str): The string from which digits should be removed.

        Returns:
            str: The text with all digits removed.
        """
        return text.translate(str.maketrans("", "", string.digits))

    def remove_whitespace(self, text):
        """
        Removes leading and trailing whitespace from the text.

        Args:
            text (str): The string from which whitespace should be removed.

        Returns:
            str: The text with leading and trailing whitespace removed.
        """
        return " ".join(text.split())

    def normalize(self, text):
        """
        Executes a sequence of cleaning steps to standardize the text for modeling.

        Args:
            text (str): The raw sentence or string to be normalized.

        Returns:
            str: The fully processed and cleaned text.
        """
        text = re.sub(r"(\w+)('s|'m|'re|'ve|'d|'ll|'t)(?=[^a-zA-Z]|$)", r"\1 \2", text)
        text = re.sub(r"(\w+s)(')(?=[^a-zA-Z]|$)", r"\1 \2", text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        text = re.sub(r"(^|\s)'(?!(s|m|re|ve|d|ll|t)\b)(\w+)", r"\1\3", text)
        text = re.sub(r"\s'(?!(s|m|re|ve|d|ll|t)\b)", r"", text)
        text = text.strip(" '")
        return text
        

    def sentence_tokenize(self):
        """
        Splits the accumulated text into a list of individual sentences.

        Returns:
            list: A list of strings, where each string is a single sentence stripped of extra whitespace.
        """
        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        return [s.strip() for s in sentences if s.strip()]


    def word_tokenize(self, sentence):
        """
        Splits a single sentence into a list of individual words (tokens).

        Args:
            sentence (str): The sentence string to be tokenized.

        Returns:
            list: A list of words derived from the sentence.
        """
        return sentence.split()
    
    def save(self, sentences, filepath):
        """
        Saves a list of processed sentences to a text file, with each sentence on a new line.

        Args:
            sentences (list): A list of strings representing the normalized sentences.
            filepath (str): The destination path where the file should be saved.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))