import json
import sys


class NGramModel:

    """
    N-Gram language model supporting MLE and Laplace smoothing.

    Attributes:
        tokenizer: Tokenizer instance used for word tokenization.
        unk_threshold (int): Minimum word count to keep in vocabulary.
        ngram_order (int): Maximum n-gram order (e.g. 3 for trigrams).
        smoothing (int): Smoothing parameter — 0 for MLE, 1 for Laplace.
        vocab (dict): Vocabulary with word counts.
        counts (dict): N-gram counts indexed by order.
        probs (dict): N-gram probabilities indexed by order.

    Methods:
        build_vocab(token_file): Builds vocabulary and applies UNK threshold.
        build_counts_and_probabilities(token_file): Counts n-grams and computes probabilities.
        lookup(context): Backoff lookup returning {word: probability} dict.
        save_model(model_path): Saves probability tables to model.json.
        save_vocab(vocab_path): Saves vocabulary list to vocab.json.
        load(model_path, vocab_path): Loads model and vocabulary from JSON files.

    Example:
        model = NGramModel(normalizer, unk_threshold=2, ngram_order=3, smoothing=1)
        model.build_vocab(TRAIN_TOKENS)
        model.build_counts_and_probabilities(TRAIN_TOKENS)
        model.save_model(MODEL)
        model.save_vocab(VOCAB)
    """

    def __init__(self, tokenizer, unk_threshold = 2, ngram_order = 3, smoothing=1, vocab=None):
        """
        Initializes the NGramModel with configuration parameters and empty data structures.

        Args:
            tokenizer (object): The tokenizer instance used for processing text.
            unk_threshold (int): The minimum frequency a word must have to remain in the vocabulary. Defaults to 2.
            ngram_order (int): The highest order of n-grams to calculate (e.g., 3 for trigrams). Defaults to 3.
            smoothing (int): The smoothing technique to apply; 0 for MLE, 1 for Laplace. Defaults to 1.
            vocab (dict, optional): A pre-existing vocabulary dictionary. Defaults to None.
        """
        self.tokenizer     = tokenizer
        self.unk_threshold = unk_threshold
        self.ngram_order   = ngram_order
        self.smoothing     = smoothing  # 0 for MLE, 1 for Laplace
        self.vocab         = vocab if vocab is not None else {}
        self.counts        = {}
        self.probs         = {}


    def build_vocab(self, token_file):
        """
        Builds vocabulary from token file and applies UNK_THRESHOLD.
        Words appearing fewer than UNK_THRESHOLD times are replaced with <UNK>.

        Args:
            token_file (str): Path to the token file (one sentence per line).
        """
        # Read all sentences
        sentences = self._read_sentences(token_file)

        # Count all words
        word_counts = {}
        for words in sentences:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Build vocab — replace words below threshold with <UNK>
        for word, count in word_counts.items():
            if count < self.unk_threshold:
                self.vocab["<UNK>"] = self.vocab.get("<UNK>", 0) + count
            else:
                self.vocab[word] = count

        # Make sure <UNK> is in vocab even if no words were replaced
        if "<UNK>" not in self.vocab:
            self.vocab["<UNK>>"] = 0



    def build_counts_and_probabilities(self, token_file):
        """
        Counts all n-grams at orders 1 through NGRAM_ORDER and computes probabilities.
        Counts and probabilities are computed together to avoid hidden ordering bugs.

        Args:
            token_file (str): Path to the token file (one sentence per line).
        """
        # Initialize counts and probabilities for each order
        self.counts = {n: {} for n in range(1, self.ngram_order + 1)}
        self.probs  = {n: {} for n in range(1, self.ngram_order + 1)}

        vocab_size = len(self.vocab)

        # Read sentences and apply UNK
        sentences = self._read_sentences(token_file)
        sentences = self._apply_unk(sentences)

        # Count all n-grams
        for words in sentences:
            for n in range(1, self.ngram_order + 1):
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i:i + n])
                    self.counts[n][ngram] = self.counts[n].get(ngram, 0) + 1

        # Compute probabilities
        for n in range(1, self.ngram_order + 1):
            for ngram, count in self.counts[n].items():
                if n == 1:
                    total = sum(self.counts[1].values())
                    if self.smoothing == "laplace":
                        self.probs[n][ngram] = (count + 1) / (total + vocab_size)
                    else:
                        self.probs[n][ngram] = count / total
                else:
                    prefix       = ngram[:-1]
                    prefix_count = self.counts[n - 1].get(prefix, 0)
                    if self.smoothing:
                        self.probs[n][ngram] = (count + 1) / (prefix_count + vocab_size)
                    else:
                        #else 0 if prefix_count is 0 to avoid division by zero
                        self.probs[n][ngram] = count / prefix_count if prefix_count > 0 else 0

    def lookup(self, context):
        """
        Backoff lookup: tries the highest-order context first, falls back to lower orders.

        Args:
            context (list): List of words representing the context.

        Returns:
            dict: {word: probability} from the highest order that matches, or empty dict.
        """
        # Apply UNK to context words not in vocab
        context = [w if w in self.vocab else "<UNK>" for w in context]

        # Try from highest order down to 1-gram
        for n in range(self.ngram_order, 0, -1):

            # Take the last n-1 words as the prefix
            prefix = tuple(context[-(n - 1):]) if n > 1 else ()

            # Find all ngrams that start with this prefix
            result = {}
            for ngram, prob in self.probs[n].items():
                if n == 1 or ngram[:-1] == prefix:
                    word = ngram[-1]
                    result[word] = prob

            # Return if match found at this order
            if result:
                return result

        # No match found at any order
        return {}
    
    def save_model(self, model_path):
        """
        Saves all probability tables to a JSON file in descending order of probability.

        Args:
            model_path (str): Path to save the model JSON file.
        """
        serializable_probs = {}

        for n, probs in self.probs.items():
            key = f"{n}gram"
            serializable_probs[key] = {}

            if n == 1:
                # Unigram: {word: probability} — sorted descending
                unigrams = {}
                for ngram, prob in probs.items():
                    word = ngram[0]
                    unigrams[word] = round(prob, 2)
                serializable_probs[key] = dict(sorted(unigrams.items(), key=lambda x: x[1], reverse=True))

            else:
                # N-gram: {prefix: {word: probability}} — each prefix sorted descending
                for ngram, prob in probs.items():
                    prefix = " ".join(ngram[:-1])
                    word   = ngram[-1]
                    if prefix not in serializable_probs[key]:
                        serializable_probs[key][prefix] = {}
                    serializable_probs[key][prefix][word] = round(prob, 2)

                # Sort each prefix's words by probability descending
                for prefix in serializable_probs[key]:
                    serializable_probs[key][prefix] = dict(sorted(
                        serializable_probs[key][prefix].items(),
                        key=lambda x: x[1],
                        reverse=True
                    ))

        with open(model_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            items = list(serializable_probs.items())
            for i, (gram, data) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                f.write(f'  "{gram}": {json.dumps(data)}{comma}\n')
            f.write("}\n")


    def save_vocab(self, vocab_path):
        """
        Saves vocabulary to a JSON file as a list of words.

        Args:
            vocab_path (str): Path to save the vocabulary JSON file.
        """
        vocab_list = list(self.vocab.keys())

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_list, f, indent=None)


    def load(self, model_path, vocab_path):
        """
        Loads and reconstructs the probability tables and vocabulary from JSON files.

        Args:
            model_path (str): Path to the model JSON file.
            vocab_path (str): Path to the vocabulary JSON file.
        """
        # Load vocabulary and reconstruct the vocab dictionary
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    print(f"Error: vocab file is empty: {vocab_path}")
                    sys.exit(1)
                vocab_list = json.loads(content)
                self.vocab = {word: 0 for word in vocab_list}

        except FileNotFoundError:
            print(f"Error: vocab file not found: {vocab_path}")
            sys.exit(1)

        except json.JSONDecodeError:
            print(f"Error: vocab file is not valid JSON: {vocab_path}")
            sys.exit(1)

        # Load the probability data from the model JSON file
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    print(f"Error: model file is empty: {model_path}")
                    sys.exit(1)
                serializable_probs = json.loads(content)

        except FileNotFoundError:
            # Custom error message for missing model.json
            print(f"Error: model.json not found. Run the Model module first: {model_path}")
            sys.exit(1)

        except json.JSONDecodeError:
            # Custom error message for malformed model.json
            print(f"Error: model.json is malformed. Re-run the Model module: {model_path}")
            sys.exit(1)

        # Reconstruct the self.probs dictionary by converting string keys back into tuples
        self.probs = {}
        for gram, data in serializable_probs.items():
            # Extract the n-order from the key (e.g., '3gram' -> 3)
            n = int(gram.replace("gram", ""))
            self.probs[n] = {}

            if n == 1:
                # Unigrams are stored as single-word tuples
                for word, prob in data.items():
                    self.probs[n][(word,)] = prob
            else:
                # Higher-order n-grams: combine the prefix and current word into a tuple
                for prefix, words in data.items():
                    for word, prob in words.items():
                        ngram = tuple(prefix.split()) + (word,)
                        self.probs[n][ngram] = prob


    # Private helper methods to prevent duplication and keep main methods cleaner
    def _read_sentences(self, token_file):
        """Reads token file and returns list of word lists — no UNK applied."""
        sentences = []
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                words = self.tokenizer(line.strip())
                sentences.append(words)
        return sentences

    def _apply_unk(self, sentences):
        """Replaces words not in vocab with <UNK>."""
        return [[w if w in self.vocab else "<UNK>" for w in words] for words in sentences]