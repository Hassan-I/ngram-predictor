class Predictor:
    def __init__(self, model, normalizer):
        """
        Accepts a pre-loaded NGramModel and Normalizer instance.
        Does not load any files.

        Args:
            model (NGramModel): Pre-loaded NGramModel instance.
            normalizer (Normalizer): Pre-loaded Normalizer instance.
        """
        self.model      = model
        self.normalizer = normalizer

    
    def normalize(self, text):
        """
        Normalizes input text and extracts the last NGRAM_ORDER - 1 words as context.

        Args:
            text (str): Input text string.

        Returns:
            list: Last NGRAM_ORDER - 1 words as context.
     """
        # Normalize the input text
        normalized = self.normalizer.normalize(text)

        # Tokenize into words
        words = self.normalizer.word_tokenize(normalized)

        # Extract last NGRAM_ORDER - 1 words as context
        context = words[-(self.model.ngram_order - 1):]

        return context
    
    def map_oov(self, context):
        """
        Replaces out-of-vocabulary words with <UNK>.

        Args:
            context (list): List of words representing the context.

        Returns:
            list: Context with OOV words replaced by <UNK>.
        """
        return [w if w in self.model.vocab else "<​UNK>" for w in context]

    def predict_next(self, text, k):
        """
        Orchestrates normalize → map_oov → lookup → return top-k predictions.

        Args:
            text (str): Input text string.
            k (int): Number of top predictions to return.

        Returns:
            list: Top-k predicted next words sorted by probability (highest first).
                Returns empty list if no predictions found or error occurs.
        """
        try:
            # Check for empty or whitespace-only strings
            if not text or not text.strip():
                raise ValueError("Input text is empty. Please type at least one word.")

            # Step 1 — Normalize and extract context
            context = self.normalize(text)

            # Step 2 — Replace OOV words with <UNK>
            context = self.map_oov(context)

            # Step 3 — Backoff lookup
            candidates = self.model.lookup(context)

            # Step 4 — Sort by probability and return top-k
            if not candidates:
                return []

            # Sort by probability value in descending order
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

            return [word for word, prob in sorted_candidates[:k]]

        except ValueError as e:
            # Catch the specific empty string error
            print(e)
            return []