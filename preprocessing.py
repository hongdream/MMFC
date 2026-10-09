from collections import Counter
import logging
from time import perf_counter
import numpy as np

LOGGER = logging.getLogger("stfed")


class TextCorpusPreprocessor:

    def __init__(self, documents):
        self.documents = list(documents)

    def _lemmatize_batch(self, documents):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        stop = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        result = []
        total = len(documents)
        started = perf_counter()
        report_every = max(100, total // 10)
        for index, document in enumerate(documents, start=1):
            tokens = [token.lower() for token in nltk.word_tokenize(document) if token.isalpha()]
            tagged = nltk.pos_tag(tokens)
            result.append(
                [
                    lemmatizer.lemmatize(
                        token, {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[0], "n")
                    )
                    for token, tag in tagged
                    if token not in stop
                ]
            )
            if index == 1 or index == total or index % report_every == 0:
                LOGGER.info(
                    "[TEXT] lemmatized=%d/%d (%.1f%%) | elapsed=%.1fs",
                    index,
                    total,
                    100 * index / max(1, total),
                    perf_counter() - started,
                )
        return result

    def build_bag_of_words(self, documents, min_token_frequency=10, max_token_frequency_ratio=0.05):
        started = perf_counter()
        LOGGER.info(
            "[TEXT] vocabulary construction START | min_frequency=%d | max_document_ratio=%.4f",
            min_token_frequency,
            max_token_frequency_ratio,
        )
        counts = [Counter(doc) for doc in documents]
        total, document_frequency = (Counter(), Counter())
        for count in counts:
            total.update(count)
            document_frequency.update(count.keys())
        vocabulary = sorted(
            (
                token
                for token, count in total.items()
                if count >= min_token_frequency
                and document_frequency[token] / max(1, len(documents)) <= max_token_frequency_ratio
            )
        )
        self.id_to_token = dict(enumerate(vocabulary))
        lookup = {token: i for i, token in self.id_to_token.items()}
        self.bag_of_words = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
        for row, count in enumerate(counts):
            for token, value in count.items():
                if token in lookup:
                    self.bag_of_words[row, lookup[token]] = value
        LOGGER.info(
            "[TEXT] vocabulary construction DONE | vocabulary=%d | matrix=%s | elapsed=%.1fs",
            len(vocabulary),
            tuple(self.bag_of_words.shape),
            perf_counter() - started,
        )
        return self.bag_of_words
