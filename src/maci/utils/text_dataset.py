from .embeddings import generate_embeddings
import nltk
import numpy as np
from tqdm import tqdm

nltk.download("punkt")
nltk.download("brown")
nltk.download("gutenberg")
nltk.download("movie_reviews")


class TextDataset:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_dot_product: float,
        rng: np.random.Generator
    ) -> None:
        corpora = [nltk.corpus.brown, nltk.corpus.gutenberg, nltk.corpus.movie_reviews]
        self._words = [word.lower() for ds in tqdm(corpora, "Unpacking datasets") for word in ds.words()]
        self._word_list = list(list(zip(*nltk.FreqDist(self._words).most_common(vocab_size)))[0])
        self._word_list.insert(0, "<unknown>")
        self._word_to_token_dict = dict(zip(self._word_list, range(len(self._word_list))))
        self._token_to_word_dict = {token: word for word, token in self._word_to_token_dict.items()}
        self._tokens = self.tokenize(self._words)
        self._embeddings = generate_embeddings(vocab_size, embedding_dim, max_dot_product, rng)


    def tokenize(self, words: list[str]) -> list[int]:
        return [self._word_to_token_dict.get(word, 0) for word in words]


    def embed(self, word_ids: list[int]) -> np.ndarray:
        return self._embeddings[word_ids]


    def detokenize(self, word_ids: list[int]) -> list[str]:
        return [self._word_list[word_id] for word_id in word_ids]


if __name__ == "__main__":
    dataset = TextDataset(30000, 500, 0.4, np.random.default_rng())
    print(
        dataset.embed(dataset.tokenize(["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]))
    )

