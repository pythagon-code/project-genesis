import numpy as np
from tqdm import tqdm


def generate_embeddings(
    num_embeddings: int,
    embedding_dim: int,
    max_dot_product: float,
    rng: np.random.Generator
) -> np.ndarray | None:
    embeddings = np.zeros((num_embeddings, embedding_dim))
    for i in tqdm(range(num_embeddings), "Generating embeddings"):
        found = False
        for _ in range(1000):
            emb = rng.standard_normal(embedding_dim)
            emb /= np.linalg.norm(emb)
            if i == 0 or np.max(embeddings @ emb) <= max_dot_product:
                embeddings[i] = emb
                found = True
                break

        if not found:
            return None

    return embeddings


if __name__ == "__main__":
    print(generate_embeddings(30001, 500, 0.4, np.random.default_rng()))