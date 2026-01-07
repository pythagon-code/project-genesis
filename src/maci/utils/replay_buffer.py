import numpy as np
from pathlib import Path


class ReplayBuffer:
    @staticmethod
    def _get_flat_size(size: tuple[int]) -> int:
        flat_size = 1
        for x in size:
            flat_size *= x
        return flat_size


    def __init__(
            self,
            filename: str,
            subitems_size: list[tuple[int]],
            max_length: int,
            dtype: type,
            sample_size: int,
            rng: np.random.Generator
        ) -> None:
        self._subitems_size = subitems_size
        self._subitems_flat_size = list(map(self._get_flat_size, subitems_size))
        self._max_length = max_length
        self._length = self._end = 0
        self._item_size = sum(self._subitems_flat_size)
        self._sample_size = sample_size
        self._rng = rng

        file_path = Path(filename)
        if not file_path.exists() or not file_path.is_file():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=False)
            np.save(file_path, np.empty(shape=(max_length, self._item_size), dtype=dtype))

        self._array = np.memmap(filename, dtype=dtype, mode="r+", shape=(max_length, self._item_size))
    

    def push(self, subitems: list[np.ndarray]) -> None:
        flat_subitems = []
        for subitem, size in zip(subitems, self._subitems_size):
            assert subitem.shape == size
            flat_subitems.append(subitem.flatten())

        self._array[self._end, :] = np.concat(flat_subitems)
        self._end = (self._end + 1) % self._max_length
        self._length = min(self._length + 1, self._max_length)


    def flush(self) -> None:
        self._array.flush()
    

    def sample(self) -> list[np.ndarray]:
        sample = self._rng.choice(self._array[:self._length], size=min(self._length, self._sample_size))

        subitems = []
        sizes_sum = 0
        for size, flat_size in zip(self._subitems_size, self._subitems_flat_size):
            subitem = np.reshape(sample[:, sizes_sum:sizes_sum + flat_size], shape=(-1, *size))
            sizes_sum += flat_size
            subitems.append(subitem)

        return subitems