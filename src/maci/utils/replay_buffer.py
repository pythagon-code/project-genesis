import numpy as np
from persistent_array import PersistentArray
from memory_state_buffer import MemoryStateBuffer


class ReplayBuffer:
    def __init__(
            self,
            state_buffer: MemoryStateBuffer,
            main_filename: str,
            backup_filename: str,
            item_size: int,
            max_length: int,
            dtype: type,
            sample_size: int,
            rng: np.random.Generator
        ) -> None:
        self._memory_state_buffer = state_buffer
        self._main_buffer = PersistentArray(main_filename, item_size, max_length, dtype, sample_size, rng)
        self._backup_buffer = PersistentArray(backup_filename, item_size, max_length, dtype, sample_size, rng)


    def push(self, item: np.ndarray, step: int) -> None:
        item[-1] = step
        self._main_buffer.push(item)
        self._backup_buffer.push(item)


    def cycle_if_necessary(self) -> None:
        if self._memory_state_buffer.just_cycled():
            tmp = self._main_buffer
            self._main_buffer = self._backup_buffer
            self._backup_buffer = tmp
            self._backup_buffer.reset()


    def flush(self) -> None:
        self._main_buffer.flush()
        self._backup_buffer.flush()
    

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        sample = self._main_buffer.sample()
        sample_memory_states = self._memory_state_buffer.get(sample[:, -1].astype(int))
        return sample[:-1], sample_memory_states