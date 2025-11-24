import numpy as np
from persistent_array import PersistentArray


class MemoryStateBuffer:
    def __init__(
            self,
            main_filename: str,
            backup_filename: str,
            item_size: int,
            max_length: int,
            dtype: type,
            main_offset_step: int,
            backup_offset_step: int
        ) -> None:
        self._main_buffer = PersistentArray(main_filename, item_size, max_length, dtype, 0, None)
        self._backup_buffer = PersistentArray(backup_filename, item_size, max_length, dtype, 0, None)
        self._max_length = max_length
        self._main_offset_step = main_offset_step
        self._backup_offset_step = backup_offset_step
        self._just_cycled = True


    def push(self, item: np.ndarray, step: int):
        if (self._main_buffer.restarted() and not self._just_cycled) or step == self._max_length // 2:
            self._cycle(step)
        else:
            self._just_cycled = False

        self._main_buffer.push(item)
        self._backup_buffer.push(item)


    def _cycle(self, step: int):
        tmp = self._main_buffer
        self._main_buffer = self._backup_buffer
        self._backup_buffer = tmp
        self._backup_buffer.reset()
        self._main_offset_step = self._backup_offset_step
        self._backup_offset_step = step
        self._just_cycled = True


    def flush(self) -> None:
        self._main_buffer.flush()
        self._backup_buffer.flush()


    def just_cycled(self) -> bool:
        return self._just_cycled
    

    def get(self, steps: np.ndarray) -> np.ndarray:
        indices = steps - self._main_offset_step
        assert ((indices >= 0) & (indices < self._max_length)).all()
        return self._main_buffer.get(indices)