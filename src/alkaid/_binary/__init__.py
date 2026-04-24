import numpy as np
from numpy.typing import NDArray

from .cmvm_bin import (
    csd_decompose,
    get_lsb_loc,
    iceil_log2,
    int_arr_to_csd,
    kernel_decompose,
    minimal_kif_batch,
    minimal_kif_scalar,
    overlap_counts,
    solve,
)


def alir_interp_run(bin_logic: NDArray[np.int32], data: NDArray, n_threads: int = 1, dump: bool = False):
    from .alir_bin import run_interp

    inp_size = int(bin_logic[2])
    assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'

    inputs = np.ascontiguousarray(np.ravel(data), dtype=np.float64)
    bin_logic = np.ascontiguousarray(np.ravel(bin_logic), dtype=np.int32)
    return run_interp(bin_logic, inputs, n_threads, dump=dump)

__all__ = [
    'alir_interp_run',
    'int_arr_to_csd',
    'csd_decompose',
    'get_lsb_loc',
    'kernel_decompose',
    'minimal_kif_batch',
    'minimal_kif_scalar',
    'solve',
    'iceil_log2',
    'overlap_counts',
]
