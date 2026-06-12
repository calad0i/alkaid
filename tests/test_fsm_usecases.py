import shutil
from pathlib import Path

import numpy as np
import pytest

from alkaid.codegen.rtl.fsm_model import FSMProject
from alkaid.stateful import FSM, Conn, ModuloSchedule, Signal
from alkaid.stateful.fsm import FSMEmu, _comb_io_signals
from alkaid.trace import FVArray, trace
from alkaid.trace.ops import quantize
from alkaid.trace.passes import optimize


def _require_verilator():
    if shutil.which('verilator') is None:
        pytest.skip('verilator not found')


def _fsmemu_step_run(emu: FSMEmu, data: dict, steps: int) -> dict[str, np.ndarray]:
    """Cycle-accurate Python reference from the emulator's current state."""
    out = {p.name: np.empty((steps, p.size), dtype=np.float64) for p in emu.fsm.out_signals}
    for step in range(steps):
        for p in emu.fsm.inp_signals:
            emu.buffers[p.name][:] = data[p.name][step]
        emu.eval()
        emu.tick()
        emu.eval()
        for p in emu.fsm.out_signals:
            out[p.name][step] = emu.buffers[p.name]
    return out


def _fsmemu_run(fsm: FSM, data: dict, steps: int) -> dict[str, np.ndarray]:
    emu = FSMEmu(fsm)
    emu.soft_reset()
    return _fsmemu_step_run(emu, data, steps)


def _rtl_step_run(emu: FSMProject, data: dict, steps: int) -> dict[str, np.ndarray]:
    """Exercise the RTL binder's scalar step API from the DUT's current state."""
    data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    out = {p.name: np.empty((steps, p.size), dtype=np.float64) for p in emu.fsm.out_signals}
    for step in range(steps):
        for p in emu.fsm.inp_signals:
            samples = data[p.name]
            value = samples[step] if samples.ndim == 1 and p.size == 1 else samples.reshape(samples.shape[0], p.size)[step]
            emu.set_port(p.name, value)
        emu.eval()
        emu.tick()
        emu.eval()
        for p in emu.fsm.out_signals:
            out[p.name][step] = emu.get_port(p.name)
    return out


def _run_both(fsm: FSM, data: dict, prj_name: str, path, steps: int | None = None) -> tuple[dict, dict]:
    """Compare the Verilated emulator against the cycle-accurate Python reference."""
    if steps is None:
        steps = min(len(np.asarray(data[p.name])) for p in fsm.inp_signals)
    data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    ref_emu = FSMEmu(fsm)
    ref_emu.soft_reset()
    ref = _fsmemu_step_run(ref_emu, data, steps)
    emu = FSMProject(fsm, path, prj_name=prj_name)
    emu.compile(nproc=1)
    emu.soft_reset()
    got = emu.run(data, steps=steps, scheduled=False)
    ref_step = _fsmemu_step_run(ref_emu, data, steps)
    step_got = _rtl_step_run(emu, data, steps)
    for name, expected in ref_step.items():
        np.testing.assert_array_equal(step_got[name], expected)
    return ref, got


def _trace_comb(k, i, f, expr):
    """Trace a CombLogic from an N-input FVArray; ``expr(inp)`` returns the (scalar) output."""
    inp = FVArray.from_kif(np.asarray(k), np.asarray(i), np.asarray(f))
    return optimize(trace(inp, expr(inp), optimize=False))


# --------------------------------------------------------------------------- MAC


def _mac_fsm():
    """acc <= acc + a*b  (8-bit unsigned operands, 20-bit accumulator)."""
    comb = _trace_comb([0, 0, 0], [20, 8, 8], [0, 0, 0], lambda x: quantize(x[0] + x[1] * x[2], 0, 20, 0))
    sin, sout = _comb_io_signals('mac', comb)
    a = Signal('a', True, ((0, 8, 0),), reg=False, mode='r')
    b = Signal('b', True, ((0, 8, 0),), reg=False, mode='r')
    acc = Signal('acc', False, ((0, 20, 0),), reg=True, mode='rw', rst_to=(0,))
    y = Signal('y', True, ((0, 20, 0),), reg=False, mode='w')
    conns = (
        Conn(acc, sin[0:1]),
        Conn(a, sin[1:2]),
        Conn(b, sin[2:3]),
        Conn(sout, acc),  # clocked accumulate
        Conn(acc, y),
    )
    return FSM({'mac': comb}, conns)


def test_mac_accumulator(temp_directory):
    _require_verilator()
    rng = np.random.default_rng(1)
    n = 24
    a = rng.integers(0, 16, n).astype(np.float64)
    b = rng.integers(0, 16, n).astype(np.float64)
    fsm = _mac_fsm()
    ref, got = _run_both(fsm, {'a': a, 'b': b}, 'mac_top', Path(temp_directory) / 'mac')
    np.testing.assert_array_equal(got['y'], ref['y'])
    np.testing.assert_array_equal(got['y'][:, 0], np.cumsum(a * b))  # no overflow at these magnitudes


# --------------------------------------------------------------------------- FIR


def _fir_fsm(coeffs):
    """y = sum_i coeff[i] * tap[i]; taps are a delay line fed from x (constant-coeff dot product)."""
    n = len(coeffs)
    comb = _trace_comb([1] * n, [8] * n, [0] * n, lambda x: quantize(np.dot(x, coeffs), 1, 16, 4))
    sin, sout = _comb_io_signals('fir', comb)
    x = Signal('x', True, ((1, 8, 0),), reg=False, mode='r')
    taps = [Signal(f'tap{i}', False, ((1, 8, 0),), reg=True, mode='rw', rst_to=(0,)) for i in range(n)]
    y = Signal('y', True, ((1, 16, 4),), reg=False, mode='w')
    conns = [Conn(x, taps[0])]
    conns += [Conn(taps[i - 1], taps[i]) for i in range(1, n)]  # shift register
    conns += [Conn(taps[i], sin[i : i + 1]) for i in range(n)]
    conns.append(Conn(sout, y))
    return FSM({'fir': comb}, tuple(conns))


def test_fir_filter(temp_directory):
    _require_verilator()
    coeffs = [0.5, 0.25, 0.125]  # exactly representable with 4 fractional bits
    fsm = _fir_fsm(coeffs)
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 128, 30).astype(np.float64)
    ref, got = _run_both(fsm, {'x': x}, 'fir_top', Path(temp_directory) / 'fir')
    np.testing.assert_array_equal(got['y'], ref['y'])

    # NumPy reference: y[n] = sum_i coeff[i] * x[n-i] (exact at these coefficients/inputs).
    expected = np.zeros(len(x))
    for n in range(len(x)):
        expected[n] = sum(coeffs[i] * x[n - i] for i in range(len(coeffs)) if n - i >= 0)
    np.testing.assert_array_equal(got['y'][:, 0], expected)


# --------------------------------------------------------------------------- IIR


def _iir_fsm(a=0.5, b=0.5):
    """Single-pole feedback: y <= quantize(b*x + a*y_prev), driven by a CombLogic block."""
    comb = _trace_comb([1, 1], [8, 8], [4, 4], lambda v: quantize(v[0] * b + v[1] * a, 1, 8, 4))
    sin, sout = _comb_io_signals('iir', comb)
    x = Signal('x', True, ((1, 8, 4),), reg=False, mode='r')
    yreg = Signal('yreg', False, ((1, 8, 4),), reg=True, mode='rw', rst_to=(0,))
    y = Signal('y', True, ((1, 8, 4),), reg=False, mode='w')
    conns = (
        Conn(x, sin[0:1]),
        Conn(yreg, sin[1:2]),  # y_prev feedback
        Conn(sout, yreg),  # clocked
        Conn(yreg, y),
    )
    return FSM({'iir': comb}, conns)


def test_iir_filter(temp_directory):
    _require_verilator()
    fsm = _iir_fsm(a=0.5, b=0.5)
    # Impulse then zeros: y = 0.5, 0.25, 0.125, 0.0625, 0, ... (truncates below the 1/16 grid).
    x = np.zeros(8, dtype=np.float64)
    x[0] = 1.0
    ref, got = _run_both(fsm, {'x': x}, 'iir_top', Path(temp_directory) / 'iir')
    np.testing.assert_array_equal(got['y'], ref['y'])
    np.testing.assert_array_equal(got['y'][:, 0], np.array([0.5, 0.25, 0.125, 0.0625, 0.0, 0.0, 0.0, 0.0]))


# ----------------------------------------------------------------------- Counter


def _counter_fsm():
    """4-bit up counter: count <= rst ? 0 : (en ? count+1 : count); wraps mod 16."""
    comb = _trace_comb([0], [4], [0], lambda x: quantize(x[0] + 1.0, 0, 4, 0))  # +1 mod 16
    sin, sout = _comb_io_signals('inc', comb)
    rst = Signal('rst', True, ((0, 1, 0),), reg=False, mode='r')  # active-high reset
    en = Signal('en', True, ((0, 1, 0),), reg=False, mode='r')
    count = Signal('count', False, ((0, 4, 0),), reg=True, mode='rw', rst_if=rst, rst_to=(0,))
    count_fb = Signal('count_fb', False, ((0, 4, 0),), reg=False)  # mirror for the hold path
    y = Signal('y', True, ((0, 4, 0),), reg=False, mode='w')
    conns = (
        Conn(count, sin[0:1]),
        Conn(count, count_fb),
        Conn(sout, count, enable_if=en, alt_src=count_fb),  # count <= en ? count+1 : count
        Conn(count, y),
    )
    return FSM({'inc': comb}, conns)


def test_counter(temp_directory):
    _require_verilator()
    fsm = _counter_fsm()
    # Count up past the 4-bit wrap, hold while disabled, then synchronous reset (active-high).
    rst = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float64)
    en = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    ref, got = _run_both(fsm, {'rst': rst, 'en': en}, 'cnt_top', Path(temp_directory) / 'cnt')
    np.testing.assert_array_equal(got['y'], ref['y'])

    expected, c = [], 0
    for k in range(len(en)):
        c = 0 if rst[k] == 1 else ((c + 1) % 16 if en[k] else c)
        expected.append(c)
    np.testing.assert_array_equal(got['y'][:, 0], np.array(expected, dtype=np.float64))


# -------------------------------------------------------------------------- FIFO


def _fifo_fsm(w=8, depth=4):
    """Circular FIFO: mem[wptr] <= din each cycle, wptr/rptr advance only on wr_en/rd_en,
    dout = mem[rptr]. The enable-gated pointers reuse the mirror-and-mux load-enable
    pattern; the memory uses dynamic-bias addressing."""
    a = (depth - 1).bit_length()
    word = (0, w, 0)
    inc = _trace_comb([0], [a], [0], lambda x: quantize(x[0] + 1.0, 0, a, 0))  # +1 mod depth
    siw, sow = _comb_io_signals('incw', inc)
    sir, sor = _comb_io_signals('incr', inc)

    wr_en = Signal('wr_en', True, ((0, 1, 0),), reg=False, mode='r')
    rd_en = Signal('rd_en', True, ((0, 1, 0),), reg=False, mode='r')
    din = Signal('din', True, (word,), reg=False, mode='r')
    dout = Signal('dout', True, (word,), reg=False, mode='w')

    wptr = Signal('wptr', False, ((0, a, 0),), reg=True, mode='rw', rst_to=(0,))
    rptr = Signal('rptr', False, ((0, a, 0),), reg=True, mode='rw', rst_to=(0,))
    wptr_fb = Signal('wptr_fb', False, ((0, a, 0),), reg=False)
    rptr_fb = Signal('rptr_fb', False, ((0, a, 0),), reg=False)
    mem_w = Signal('mem', False, (word,) * depth, reg=True, view=(0, 1), _dynamic_bias=(wptr, 1))
    mem_r = Signal('mem', False, (word,) * depth, reg=True, view=(0, 1), _dynamic_bias=(rptr, 1))
    mem_at_w = Signal('mem', False, (word,) * depth, reg=True, view=(0, 1), _dynamic_bias=(wptr, 1))
    mem_hold = Signal('mem_hold', False, (word,), reg=False)  # combinational mirror of mem[wptr]

    conns = (
        Conn(mem_at_w, mem_hold),  # mem_hold = mem[wptr] (comb)
        Conn(din, mem_w, enable_if=wr_en, alt_src=mem_hold),  # mem[wptr] <= wr_en ? din : mem[wptr]
        Conn(wptr, siw[0:1]),
        Conn(wptr, wptr_fb),
        Conn(sow, wptr, enable_if=wr_en, alt_src=wptr_fb),  # wptr advances only on wr_en
        Conn(rptr, sir[0:1]),
        Conn(rptr, rptr_fb),
        Conn(sor, rptr, enable_if=rd_en, alt_src=rptr_fb),  # rptr advances only on rd_en
        Conn(mem_r, dout),  # dout = mem[rptr] (comb, dynamic bias)
    )
    return FSM({'incw': inc, 'incr': inc}, conns)


def test_sync_fifo(temp_directory):
    _require_verilator()
    w, depth = 8, 4
    fsm = _fifo_fsm(w, depth)
    rng = np.random.default_rng(7)

    # Fill the FIFO, then drain it: the drained dout sequence must equal what was pushed.
    pushed = rng.integers(0, 1 << w, depth).astype(np.float64)
    wr_en = np.concatenate([np.ones(depth), np.zeros(depth)])
    rd_en = np.concatenate([np.zeros(depth), np.ones(depth)])
    din = np.concatenate([pushed, np.zeros(depth)])

    ref, got = _run_both(fsm, {'wr_en': wr_en, 'rd_en': rd_en, 'din': din}, 'fifo_top', Path(temp_directory) / 'fifo')
    # First-word-fall-through: mem[rptr] shows each stored word the cycle before its pop
    # completes, so the head walks the FIFO in push order over this window.
    np.testing.assert_array_equal(got['dout'], ref['dout'])
    np.testing.assert_array_equal(got['dout'][depth - 1 : 2 * depth - 1, 0], pushed)


# ----------------------------------------------------------- 2x2 systolic array


def _systolic2x2_fsm():
    """Output-stationary 2x2 systolic matmul C = A @ B.

    Each PE(i,j) holds an accumulator and a MAC block (acc <= acc + a_in*b_in); `a`
    operands flow left->right and `b` operands flow top->bottom through pass-through
    registers.  Driven with the standard skewed operand streams, the four accumulators
    settle to the four entries of C.
    """
    logic, sin, sout = {}, {}, {}
    for i in range(2):
        for j in range(2):
            comb = _trace_comb([1, 1, 1], [18, 7, 7], [0, 0, 0], lambda x: quantize(x[0] + x[1] * x[2], 1, 18, 0))
            logic[f'mac{i}{j}'] = comb
            sin[i, j], sout[i, j] = _comb_io_signals(f'mac{i}{j}', comb)

    ar = [Signal(f'ar{i}', True, ((1, 7, 0),), reg=False, mode='r') for i in range(2)]
    bc = [Signal(f'bc{j}', True, ((1, 7, 0),), reg=False, mode='r') for j in range(2)]
    # Only the edge PEs forward operands: areg[i] passes A across row i, breg[j] passes B down col j.
    areg = [Signal(f'areg{i}', False, ((1, 7, 0),), reg=True, mode='rw', rst_to=(0,)) for i in range(2)]
    breg = [Signal(f'breg{j}', False, ((1, 7, 0),), reg=True, mode='rw', rst_to=(0,)) for j in range(2)]
    acc = {ij: Signal(f'acc{ij[0]}{ij[1]}', False, ((1, 18, 0),), reg=True, mode='rw', rst_to=(0,)) for ij in sin}
    cout = {ij: Signal(f'c{ij[0]}{ij[1]}', True, ((1, 18, 0),), reg=False, mode='w') for ij in sin}

    def a_in(i, j):
        return ar[i] if j == 0 else areg[i]  # column 0 from the left edge, else from the left neighbor's register

    def b_in(i, j):
        return bc[j] if i == 0 else breg[j]  # row 0 from the top edge, else from the upper neighbor's register

    conns = [Conn(ar[i], areg[i]) for i in range(2)] + [Conn(bc[j], breg[j]) for j in range(2)]
    for i in range(2):
        for j in range(2):
            ai, bi = a_in(i, j), b_in(i, j)
            conns += [Conn(acc[i, j], sin[i, j][0:1]), Conn(ai, sin[i, j][1:2]), Conn(bi, sin[i, j][2:3])]
            conns += [Conn(sout[i, j], acc[i, j]), Conn(acc[i, j], cout[i, j])]
    return FSM(logic, tuple(conns))


def test_systolic_2x2(temp_directory):
    _require_verilator()
    fsm = _systolic2x2_fsm()
    rng = np.random.default_rng(11)
    A = rng.integers(-8, 8, (2, 2)).astype(np.float64)
    B = rng.integers(-8, 8, (2, 2)).astype(np.float64)
    z = 0.0
    # Standard skewed feed: row i of A enters left edge i cycles late; col j of B likewise.
    data = {
        'ar0': np.array([A[0, 0], A[0, 1], z, z, z, z]),
        'ar1': np.array([z, A[1, 0], A[1, 1], z, z, z]),
        'bc0': np.array([B[0, 0], B[1, 0], z, z, z, z]),
        'bc1': np.array([z, B[0, 1], B[1, 1], z, z, z]),
    }
    ref, got = _run_both(fsm, data, 'sys_top', Path(temp_directory) / 'sys', steps=6)
    for ij in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        np.testing.assert_array_equal(got[f'c{ij[0]}{ij[1]}'], ref[f'c{ij[0]}{ij[1]}'])
    c_emu = np.array([[got[f'c{i}{j}'][-1, 0] for j in range(2)] for i in range(2)])
    np.testing.assert_array_equal(c_emu, A @ B)


# -------------------------------------------------------------


def _pair_sum_scheduled_fsm():
    """Accumulate pairs: output x[0]+x[1], x[2]+x[3], ... on every other cycle."""
    add = _trace_comb([1, 1], [9, 8], [0, 0], lambda x: quantize(x[0] + x[1], 1, 9, 0))
    sin_add, sout_add = _comb_io_signals('add', add)
    toggle = _trace_comb([0], [1], [0], lambda x: quantize(x[0] + 1.0, 0, 1, 0))
    sin_toggle, sout_toggle = _comb_io_signals('toggle', toggle)

    inp = Signal('inp', True, ((1, 8, 0),), reg=False, schedule=ModuloSchedule((0,), 2), mode='r')
    out = Signal('out', True, ((1, 9, 0),), reg=False, schedule=ModuloSchedule((0, 1), 2), mode='w')
    acc = Signal('acc', False, ((1, 9, 0),), reg=True, mode='rw', rst_to=(0,))
    phase = Signal('phase', False, ((0, 1, 0),), reg=True, mode='rw', rst_to=(0,))
    conns = (
        Conn(acc, sin_add[0:1]),
        Conn(inp, sin_add[1:2]),
        Conn(sout_add, acc, enable_if=phase, alt_src=inp),
        Conn(phase, sin_toggle[0:1]),
        Conn(sout_toggle, phase),
        Conn(acc, out),
    )
    return FSM({'add': add, 'toggle': toggle}, conns)


def test_scheduled_run_pair_sum(temp_directory):
    _require_verilator()
    fsm = _pair_sum_scheduled_fsm()
    data = {'inp': np.array([2, -3, 4, 5, -6, 7], dtype=np.float64)}
    expected = np.array([[-1], [9], [1]], dtype=np.float64)

    ref = FSMEmu(fsm)
    ref.soft_reset()
    ref_out = ref.run(data, scheduled=True)
    np.testing.assert_array_equal(ref_out['out'], expected)

    emu = FSMProject(fsm, Path(temp_directory) / 'scheduled_pair_sum', prj_name='pair_sched_top')
    emu.compile(nproc=1)
    emu.soft_reset()
    got = emu.run(data, scheduled=True)
    np.testing.assert_array_equal(got['out'], ref_out['out'])
    np.testing.assert_array_equal(got['out'], expected)
