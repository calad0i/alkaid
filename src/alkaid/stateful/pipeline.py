from ..types import CombLogic, Pipeline
from .fsm import FSM, Conn, ModuloSchedule, Signal, _comb_io_signals


def pipeline_to_fsm(pipe: Pipeline, reg_inp=True, reg_out=True) -> FSM:
    lat = len(pipe.solutions)
    assert lat > 0, 'Pipeline must not be empty'

    inp_precisions = tuple(qint.kif for qint in pipe.inp_qint)
    out_precisions = tuple(qint.kif for qint in pipe.out_qint)
    io_period = 0 if lat == 1 and not reg_inp and not reg_out else 1

    conns: list[Conn] = []
    inp_sig = Signal(
        'model_inp',
        True,
        inp_precisions,
        reg=False,
        mode='r',
        schedule=ModuloSchedule((0,), io_period),
    )
    out_sig = Signal(
        'model_out',
        True,
        out_precisions,
        reg=reg_out,
        mode='w',
        schedule=ModuloSchedule((lat - 1 + reg_out + reg_inp,), io_period),
    )

    if reg_inp:
        inp_sig_reg = Signal(
            'model_inp_reg',
            False,
            inp_precisions,
            reg=True,
            mode='rw',
        )
        conns.append(Conn(inp_sig, inp_sig_reg))
        inp_sig = inp_sig_reg

    ports: list[Signal] = [inp_sig]
    for i in range(1, lat):
        kifs = tuple(qint.kif for qint in pipe.solutions[i].inp_qint)
        ports.append(Signal(f'stage{i}_inp', False, kifs, reg=True, mode=''))
    ports.append(out_sig)

    logic: dict[str, CombLogic] = {}
    for i in range(lat):
        comb = pipe.solutions[i]
        n_in, n_out = comb.shape
        if n_in == 0 and n_out == 0:
            continue
        name = f'logic{i}'
        logic[name] = comb
        sig_in, sig_out = _comb_io_signals(name, comb)
        # connect each side that exists; a stage may consume inputs without
        # producing outputs (e.g. a constant model discards its input), so the
        # input port stays wired even when there is nothing downstream
        if n_in > 0:
            conns.append(Conn(ports[i], sig_in))  # combinational: register -> logic input wire
        if n_out > 0:
            conns.append(Conn(sig_out, ports[i + 1]))  # registered: logic output wire -> next register

    return FSM(logic, tuple(conns))
