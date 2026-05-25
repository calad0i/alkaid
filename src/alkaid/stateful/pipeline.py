from ..types import Pipeline
from .fsm import FSM, AddrMap, Dir, ModuloSchedule, NamedLogic, NamedPort


def pipeline_to_fsm(pipe: Pipeline) -> FSM:
    lat = len(pipe.solutions)
    assert lat > 1, 'Pipeline must not be empty'

    logics = tuple(NamedLogic(f'logic{i}', sol) for i, sol in enumerate(pipe.solutions))

    _inp_precisions = tuple(qint.kif for qint in pipe.inp_qint)
    _out_precisions = tuple(qint.kif for qint in pipe.out_qint)

    inp_port = NamedPort('model_inp', Dir.IN, _inp_precisions, schedule=ModuloSchedule((0,), 1))
    out_port = NamedPort('model_out', Dir.OUT, _out_precisions, schedule=ModuloSchedule((lat,), 1))
    _ports = []

    for i in range(1, lat):
        _kifs = tuple(qint.kif for qint in pipe.solutions[i].inp_qint)
        _ports.append(NamedPort(f'stage{i}_inp', Dir.INTERNAL, _kifs, need_rst=False))

    ports = [inp_port] + _ports + [out_port]

    addr_maps: list[AddrMap] = []
    for i in range(lat):
        n_in, n_out = pipe.solutions[i].shape
        addr_map_in = AddrMap(ports[i].name, (0, n_in), logics[i].name, (0, n_in))
        addr_map_out = AddrMap(logics[i].name, (0, n_out), ports[i + 1].name, (0, n_out))
        addr_maps.append(addr_map_in)
        addr_maps.append(addr_map_out)

    return FSM(logics, tuple(ports), tuple(addr_maps))
