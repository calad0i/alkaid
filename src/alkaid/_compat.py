from collections.abc import Sequence

from .types import Op, QInterval


def _s32(v: int) -> int:
    return ((int(v) & 0xFFFFFFFF) + 0x80000000) % 0x100000000 - 0x80000000


def _s64(v: int) -> int:
    v = int(v)
    if v > 0x7FFFFFFFFFFFFFFF:
        v -= 0x10000000000000000
    return v


def _op_from_v2_record(record: Sequence) -> Op:
    """Convert one v2 JSON op record to the v3 tuple-address format."""

    id0, id1, opcode, packed_data = int(record[0]), int(record[1]), int(record[2]), _s64(int(record[3]))
    qint = QInterval(*record[4])
    latency, cost = record[5], record[6]

    match opcode:
        case -2:
            op = Op((id0,), opcode, (), qint, latency, cost)
        case -1:
            op = Op((), opcode, (id0,), qint, latency, cost)
        case 0 | 1:
            op = Op((id0, id1), opcode, (packed_data,), qint, latency, cost)
        case 2 | 3:
            op = Op((id0,), opcode, (), qint, latency, cost)
        case 4:
            op = Op((id0,), opcode, (_s32(packed_data), _s32(packed_data >> 32)), qint, latency, cost)
        case 5:
            op = Op((), opcode, (packed_data,), qint, latency, cost)
        case 6:
            op = Op((id0, id1, _s32(packed_data)), opcode, (_s32(packed_data >> 32),), qint, latency, cost)
        case 7:
            op = Op((id0, id1), opcode, (), qint, latency, cost)
        case 8:
            op = Op((id0,), opcode, (packed_data,), qint, latency, cost)
        case 9:
            op = Op((id0,), opcode, (packed_data,), qint, latency, cost)
        case 10:
            op = Op((id0, id1), opcode, (_s32(packed_data), (packed_data >> 56) & 0xFF), qint, latency, cost)
        case _:
            raise ValueError(f'Unknown v2 opcode {opcode}')
    return op


def _op_from_v3_record(record: Sequence) -> Op:
    addr, opcode, payload, qint, latency, cost = record
    return Op(tuple(int(v) for v in addr), int(opcode), tuple(int(v) for v in payload), QInterval(*qint), latency, cost)


def _upgrade_v2_to_v3(data: Sequence) -> list:
    data = list(data)
    data[5] = [_op_from_v2_record(op) for op in data[5]]
    return data


def _upgrade_v3_to_v4(data: Sequence) -> list:
    data = list(data)
    ops = []
    for record in data[5]:
        op = _op_from_v3_record(record)
        if op.opcode == 3:
            op = Op(op.addr, op.opcode, (0,), op.qint, op.latency, op.cost)
        ops.append(op)
    data[5] = ops
    return data


_UPGRADE_STEPS = {
    2: (3, _upgrade_v2_to_v3),
    3: (4, _upgrade_v3_to_v4),
}


def compatible_upgrade_versions() -> tuple[int, ...]:
    return tuple(sorted(_UPGRADE_STEPS))


def upgrade_model_data(data: Sequence, spec_version: int | None, target_version: int) -> list:
    data = list(data)
    while spec_version != target_version:
        if spec_version not in _UPGRADE_STEPS:
            raise ValueError(f'Cannot upgrade ALIR spec version {spec_version} to {target_version}')
        spec_version, upgrade = _UPGRADE_STEPS[spec_version]
        data = upgrade(data)
    return data
