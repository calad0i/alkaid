from types import SimpleNamespace

from alkaid.codegen.rtl import _utils


def test_verilator_warn_suppression_disables_expected_generated_rtl_warnings(monkeypatch):
    monkeypatch.setattr(
        _utils.subprocess,
        'run',
        lambda *_args, **_kwargs: SimpleNamespace(stdout='Verilator 5.048 2026-01-01\n'),
    )

    flags = _utils.verilator_warn_suppression().split()

    assert '-Wno-UNUSEDSIGNAL' in flags
    assert '-Wno-ALWNEVER' in flags
