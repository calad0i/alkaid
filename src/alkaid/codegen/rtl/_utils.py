import re
import subprocess
import sys
import warnings
from pathlib import Path


def canon_name(name: str) -> str:
    return re.sub(r'\W|^(?=\d)', '_', name)


def verilator_warn_suppression() -> str:
    """Return warning suppressions needed by newer Verilator versions."""
    out = subprocess.run(['verilator', '--version'], capture_output=True, text=True, check=True).stdout
    version_text = re.search(r'Verilator\s+([\d\.]+)', out).group(1)  # type: ignore
    version = float(version_text)
    if version < 5.048:
        warnings.warn(
            f'Verilator {version_text}<5.048 may miscompile the generated RTL, likely due to the issue fixed in '
            'https://github.com/verilator/verilator/pull/7012. '
            'Upgrade to Verilator >= 5.048 for reliable RTL validation.',
            RuntimeWarning,
            stacklevel=2,
        )
    return '-Wno-ALWNEVER' if version >= 5.044 else ''


def run_make_build(
    sim_dir: str | Path,
    makefile: str,
    env: dict[str, str],
    *,
    fast: bool = False,
    clean: bool = True,
    verbose: bool = False,
    stale_lib_re: re.Pattern[str] | None = None,
) -> None:
    """Build a Verilator emulator shared library via ``make -f <makefile>`` in ``sim_dir``."""
    sim_dir = Path(sim_dir)
    args = ['make', '-f', makefile]
    if fast:
        args.append('fast')

    if clean:
        if stale_lib_re is not None:
            for p in sim_dir.iterdir():
                if not p.is_dir() and stale_lib_re.match(p.name):
                    p.unlink()
        subprocess.run(['make', '-f', makefile, 'clean'], env=env, cwd=sim_dir, check=True, capture_output=not verbose)

    try:
        subprocess.run(args, env=env, check=True, cwd=sim_dir, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode(), file=sys.stderr)
        print(e.stdout.decode(), file=sys.stdout)
        raise RuntimeError('Compilation failed!!') from e

    if clean:
        subprocess.run(['rm', '-rf', 'obj_dir'], cwd=sim_dir, check=True, capture_output=not verbose)
