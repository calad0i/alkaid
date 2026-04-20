"""Second-level plugin loader.

Third-party packages can register framework-specific handlers by declaring an
entry point under the groups ``alkaid_keras`` or ``alkaid_torch`` that points
to a zero-argument callable. The callable's job is to import modules whose
class-creation side effects register handlers in the appropriate registry.

A plugin is loaded at most once per Python process: the first time the tracer
encounters a class / callable whose owning distribution provides the entry
point, the registration function is executed and the distribution is
remembered in ``_LOADED``.
"""

from importlib.metadata import entry_points, packages_distributions
from typing import Any

_LOADED: dict[str, set[str]] = {}


def maybe_load_for_module(module_name: str, group: str, verbose: bool = False) -> bool:
    """Load any ``group`` plugin contributed by the distribution owning the
    top-level package of ``module_name``. Returns True iff at least one entry
    point was newly executed.
    """
    if not module_name:
        return False
    loaded = _LOADED.setdefault(group, set())
    top = module_name.split('.', 1)[0]
    dists = packages_distributions().get(top, ())
    fired = False
    for dist_name in dists:
        if dist_name in loaded:
            continue
        loaded.add(dist_name)
        for ep in entry_points().select(group=group):
            if ep.dist is None or ep.dist.name != dist_name:
                continue
            if verbose:
                print(f'[alkaid] loading {group} plugin: {ep.value}')
            ep.load()()
            fired = True
    return fired


def maybe_load_for_class(cls: type, group: str, verbose: bool = False) -> bool:
    return maybe_load_for_module(cls.__module__, group, verbose)


def maybe_load_for_callable(fn: Any, group: str, verbose: bool = False) -> bool:
    return maybe_load_for_module(getattr(fn, '__module__', '') or '', group, verbose)
