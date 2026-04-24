"""Second-level plugin loader.

Third-party packages can register framework-specific handlers by declaring an
entry point under the groups ``alkaid_keras`` or ``alkaid_torch`` that points
to a zero-argument callable. The callable's job is to import modules whose
class-creation side effects register handlers in the appropriate registry. A
plugin is loaded at most once per Python process.
"""

from functools import cache
from importlib.metadata import entry_points

_LOADED: set[tuple[str, str]] = {('keras', 'keras'), ('torch', 'torch'), ('builtins', 'torch')}  # base frameworks


@cache
def get_plugins():
    _PLUGIN_KERAS = {(ep.name, 'keras'): ep for ep in entry_points().select(group='alkaid_keras')}
    _PLUGIN_TORCH = {(ep.name, 'torch'): ep for ep in entry_points().select(group='alkaid_torch')}
    _PLUGINS = {**_PLUGIN_KERAS, **_PLUGIN_TORCH}
    return _PLUGINS


def maybe_load_for(obj: object, group: str, verbose: bool = False, lax=False) -> bool:
    module_name = getattr(obj, '__module__', None)
    if module_name is None:
        return False
    key = (module_name.split('.', 1)[0], group)
    if key in _LOADED:
        return False
    plugins = get_plugins()
    if lax and key not in plugins:
        return False
    assert key in plugins, f'{key} not found in plugins: {list(plugins.keys())}'
    ep = plugins[key]
    if verbose:
        print(f'[alkaid] loading {ep.group} plugin: {ep.value}')
    ep.load()()
    _LOADED.add(key)
    return True
