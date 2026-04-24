# Installation Guide

`alkaid` is available on PyPI and can be installed with pip.

```bash
pip install alkaid
```

## Requirements

### Binary Wheels

 - `python>=3.10`
 - `numpy>=1.26.4`
 - `quantizers>=1,<2`
 - Linux x86_64 or macOS ARM64

### Building from Source

 - `python>=3.10`
 - `numpy>=2` for the isolated build environment
 - `meson-python>=0.13.1`
 - A C++20 compliant compiler with OpenMP support



```bash
git clone https://github.com/calad0i/alkaid.git
pip install ./alkaid
```

For an editable development install, disable build isolation so meson-python can rebuild the in-place extension when it is imported:

```bash
pip install --no-build-isolation -e '.[tests]'
```

Documentation dependencies are available through the `docs` extra:

```bash
pip install 'alkaid[docs]'
```

The XLS backend imports `xls-python`. It is included in the test extra but should be installed separately for normal XLS use:

```bash
pip install xls-python
```

Alternatively, you can configure and build alkaid with meson directly:

```bash
meson setup build/cp31*
meson compile -C build/cp31*
```

```{warning}
If you are building an editable installation, `--no-build-isolation` must be used. Editable installation with meson-python relies on dynamic build hooks to recompile the C++ extension in place; build isolation breaks those hooks and can cause `ninja` to fail when alkaid is imported. Install `meson-python` in the active environment before running the editable pip install.
```
