===========================================
Distributed Arithmetic for Machine Learning
===========================================

.. .. image:: https://img.shields.io/badge/License-LGPLv3-blue.svg
..    :target: https://www.gnu.org/licenses/lgpl-3.0.en.html
.. image:: https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/unit-test.yml?label=test
   :target: https://github.com/calad0i/alkaid/actions/workflows/unit-test.yml
.. image:: https://img.shields.io/github/actions/workflow/status/calad0i/alkaid/sphinx-build.yml?label=doc
   :target: https://calad0i.github.io/alkaid/
.. image:: https://img.shields.io/pypi/v/alkaid
   :target: https://pypi.org/project/alkaid/
.. image:: https://img.shields.io/badge/arXiv-2507.04535-b31b1b.svg
   :target: https://arxiv.org/abs/2507.04535
.. image:: https://img.shields.io/codecov/c/github/calad0i/alkaid
   :target: https://codecov.io/gh/calad0i/alkaid


.. rst-class:: light
.. image:: _static/example.svg
   :alt: alkaid-overview
   :width: 600

alkaid is a lightweight compiler for generating low-latency, static-dataflow kernels for FPGAs. It traces quantized arithmetic graph into ALIR, applies distributed-arithmetic optimization through CMVM where useful, and emits RTL or HLS projects.

As a static-dataflow compiler, alkaid is specialized for kernels that are equivalent to combinational logic or an initiation-interval-one pipeline. The generated kernels are intended to be building blocks that users can compose into larger designs when resource sharing or time multiplexing is required.

With DA in its name, alkaid performs distributed-arithmetic (DA) optimization to generate efficient kernels for linear DSP operations. The algorithm is described in the `TRETS'25 paper <https://doi.org/10.1145/3777387>`_. With DA optimization, linear DSP operations can be implemented with adders and lookup tables instead of hardened multipliers; users can also exclude selected multiplication pairs from DA optimization.


Installation
------------

.. code-block:: bash

   pip install alkaid

Binary wheels are published for Linux x86_64 and macOS ARM64. Building from source requires Python 3.10 or newer, NumPy, meson-python, and a C++20 compiler with OpenMP support.


Index
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   status.md
   install.md
   getting_started.md
   alir.md
   cmvm.md
   plugin.md
   faq.md

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   autodoc/alkaid

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
