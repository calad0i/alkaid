project('alkaid', 'cpp',
  version: run_command('python3', 'tools/gitversion.py', check: true).stdout().strip(),
  meson_version: '>=1.9.0',
  default_options: ['buildtype=release', 'cpp_std=c++20'],
  license: 'LGPL-3.0-only',
)

py = import('python').find_installation(pure: false)

vcs_tag(
  command: ['python3', 'tools/gitversion.py'],
  input: 'src/alkaid/_version.py.in',
  output: '_version.py',
  install_dir: py.get_install_dir() / 'alkaid',
  replace_string: '@VCS_TAG@',
  install: true,
)

nanobind_dep = dependency('nanobind')
omp_dep = dependency('openmp')
xtensor_dep = dependency('xtensor', default_options: ['use_xsimd=true', 'use_tbb=false', 'use_openmp=true'])
nlohmann_json_dep = dependency('nlohmann_json')
zlib_dep = dependency('zlib')

_limited_abi = ''

alir_bin = py.extension_module(
  'alir_bin',
  sources: [
    'src/alkaid/_binary/alir/bindings.cc',
    'src/alkaid/_binary/alir/ALIRInterpreter_load.cc',
    'src/alkaid/_binary/alir/ALIRInterpreter_load_json.cc',
    'src/alkaid/_binary/alir/ALIRInterpreter_transpile.cc',
    'src/alkaid/_binary/alir/ALIRInterpreter_dispatch.cc',
  ],
  dependencies: [nanobind_dep, omp_dep, nlohmann_json_dep, zlib_dep],
  install: true,
  subdir: 'alkaid/_binary',
)

cmvm_bin = py.extension_module(
  'cmvm_bin',
  sources: [
    'src/alkaid/_binary/cmvm/api.cc',
    'src/alkaid/_binary/cmvm/bindings.cc',
    'src/alkaid/_binary/cmvm/bit_decompose.cc',
    'src/alkaid/_binary/cmvm/cmvm_core.cc',
    'src/alkaid/_binary/cmvm/indexers.cc',
    'src/alkaid/_binary/cmvm/mat_decompose.cc',
    'src/alkaid/_binary/cmvm/scm.cc',
    'src/alkaid/_binary/cmvm/state_opr.cc',
  ],
  dependencies: [nanobind_dep, xtensor_dep, omp_dep],
  install: true,
  subdir: 'alkaid/_binary',
  limited_api: _limited_abi,
)

install_subdir('src/alkaid',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    '_binary',
    '**/__pycache__',
    'codegen/hls/source/ap_types',
    'codegen/hls/source/ac_types'
  ],
  exclude_files: ['_version.py.in']
)

install_data('LICENSE', install_dir: py.get_install_dir() / 'alkaid')

install_data(
  'src/alkaid/_binary/__init__.py',
  install_dir: py.get_install_dir() / 'alkaid' / '_binary',
)

install_subdir('src/alkaid/codegen/hls/source/ac_types/include',
  install_dir: py.get_install_dir() / 'alkaid/codegen/hls/source/ac_types',
)

install_subdir('src/alkaid/codegen/hls/source/ap_types/include',
  install_dir: py.get_install_dir() / 'alkaid/codegen/hls/source/ap_types',
)

stubgen = nanobind_dep.get_variable('stubgen')

custom_target(
  output: 'alir_bin.pyi',
  depends: alir_bin,
  command: [py, stubgen, '-m', 'alir_bin', '-M', 'py.typed'],
  install: true,
  install_dir: py.get_install_dir() / 'alkaid' / '_binary',
)

custom_target(
  output: 'cmvm_bin.pyi',
  depends: cmvm_bin,
  command: [py, stubgen, '-m', 'cmvm_bin', '-M', 'py.typed'],
  install: true,
  install_dir: py.get_install_dir() / 'alkaid' / '_binary',
)
