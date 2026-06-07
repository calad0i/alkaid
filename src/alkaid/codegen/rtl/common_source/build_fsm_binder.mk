default: slow

VERILATOR_ROOT = $(shell verilator -V | grep -a VERILATOR_ROOT | tail -1 | awk '{{print $$3}}')
INCLUDES = -I./obj_dir -I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd -I. -I../src -I../src/static
WARNINGS = -Wl,--no-undefined
CFLAGS = -std=c++17 -fPIC
LINKFLAGS = $(INCLUDES) $(WARNINGS)
LIBNAME = lib$(VM_PREFIX)_fsm_$(STAMP).so
N_JOBS ?= $(shell nproc)
VERILATOR_FLAGS ?= -Wall
TOP_MODULE ?= $(VM_PREFIX)
VMOD_PREFIX ?= $(TOP_MODULE)
VERILOG_SOURCES = $(wildcard ../src/static/*.v) $(wildcard ../src/*.v)

./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a: $(VERILOG_SOURCES)
	cp ../src/memfiles/* ./ 2>/dev/null || true
	verilator --cc -j $(N_JOBS) -build --top-module $(TOP_MODULE) --prefix V$(VMOD_PREFIX) $(VERILATOR_FLAGS) $(VERILOG_SOURCES) -CFLAGS "$(CFLAGS)" -I../src -I../src/static

$(LIBNAME): ./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a fsm_binder.cc fsm_config.hh fsm_wrapper.hh ioutil.hh
	$(CXX) $(CFLAGS) $(LINKFLAGS) $(CXXFLAGS2) -pthread -shared -o $(LIBNAME) fsm_binder.cc ./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a

fast: CFLAGS += -O3
fast: $(LIBNAME)

slow: CFLAGS += -O
slow: $(LIBNAME)

clean:
	rm -rf obj_dir
	rm -f $(LIBNAME)
