default: slow

VERILATOR_ROOT = $(shell verilator -V | grep -a VERILATOR_ROOT | tail -1 | awk '{{print $$3}}')
INCLUDES = -I./obj_dir -I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd -I. -I../src -I../src/static
WARNINGS = -Wl,--no-undefined
CFLAGS = -std=c++17 -fPIC
LINKFLAGS = $(INCLUDES) $(WARNINGS)
LIBNAME = lib$(VM_PREFIX)_fsm_$(STAMP).so
N_JOBS ?= $(shell nproc)
VERILATOR_FLAGS ?= -Wall
SOURCE_TYPE ?= verilog
TOP_MODULE ?= $(VM_PREFIX)
VMOD_PREFIX ?= $(TOP_MODULE)
VERILOG_SOURCES = $(wildcard ../src/static/*.v) $(wildcard ../src/*.v)
VHDL_STATIC_SOURCES = $(wildcard ../src/static/*.vhd)
VHDL_WRAPPER_SOURCE = ../src/$(TOP_MODULE).vhd
VHDL_FSM_SOURCE = ../src/$(TOP_MODULE:_wrapper=).vhd
VHDL_TOP_SOURCES = $(VHDL_FSM_SOURCE) $(filter-out $(VHDL_FSM_SOURCE),$(VHDL_WRAPPER_SOURCE))
VHDL_TABLE_SOURCES = $(wildcard ../src/*_tables.vhd)
VHDL_COMB_SOURCES = $(filter-out $(VHDL_WRAPPER_SOURCE) $(VHDL_FSM_SOURCE) $(VHDL_TABLE_SOURCES),$(wildcard ../src/*.vhd))
VHDL_SYNTH_SOURCE = $(TOP_MODULE).v

ifneq ($(filter $(SOURCE_TYPE),verilog vhdl),$(SOURCE_TYPE))
$(error SOURCE_TYPE must be either 'verilog' or 'vhdl')
endif

ifeq ($(SOURCE_TYPE),vhdl)
VERILATOR_SOURCES = $(VHDL_SYNTH_SOURCE)
else
VERILATOR_SOURCES = $(VERILOG_SOURCES)
endif

$(VHDL_SYNTH_SOURCE): $(VHDL_STATIC_SOURCES) $(VHDL_TABLE_SOURCES) $(VHDL_COMB_SOURCES) $(VHDL_TOP_SOURCES)
	mkdir -p obj_dir
	ghdl -a --std=08 --workdir=obj_dir $(VHDL_STATIC_SOURCES) $(VHDL_TABLE_SOURCES) $(VHDL_COMB_SOURCES) $(VHDL_TOP_SOURCES)
	ghdl synth --std=08 --workdir=obj_dir --out=verilog $(TOP_MODULE) > $(VHDL_SYNTH_SOURCE)

./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a: $(VERILATOR_SOURCES)
	verilator --cc -j $(N_JOBS) -build --top-module $(TOP_MODULE) --prefix V$(VMOD_PREFIX) $(VERILATOR_FLAGS) $(VERILATOR_SOURCES) -CFLAGS "$(CFLAGS)" -I../src -I../src/static

$(LIBNAME): ./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a fsm_binder.cc fsm_config.hh fsm_wrapper.hh ioutil.hh
	$(CXX) $(CFLAGS) $(LINKFLAGS) $(CXXFLAGS2) -pthread -shared -o $(LIBNAME) fsm_binder.cc ./obj_dir/libV$(VMOD_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VMOD_PREFIX)__ALL.a $(EXTRA_CXXFLAGS)

fast: CFLAGS += -O3
fast: $(LIBNAME)

slow: CFLAGS += -O
slow: $(LIBNAME)

clean:
	rm -rf obj_dir
	rm -f $(LIBNAME)
	rm -f $(VHDL_SYNTH_SOURCE)
