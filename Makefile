CC       = cc
CXX		 = c++

LIBTORCH_DIR ?= thirdparty/torch/libtorch
CPPFLAGS += -I src/ \
			-I $(LIBTORCH_DIR)/include/torch/csrc/api/include \
			-I $(LIBTORCH_DIR)/include -I thirdparty/
CFLAGS	+= 	-g -Wall -O2
CXXFLAGS   += -g -Wall -O2  -std=c++17
LIBS    +=  -Wl,-rpath,'$$ORIGIN/$(LIBTORCH_DIR)/lib' -Wl,-rpath,'$$ORIGIN/../lib' \
			-Wl,-rpath,$(LIBTORCH_DIR)/lib \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cpu.so"  \
			-Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libtorch.so"  \
			-Wl,--as-needed $(LIBTORCH_DIR)/lib/libc10.so
LDFLAGS  += $(LIBS) -lz -lm -lpthread
BUILD_DIR = build

# # https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
# ifeq ($(cxx11_abi),) #  cxx11_abi not defined
# CXXFLAGS		+= -D_GLIBCXX_USE_CXX11_ABI=0
# endif

# change the tool name to what you want
BINARY = flash_test

OBJ = $(BUILD_DIR)/main.o \

# add more objects here if needed

VERSION = `git describe --tags`

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
    CPPFLAGS += -DUSE_GPU=1 -DUSE_CUDA=1
	CUDA_ROOT ?= /usr/local/cuda
	CUDA_LIB ?= $(CUDA_ROOT)/lib64
	CUDA_INC ?= $(CUDA_ROOT)/include
	CPPFLAGS += -I $(CUDA_INC)
	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_cuda.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_cuda.so"
	LDFLAGS += -L$(CUDA_LIB) -lcudart_static -lrt -ldl
else ifdef rocm
	CPPFLAGS += -DUSE_GPU=1 -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__
	ROCM_ROOT ?= /opt/rocm
	ROCM_INC ?= $(ROCM_ROOT)/include
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	CPPFLAGS += -I $(ROCM_INC)
	LIBS += -Wl,--as-needed -lpthread -Wl,--no-as-needed,"$(LIBTORCH_DIR)/lib/libtorch_hip.so" -Wl,--as-needed,"$(LIBTORCH_DIR)/lib/libc10_hip.so"
	LDFLAGS += -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
endif

.PHONY: clean distclean test

# flash_test
$(BINARY): $(OBJ)
	$(CXX) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*.o

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BUILD_DIR)/* autom4te.cache
