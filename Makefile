.SUFFIXES:
TARGET = demo
CC = g++
CUCC = nvcc

CFLAGS = -std=c++11 -Wall
CUFLAGS = -m64 -arch=sm_35 -ccbin $(CC)
LDFLAGS = $(CFLAGS)

ifdef DEBUG
CFLAGS += -g -O0 -DDEBUG
CUFLAGS += -lineinfo
LDFLAGS += -g -O0
else
CFLAGS += -O3 #-DNDEBUG
CUFLAGS +=
LDFLAGS += -O3
endif

ifdef VERBOSE
AT =
else
AT = @
endif

AR = ar cr
ECHO = @echo
SHELL = /bin/sh

define concat
  $1$2$3$4$5$6$7$8
endef

#$(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(AT)$(CC) -MM -MF $3 -MP -MT $2 $(CFLAGS) $1
endef

define make-depend-cu
  $(AT)$(CUCC) -M $(CUFLAGS) $1 > $3.$$$$; \
  sed 's,.*\.o[ :]*,$2 : ,g' < $3.$$$$ > $3; \
  rm -f $3.$$$$
endef

# SRCS_C = sqdtrt.cpp trtUtil.cpp common.cpp tensorUtil.cu errorHandle.cu sdt_alloc.c
SRCS = *.cpp *.c *.cu
OUTDIR = .
OBJDIR = $(call concat,$(OUTDIR),/obj)
OBJS   = $(patsubst %.c, $(OBJDIR)/%.o, $(wildcard *.c))
OBJS  += $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard *.cpp))
CUOBJS = $(patsubst %.cu, $(OBJDIR)/%.o, $(wildcard *.cu))

TRIPLE?=x86_64-linux
CUDA_INSTALL_DIR = /usr/local/cuda-9.0
CUDA_LIBDIR = lib
TENSORRT_INSTALL_DIR = /home/ubuntu/TensorRT-4.0.0.3

INCPATHS    =-I"$(CUDA_INSTALL_DIR)/include" -I"/usr/local/include" -I"$(TENSORRT_INSTALL_DIR)/include" -I"/usr/local/include/opencv"
LIBPATHS    =-L"$(CUDA_INSTALL_DIR)/targets/$(TRIPLE)/$(CUDA_LIBDIR)" -L"/usr/local/lib" -L"$(CUDA_INSTALL_DIR)/lib64" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(TENSORRT_INSTALL_DIR)/lib"
LIBS = $(LIBPATHS) -lcudart -lcudart_static -lnvinfer `pkg-config --libs opencv` -lnvcaffe_parser -lnvinfer_plugin -lnvparsers
CFLAGS += $(INCPATHS) `pkg-config --cflags opencv`
CUFLAGS += $(INCPATHS) `pkg-config --cflags opencv`
LDFLAGS += $(LIBS)

.PHONY: all
all: $(OUTDIR)/$(TARGET)

$(OUTDIR)/$(TARGET): $(OBJS) $(CUOBJS) $(TESTOBJS)
	$(ECHO) Linking: $^
	$(AT)$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend-cu,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d
endif
