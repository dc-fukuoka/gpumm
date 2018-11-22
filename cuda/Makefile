NVCC        = nvcc
CXX         = icpc
CXXFLAGS    = -g -O3 -march=core-avx2 -MMD -MP -Wall
NV_CXXFLAGS = -g -O3 -arch sm_60
CPPFLAGS    = -I.
CPPFLAGS   +=  -D_USE_SM
LDFLAGS     = -L. -L$(MKLROOT)/lib/intel64
LIBS        = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt
SRCS        = dclock.cxx gen_rand.cxx
NV_SRCS     = mydgemm.cu cumm.cu reduce_dsum.cu
OBJS        = $(SRCS:%.cxx=%.o)
NV_OBJS     = $(NV_SRCS:%.cu=%.o)
DEPS        = $(SRCS:%.cxx=%.d)
BIN         = cumm

.SUFFIXES: .cxx .cu .o

.cxx.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<
.cu.o:
	$(NVCC) $(CPPFLAGS) $(NV_CXXFLAGS) -c $<

$(BIN): $(NV_OBJS) $(OBJS)
	$(NVCC) $(LDFLAGS) $(LIBS) $^ -o $@

ALL: $(BIN)

-include $(DEPS)

clean:
	rm -f $(BIN) $(OBJS) $(NV_OBJS) $(DEPS) *~ C.cuda
