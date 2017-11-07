CXX         = icpc
CXXFLAGS    = -g -O3 -march=core-avx2 -MMD -MP -Wall
CPPFLAGS    = -I. -I$(CUDA_HOME)/include
LDFLAGS     = -L. -L$(MKLROOT)/lib/intel64 -L$(CUDA_HOME)/lib64
LIBS        = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lirc -lcublas -lcudart
SRCS        = dclock.cxx gen_rand.cxx cublasmm.cxx cublasdgemm.cxx
OBJS        = $(SRCS:%.cxx=%.o)
DEPS        = $(SRCS:%.cxx=%.d)
BIN         = cublasmm

.SUFFIXES: .cxx .o

.cxx.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) $(LIBS) $^ -o $@

ALL: $(BIN)

-include $(DEPS)

clean:
	rm -f $(BIN) $(OBJS) $(NV_OBJS) $(DEPS) *~
