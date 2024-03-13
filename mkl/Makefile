CC       = icx
CFLAGS   = -g -O3 -fopenmp -march=core-avx2 -MMD -MP -Wall
CPPFLAGS = -I.
LDFLAGS  = -L. -L$(MKLROOT)/lib/intel64
LIBS     = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt
SRCS     = dgemm.c
OBJS     = $(SRCS:%.c=%.o)
NV_OBJS  = $(NV_SRCS:%.cu=%.o)
DEPS     = $(SRCS:%.c=%.d)
BIN      = dgemm

.SUFFIXES: .c .o

.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $<

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) $(LIBS) $^ -o $@

ALL: $(BIN)

-include $(DEPS)

clean:
	rm -f $(BIN) $(OBJS) $(DEPS) *~ C.mkl
