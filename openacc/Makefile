CC       = pgcc
CFLAGS   = -O4 -Mvect=simd:256 -acc -ta=tesla:cc35 -Minfo=accel -MMD
CPPFLAGS = -I.
LDFLAGS  = -L. -L$(MKLROOT)/lib/intel64 -L$(MKLROOT)/../compiler/lib/intel64
LIBS     = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt
SRCS     = accmm.c
OBJS     = $(SRCS:%.c=%.o)
NV_OBJS  = $(NV_SRCS:%.cu=%.o)
DEPS     = $(SRCS:%.c=%.d)
BIN      = accmm

.SUFFIXES: .c .o

.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $<

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) $^ -o $@

ALL: $(BIN)

-include $(DEPS)

clean:
	rm -f $(BIN) $(OBJS) $(DEPS) *~ C
