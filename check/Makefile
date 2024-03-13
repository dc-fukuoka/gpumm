CC       = icx
CFLAGS   = -g -O3 -fopenmp -march=core-avx2 -MMD -MP -Wall
CPPFLAGS = -I.
LDFLAGS  = -L.
LIBS     = -liomp5
SRCS     = check.c
OBJS     = $(SRCS:%.c=%.o)
NV_OBJS  = $(NV_SRCS:%.cu=%.o)
DEPS     = $(SRCS:%.c=%.d)
BIN      = check

.SUFFIXES: .c .o

.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $<

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) $(LIBS) $^ -o $@

ALL: $(BIN)

-include $(DEPS)

clean:
	rm -f $(BIN) $(OBJS) $(DEPS) *~
