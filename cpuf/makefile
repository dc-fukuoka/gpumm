FC       = ifort
FCFLAGS  = -g -O3 -fopenmp -march=core-avx2 -MMD
FPPFLAGS = -I.
LDFLAGS  = -L. -L$(MKLROOT)/lib/intel64
LIBS     = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
SRCS     = mm.F90 $(MKLROOT)/include/mkl_vsl.f90
OBJS     = mm.o
MODS     = mkl_vsl.mod mkl_vsl_type.mod
DEPS     = $(OBJS:%.o=%.d)
BIN      = mmf

ALL: $(BIN)

$(OBJS): mm.F90 $(MODS)
	$(FC) $(FPPFLAGS) $(FCFLAGS) -c $<

$(MODS): $(MKLROOT)/include/mkl_vsl.f90
	$(FC) $(FPPFLAGS) $(FCFLAGS) -c $<

$(BIN): $(OBJS)
	$(FC) $(LDFLAGS) $(LIBS) $^ -o $@

-include $(DEPS)

clean:
	rm -f $(BIN) *.o *.mod *.d *~ C_f
