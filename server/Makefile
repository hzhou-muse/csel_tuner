# Build the benchmark server with the MPICH under test:
#   make MPICC=$MPICH_PREFIX/bin/mpicc
MPICC ?= mpicc
CFLAGS ?= -O2 -Wall

mpisrv: mpisrv.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f mpisrv
