IDIR = ./include
ODIR = ./object
SDIR = ./source

CC = nvcc -arch=sm_70
CFLAGS = -I $(IDIR)

_DEPS = const.h funclib.h funclib.cuh
DEPS  = $(patsubst %, $(IDIR)/%, $(_DEPS))

_OBJ = file_write.o lattice.o linear_interp.o init.o prof_gen.o integrator.o main.o
OBJ = $(patsubst %, $(ODIR)/%, $(_OBJ))

all: iri3d

iri3d: $(OBJ)
	$(CC) -o $@ $^

$(ODIR)/main.o: $(SDIR)/main.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/init.o: $(SDIR)/init.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/integrator.o: $(SDIR)/integrator.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/lattice.o: $(SDIR)/lattice.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/linear_interp.o: $(SDIR)/linear_interp.cu $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)
 
$(ODIR)/file_write.o: $(SDIR)/file_write.cpp $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

$(ODIR)/prof_gen.o: $(SDIR)/prof_gen.cpp $(DEPS)
	$(CC) --device-c -o $@ $< $(CFLAGS)

clean:
	rm iri3d outputs/* $(OBJ)

