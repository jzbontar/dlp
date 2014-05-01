PREFIX=/home/tom/torch-sermanet/installed
CFLAGS=-I$(PREFIX)/include/torch/THC -I$(PREFIX)/include/torch/TH -I$(PREFIX)/include
LDFLAGS=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH

OBJ = dlp.o

%.o : %.cu
	nvcc -arch sm_35 --compiler-options '-fPIC' -c $(CFLAGS) $<

libdlp.so: ${OBJ}
	nvcc -o libdlp.so --shared ${OBJ} $(LDFLAGS)
