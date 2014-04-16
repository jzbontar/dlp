PREFIX=/home/jure/build/torch7
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lcublas -lluaT -lTHC -lTH

OBJ = dlp.o

%.o : %.cu
	nvcc -arch sm_35 --compiler-options '-fPIC' -c $(CFLAGS) $<

libdlp.so: ${OBJ}
	nvcc -o libdlp.so --shared ${OBJ} $(LDFLAGS)
