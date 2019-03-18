LINK = nvcc

OUT_FILE = edge

FLAGS = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11

all: edge

edge: sobelFilter.cu lodepng.cpp
	$(LINK) -o $(OUT_FILE) $(FLAGS) $^

clean: 
	rm -f *.o *~ core
