NVCC        = nvcc
NVCC_FLAGS  = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11
EXE         = edge

default: $(EXE)

edge: edge_detection.cu lodepng.cpp
	$(NVCC) -o $(EXE) $(NVCC_FLAGS) $^

clean: 
	rm -f *.o *~ core $(EXE)
