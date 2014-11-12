run: raycast
	./raycast

run-mem: raycast-mem
	cuda-memcheck ./raycast > memcheck
	
raycast : bmp.cpp raycast.cu
	nvcc bmp.cpp raycast.cu -arch=sm_20 -lcudart -lm -o raycast

raycast-mem: bmp.cpp raycast.cu
	nvcc bmp.cpp raycast.cu -G -g -arch=sm_20 -lcudart -lm -o raycast
