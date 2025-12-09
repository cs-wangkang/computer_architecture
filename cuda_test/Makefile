# Makefile for CUDA matrix multiplication

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++11
TARGET = matrix_multiply
SOURCE = matrix_multiply.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

