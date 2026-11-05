# Build all four examples:      make
# Pick the GPU architecture:    make ARCH=-arch=sm_75   (T4: Colab, Kaggle)
#                               make ARCH=-arch=sm_80   (A100: LSU SuperMike-III)
# -arch=native (the default) asks the GPU in this machine, so it needs a GPU at
# build time. On a login node without a GPU, pass ARCH explicitly.

NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS ?= -O3 -std=c++17 -Xcompiler -O3,-march=native

PROGRAMS = vector_add matmul image_blur reduction

all: $(PROGRAMS)

vector_add: vector_add.cu common.h
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $<

matmul: matmul.cu common.h
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $< -lcublas

image_blur: image_blur.cu common.h
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $<

reduction: reduction.cu common.h
	$(NVCC) $(FLAGS) $(ARCH) -o $@ $<

# Run every benchmark and collect the CSV rows in results.csv
bench: all
	./run_all.sh

clean:
	rm -f $(PROGRAMS) results.csv *.pgm

.PHONY: all bench clean
