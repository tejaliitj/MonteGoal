# =============================================================================
# Makefile for the Monte Carlo Engine
# -----------------------------------------------------------------------------
# Build:  make
# Run Pi RNG test:       ./montecarlo_engine pi 1048576 1000 42
# Run Poisson kernel:    ./montecarlo_engine poisson 1048576 1000 42
# Clean:  make clean
# =============================================================================
NVCC        := nvcc
TARGET      := montecarlo_engine
SRC         := src/main.cu
NVCC_FLAGS  := -O3 -std=c++17 -arch=sm_70 -lcurand

.PHONY: all clean run_pi run_poisson

all: $(TARGET)

$(TARGET): $(SRC) include/rng/rng_engine.cuh \
           include/distributions/uniform_dist.cuh \
           include/distributions/poisson_dist.cuh \
           include/kernels/kernel_interface.cuh \
           include/kernels/pi_estimation_kernel.cuh \
           include/kernels/poisson_sim_kernel.cuh \
           include/csv_writer.hpp
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

run_pi: $(TARGET)
	./$(TARGET) pi 1048576 1000 42

run_poisson: $(TARGET)
	./$(TARGET) poisson 1048576 1000 42

clean:
	rm -f $(TARGET)
	rm -f data/montecarlo_features*.csv
