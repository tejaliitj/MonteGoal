# Monte Carlo Engine

CUDA Monte Carlo engine using cuRAND Philox.

## Structure

```text
montecarlo_engine/
├── include/
│   ├── rng/
│   │   └── rng_engine.cuh
│   ├── distributions/
│   │   ├── uniform_dist.cuh
│   │   └── poisson_dist.cuh
│   ├── kernels/
│   │   ├── kernel_interface.cuh
│   │   ├── pi_estimation_kernel.cuh
│   │   └── poisson_sim_kernel.cuh
│   └── csv_writer.hpp
├── src/
│   ├── rng/
│   │   └── rng_engine.cu
│   ├── distributions/
│   │   ├── uniform_dist.cu
│   │   └── poisson_dist.cu
│   ├── kernels/
│   │   ├── pi_estimation_kernel.cu
│   │   └── poisson_sim_kernel.cu
│   ├── csv_writer.cpp
│   └── main.cu
├── data/
├── CMakeLists.txt
└── README.md
```

`main.cu` contains orchestration only. CUDA kernels, RNG setup, and distributions live in their own files.

The same fresh master seed is used for both demos in one program run, while each GPU thread gets a unique Philox sequence using its thread index.
