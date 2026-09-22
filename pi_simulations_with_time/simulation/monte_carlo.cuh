#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

#include <curand_kernel.h>

// Each thread runs one "simulation": it draws `points_per_thread`
// (x, y) points uniformly from [-1, 1] x [-1, 1] and counts how many
// land inside the unit circle (d_hits[idx]). It also timestamps its
// own start/end using the GPU's global nanosecond timer (%globaltimer),
// which is a single monotonic clock shared by every SM on the device.
// Because it's one shared clock, comparing d_start_ns[i] across threads
// tells you which threads were truly running at the same wall-clock
// instant -- that's what lets the host measure real concurrency
// afterwards, instead of just assuming it.
__global__ void monte_carlo_kernel(
    curandStatePhilox4_32_10_t *states,
    int points_per_thread,
    int total_threads,
    int *d_hits,
    unsigned long long *d_start_ns,
    unsigned long long *d_end_ns
);

#endif
