// =============================================================================
// kernel_interface.cuh
// -----------------------------------------------------------------------------
// This is what makes the engine's kernels "swappable" (your note: "Change
// kernel for different applications"). Every application kernel exposes the
// SAME two functions with the SAME signature:
//
//   void launch(const RNGEngine& rng, int trials_per_thread, float* d_out);
//   const char* name();
//
// main.cu doesn't care *which* application it's running -- it just calls
// whichever ApplicationKernel it was told to via the command line. Adding a
// new simulation later means writing one new .cuh file here and registering
// it in main.cu; nothing else in the engine changes.
// =============================================================================
#pragma once
#include "../rng/rng_engine.cuh"

// Per-simulation result produced by each GPU thread
struct SimulationResult {
    float estimate;              // Thread's Monte Carlo estimate
    float error;                 // Absolute error compared to ground truth
    unsigned long long start_ns; // Nanosecond timestamp when simulation started on GPU
    unsigned long long end_ns;   // Nanosecond timestamp when simulation ended on GPU
    float time_us;               // Execution duration in microseconds
};

__device__ inline unsigned long long get_globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

class ApplicationKernel {
public:
    virtual ~ApplicationKernel() = default;

    // Runs 1 simulation per thread. d_out is a pre-allocated device buffer sized by
    // the caller according to out_size().
    virtual void launch(const RNGEngine& rng, int trials_per_thread,
                         SimulationResult* d_out) const = 0;

    // Number of SimulationResult elements written into d_out
    virtual int out_size(const RNGEngine& rng) const = 0;

    // Column header(s) written to the output CSV.
    virtual const char* csv_header() const = 0;

    virtual const char* name() const = 0;

    // Ground truth value for evaluating error
    virtual double ground_truth() const = 0;
};
