#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <limits>

#define M_PI_CONST 3.14159265358979323846

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct ThreadResult {
    int thread_id;
    int block_id;
    int thread_index;
    int darts;
    int inside_circle;
    double estimated_pi;
    double actual_pi;
    double percentage_error;
    unsigned long long elapsed_cycles;
    double thread_time;
};

inline void generateCSV(const std::vector<ThreadResult>& results, double overall_kernel_time, const std::string& filename = "data.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }
    file << "Thread_ID,Block_ID,Thread_Index,Darts,Inside_Circle,Estimated_Pi,Actual_Pi,Percentage_Error,Elapsed_Cycles,Thread_Time,Overall_Kernel_Time\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        file << res.thread_id << "," << res.block_id << "," << res.thread_index << "," << res.darts << ","
             << res.inside_circle << "," << std::setprecision(15) << res.estimated_pi << ","
             << std::setprecision(15) << res.actual_pi << "," << std::setprecision(15) << res.percentage_error << ","
             << res.elapsed_cycles << "," << std::setprecision(6) << res.thread_time << ",";
        file << std::fixed << std::setprecision(4) << overall_kernel_time << "\n";
    }
    file.close();
}

inline void printSummary(const std::vector<ThreadResult>& results, double overall_kernel_time, int blocks, int threads_per_block, int total_threads, const std::string& output_filename = "data.csv") {
    double sum_estimated_pi = 0, sum_error = 0, max_error = 0, max_pi = 0;
    double min_error = std::numeric_limits<double>::max();
    double min_pi = std::numeric_limits<double>::max();
    
    for(const auto& r : results) {
        sum_estimated_pi += r.estimated_pi;
        sum_error += r.percentage_error;
        if(r.percentage_error < min_error) min_error = r.percentage_error;
        if(r.percentage_error > max_error) max_error = r.percentage_error;
        if(r.estimated_pi < min_pi) min_pi = r.estimated_pi;
        if(r.estimated_pi > max_pi) max_pi = r.estimated_pi;
    }
    
    std::cout << "==========================================\n";
    std::cout << "   CUDA MONTE CARLO PI ESTIMATION\n";
    std::cout << "==========================================\n\n";
    std::cout << "Blocks              : " << blocks << "\n";
    std::cout << "Threads per Block   : " << threads_per_block << "\n";
    std::cout << "Total Threads       : " << total_threads << "\n";
    std::cout << "Darts per Thread    : 50000\n\n";
    std::cout << "Actual Pi           : " << std::setprecision(15) << M_PI_CONST << "\n\n";
    std::cout << "Overall Kernel Time : " << std::fixed << std::setprecision(4) << overall_kernel_time << " ms\n\n";
    std::cout << "Average Estimated Pi: " << std::setprecision(12) << (sum_estimated_pi / results.size()) << "\n";
    std::cout << "Average Error       : " << std::setprecision(4) << (sum_error / results.size()) << " %\n\n";
    std::cout << "Minimum Error       : " << min_error << " %\n";
    std::cout << "Maximum Error       : " << max_error << " %\n\n";
    std::cout << "Results saved to:\n" << output_filename << "\n";
}
#endif