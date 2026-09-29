// // #include <iostream>
// // #include <iomanip>
// // #include <cmath>

// // #include "rng/rng_engine.cuh"
// // #include "kernels/kernel_interface.cuh"
// // #include "kernels/pi_estimation_kernel.cuh"
// // #include "kernels/poisson_sim_kernel.cuh"

// // int main()
// // {
// //     constexpr int PI_SAMPLES = 10'000'000;
// //     constexpr int POISSON_SAMPLES = 10'000;
// //     constexpr double LAMBDA = 2.5;

// //     const unsigned long long seed = generate_seed();

// //     std::cout << "Monte Carlo Engine - Philox\n";
// //     std::cout << "===========================\n";
// //     std::cout << "Seed: " << seed << "\n\n";

// //     // =========================================================
// //     // 1. PI ESTIMATION
// //     // =========================================================
// //     {
// //         std::cout << "Monte Carlo Pi Estimation\n";
// //         std::cout << "Generating " << PI_SAMPLES << " points...\n";

// //         curandStatePhilox4_32_10_t* d_states = nullptr;
// //         unsigned long long* d_inside = nullptr;

// //         check_cuda(cudaMalloc(
// //             &d_states,
// //             PI_SAMPLES * sizeof(curandStatePhilox4_32_10_t)),
// //             "cudaMalloc(pi states)");

// //         check_cuda(cudaMalloc(
// //             &d_inside,
// //             sizeof(unsigned long long)),
// //             "cudaMalloc(pi counter)");

// //         check_cuda(cudaMemset(
// //             d_inside, 0, sizeof(unsigned long long)),
// //             "cudaMemset(pi counter)");

// //         setup_philox_states<<<blocks_for(PI_SAMPLES), THREADS_PER_BLOCK>>>(
// //             d_states, seed, PI_SAMPLES);

// //         check_cuda(cudaGetLastError(), "Pi RNG kernel launch");
// //         check_cuda(cudaDeviceSynchronize(), "Pi RNG kernel");

// //         estimate_pi_kernel<<<blocks_for(PI_SAMPLES), THREADS_PER_BLOCK>>>(
// //             d_states, d_inside, PI_SAMPLES);

// //         check_cuda(cudaGetLastError(), "Pi kernel launch");
// //         check_cuda(cudaDeviceSynchronize(), "Pi kernel");

// //         unsigned long long inside = 0;

// //         check_cuda(cudaMemcpy(
// //             &inside, d_inside, sizeof(inside),
// //             cudaMemcpyDeviceToHost),
// //             "cudaMemcpy(pi result)");

// //         double pi = 4.0 * static_cast<double>(inside) / PI_SAMPLES;
// //         constexpr double PI_TRUE = 3.14159265358979323846;
// //         double abs_error = std::abs(pi - PI_TRUE);
// //         double pct_error = (abs_error / PI_TRUE) * 100.0;

// //         std::cout << std::fixed << std::setprecision(12);
// //         std::cout << "Points inside    : " << inside << '\n';
// //         std::cout << "Estimated Pi     : " << pi << '\n';
// //         std::cout << "Actual Pi        : " << PI_TRUE << '\n';
// //         std::cout << "Absolute error   : " << abs_error << '\n';
// //         std::cout << "Percentage error : " << pct_error << "%\n\n";

// //         cudaFree(d_states);
// //         cudaFree(d_inside);
// //     }

// //     // =========================================================
// //     // 2. POISSON SIMULATION
// //     // =========================================================
// //     {
// //         std::cout << "Poisson Simulation\n";
// //         std::cout << "Samples: " << POISSON_SAMPLES << '\n';
// //         std::cout << "Lambda : " << LAMBDA << '\n';

// //         curandStatePhilox4_32_10_t* d_states = nullptr;
// //         unsigned int* d_results = nullptr;

// //         check_cuda(cudaMalloc(
// //             &d_states,
// //             POISSON_SAMPLES * sizeof(curandStatePhilox4_32_10_t)),
// //             "cudaMalloc(poisson states)");

// //         check_cuda(cudaMalloc(
// //             &d_results,
// //             POISSON_SAMPLES * sizeof(unsigned int)),
// //             "cudaMalloc(poisson results)");

// //         setup_philox_states<<<blocks_for(POISSON_SAMPLES), THREADS_PER_BLOCK>>>(
// //             d_states, seed, POISSON_SAMPLES);

// //         check_cuda(cudaGetLastError(), "Poisson RNG kernel launch");
// //         check_cuda(cudaDeviceSynchronize(), "Poisson RNG kernel");

// //         generate_poisson_kernel<<<
// //             blocks_for(POISSON_SAMPLES), THREADS_PER_BLOCK>>>(
// //             d_states, d_results, LAMBDA, POISSON_SAMPLES);

// //         check_cuda(cudaGetLastError(), "Poisson kernel launch");
// //         check_cuda(cudaDeviceSynchronize(), "Poisson kernel");

// //         auto* h_results = new unsigned int[POISSON_SAMPLES];

// //         check_cuda(cudaMemcpy(
// //             h_results, d_results,
// //             POISSON_SAMPLES * sizeof(unsigned int),
// //             cudaMemcpyDeviceToHost),
// //             "cudaMemcpy(poisson results)");

// //         std::cout << "\nFirst 10 Poisson values:\n";
// //         for (int i = 0; i < 10; ++i)
// //             std::cout << h_results[i] << '\n';

// //         double sum = 0.0;
// //         for (int i = 0; i < POISSON_SAMPLES; ++i)
// //             sum += h_results[i];

// //         std::cout << "\nSample mean   : "
// //                   << sum / POISSON_SAMPLES << '\n';
// //         std::cout << "Expected mean : " << LAMBDA << '\n';

// //         delete[] h_results;
// //         cudaFree(d_states);
// //         cudaFree(d_results);
// //     }

// //     return 0;
// // }



// #include "wc2026_io.hpp"
// #include "kernels/wc2026_sim_kernel.cuh"

// #include <cuda_runtime.h>

// #include <chrono>
// #include <iostream>
// #include <random>
// #include <vector>


// #define CUDA_CHECK(call)                                      \
// do {                                                          \
//     cudaError_t err = (call);                                 \
//     if (err != cudaSuccess) {                                 \
//         std::cerr                                             \
//             << "CUDA ERROR: "                                   \
//             << cudaGetErrorString(err)                        \
//             << " at " << __FILE__ << ":" << __LINE__          \
//             << "\n";                                          \
//         std::exit(EXIT_FAILURE);                              \
//     }                                                         \
// } while (0)


// int main(
//     int argc,
//     char** argv
// ) {
//     int N =
//         (argc > 1)
//         ? std::stoi(argv[1])
//         : 10000;


//     std::cout
//         << "World Cup CUDA Monte Carlo\n";

//     std::cout
//         << "Simulations: "
//         << N
//         << "\n";


//     // ============================================================
//     // LOAD CPU DATA
//     // ============================================================

//     WC2026Data data;

//     try {

//         data =
//             load_wc2026_data(
//                 "data/team_features.csv",
//                 "data/model_coefs.csv"
//             );

//     }
//     catch (const std::exception& e) {

//         std::cerr
//             << "FATAL: "
//             << e.what()
//             << "\n";

//         return 1;
//     }


//     // ============================================================
//     // FRESH MASTER SEED
//     // ============================================================

//     std::random_device rd;

//     unsigned long long seed =
//         (static_cast<unsigned long long>(rd()) << 32)
//         ^ static_cast<unsigned long long>(rd());


//     std::cout
//         << "Master seed: "
//         << seed
//         << "\n";


//     // ============================================================
//     // GPU MEMORY
//     // ============================================================

//     TeamGPU* d_teams = nullptr;
//     ModelGPU* d_model = nullptr;
//     FixtureGPU* d_fixtures = nullptr;

//     int* d_champion_counts = nullptr;
//     int* d_round_counts = nullptr;


//     CUDA_CHECK(
//         cudaMalloc(
//             &d_teams,
//             WC_NUM_TEAMS *
//             sizeof(TeamGPU)
//         )
//     );


//     CUDA_CHECK(
//         cudaMalloc(
//             &d_model,
//             sizeof(ModelGPU)
//         )
//     );


//     CUDA_CHECK(
//         cudaMalloc(
//             &d_fixtures,
//             WC_NUM_FIXTURES *
//             sizeof(FixtureGPU)
//         )
//     );


//     CUDA_CHECK(
//         cudaMalloc(
//             &d_champion_counts,
//             WC_NUM_TEAMS *
//             sizeof(int)
//         )
//     );


//     CUDA_CHECK(
//         cudaMalloc(
//             &d_round_counts,
//             WC_NUM_TEAMS *
//             WC_NUM_STAGES *
//             sizeof(int)
//         )
//     );


//     // ============================================================
//     // COPY INPUT DATA → GPU
//     // ============================================================

//     std::vector<TeamGPU> host_teams(
//         WC_NUM_TEAMS
//     );

//     for (int i = 0;
//          i < WC_NUM_TEAMS;
//          ++i) {

//         host_teams[i] =
//             data.teams[i].gpu;
//     }


//     CUDA_CHECK(
//         cudaMemcpy(
//             d_teams,
//             host_teams.data(),
//             WC_NUM_TEAMS * sizeof(TeamGPU),
//             cudaMemcpyHostToDevice
//         )
//     );


//     CUDA_CHECK(
//         cudaMemcpy(
//             d_model,
//             &data.model.gpu,
//             sizeof(ModelGPU),
//             cudaMemcpyHostToDevice
//         )
//     );


//     CUDA_CHECK(
//         cudaMemcpy(
//             d_fixtures,
//             data.fixtures.data(),
//             WC_NUM_FIXTURES *
//             sizeof(FixtureGPU),
//             cudaMemcpyHostToDevice
//         )
//     );


//     // ============================================================
//     // RESET OUTPUT COUNTERS
//     // ============================================================

//     CUDA_CHECK(
//         cudaMemset(
//             d_champion_counts,
//             0,
//             WC_NUM_TEAMS * sizeof(int)
//         )
//     );


//     CUDA_CHECK(
//         cudaMemset(
//             d_round_counts,
//             0,
//             WC_NUM_TEAMS *
//             WC_NUM_STAGES *
//             sizeof(int)
//         )
//     );


//     // ============================================================
//     // RUN SIMULATION
//     // ============================================================

//     std::cout
//         << "Launching GPU simulation...\n";


//     auto start =
//         std::chrono::high_resolution_clock::now();


//     launch_wc2026_simulation(
//         d_teams,
//         d_model,
//         d_fixtures,
//         N,
//         seed,
//         d_champion_counts,
//         d_round_counts
//     );


//     CUDA_CHECK(
//         cudaGetLastError()
//     );

//     CUDA_CHECK(
//         cudaDeviceSynchronize()
//     );


//     auto end =
//         std::chrono::high_resolution_clock::now();


//     double seconds =
//         std::chrono::duration<double>(
//             end - start
//         ).count();


//     std::cout
//         << "GPU simulation complete.\n";

//     std::cout
//         << "Runtime: "
//         << seconds
//         << " seconds\n";

//     std::cout
//         << "Simulations/sec: "
//         << N / seconds
//         << "\n";


//     // ============================================================
//     // COPY RESULTS GPU → CPU
//     // ============================================================

//     std::vector<int> champion_counts(
//         WC_NUM_TEAMS,
//         0
//     );

//     std::vector<int> round_counts(
//         WC_NUM_TEAMS *
//         WC_NUM_STAGES,
//         0
//     );


//     CUDA_CHECK(
//         cudaMemcpy(
//             champion_counts.data(),
//             d_champion_counts,
//             WC_NUM_TEAMS * sizeof(int),
//             cudaMemcpyDeviceToHost
//         )
//     );


//     CUDA_CHECK(
//         cudaMemcpy(
//             round_counts.data(),
//             d_round_counts,
//             WC_NUM_TEAMS *
//             WC_NUM_STAGES *
//             sizeof(int),
//             cudaMemcpyDeviceToHost
//         )
//     );


//     // ============================================================
//     // WRITE CSV
//     // ============================================================

//     write_wc2026_results(
//         data,
//         champion_counts,
//         round_counts,
//         N
//     );


//     // ============================================================
//     // CLEANUP
//     // ============================================================

//     CUDA_CHECK(cudaFree(d_teams));
//     CUDA_CHECK(cudaFree(d_model));
//     CUDA_CHECK(cudaFree(d_fixtures));
//     CUDA_CHECK(cudaFree(d_champion_counts));
//     CUDA_CHECK(cudaFree(d_round_counts));


//     std::cout
//         << "\nDone.\n";

//     std::cout
//         << "Wrote:\n"
//         << "  champion_probabilities_cuda.csv\n"
//         << "  round_reach_probabilities_cuda.csv\n";


//     return 0;
// }



#include "wc2026_io.hpp"
#include "kernels/wc2026_sim_kernel.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <stdexcept>


#define CUDA_CHECK(call)                                      \
    do {                                                       \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess) {                           \
            std::cerr                                       \
                << "CUDA error: "                            \
                << cudaGetErrorString(error)                \
                << "\nFile: " << __FILE__                  \
                << "\nLine: " << __LINE__                  \
                << std::endl;                                \
            std::exit(EXIT_FAILURE);                          \
        }                                                      \
    } while (0)


int main(int argc, char* argv[])
{
    /*
     * --------------------------------------------------------
     * Number of simulations
     * --------------------------------------------------------
     */

    int num_simulations = 10000;

    if (argc >= 2) {

        num_simulations =
            std::stoi(argv[1]);
    }

    if (num_simulations <= 0) {

        std::cerr
            << "Number of simulations must be > 0.\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "World Cup CUDA Monte Carlo\n";

    std::cout
        << "Simulations: "
        << num_simulations
        << "\n";


    /*
     * --------------------------------------------------------
     * Load CPU data
     * --------------------------------------------------------
     */

    std::cout
        << "Loading lambda matrix...\n";

    WC2026Data data =
        load_wc2026_data(
            "../data/lambda_matrix.csv"
        );

    std::cout
        << "Loaded "
        << data.teams.size()
        << " teams.\n";

    std::cout
        << "Loaded "
        << data.lambda_matrix.size()
        << " lambda values.\n";

    std::cout
        << "Generated "
        << data.fixtures.size()
        << " group fixtures.\n";


    /*
     * --------------------------------------------------------
     * Fresh random seed
     * --------------------------------------------------------
     */

    std::random_device rd;

    unsigned long long seed =
        (
            static_cast<unsigned long long>(
                rd()
            )
            << 32
        )
        ^
        static_cast<unsigned long long>(
            rd()
        );

    std::cout
        << "Master seed: "
        << seed
        << "\n";


    /*
     * --------------------------------------------------------
     * Device pointers
     * --------------------------------------------------------
     */

    TeamGPU* d_teams = nullptr;

    double* d_lambda_matrix = nullptr;

    FixtureGPU* d_fixtures = nullptr;

    int* d_champion_counts = nullptr;

    int* d_round_counts = nullptr;


    /*
     * --------------------------------------------------------
     * Allocate GPU memory
     * --------------------------------------------------------
     */

    CUDA_CHECK(
        cudaMalloc(
            &d_teams,
            WC_NUM_TEAMS
            * sizeof(TeamGPU)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_lambda_matrix,
            WC_NUM_TEAMS
            * WC_NUM_TEAMS
            * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_fixtures,
            data.fixtures.size()
            * sizeof(FixtureGPU)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_champion_counts,
            WC_NUM_TEAMS
            * sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_round_counts,
            WC_NUM_TEAMS
            * WC_NUM_STAGES
            * sizeof(int)
        )
    );


    /*
     * --------------------------------------------------------
     * Copy CPU -> GPU
     * --------------------------------------------------------
     */

    std::vector<TeamGPU> host_teams(
        WC_NUM_TEAMS
    );

    for (int i = 0;
         i < WC_NUM_TEAMS;
         ++i) {

        host_teams[i] =
            data.teams[i].gpu;
    }

    CUDA_CHECK(
        cudaMemcpy(
            d_teams,
            host_teams.data(),
            WC_NUM_TEAMS
            * sizeof(TeamGPU),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_lambda_matrix,
            data.lambda_matrix.data(),
            WC_NUM_TEAMS
            * WC_NUM_TEAMS
            * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_fixtures,
            data.fixtures.data(),
            data.fixtures.size()
            * sizeof(FixtureGPU),
            cudaMemcpyHostToDevice
        )
    );


    /*
     * --------------------------------------------------------
     * Initialize output counters
     * --------------------------------------------------------
     */

    CUDA_CHECK(
        cudaMemset(
            d_champion_counts,
            0,
            WC_NUM_TEAMS * sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_round_counts,
            0,
            WC_NUM_TEAMS
            * WC_NUM_STAGES
            * sizeof(int)
        )
    );


    /*
     * --------------------------------------------------------
     * Run simulation
     * --------------------------------------------------------
     */

    std::cout
        << "Starting CUDA simulation...\n";

    auto start =
        std::chrono::high_resolution_clock::now();


    launch_wc2026_simulation(
        d_teams,
        d_lambda_matrix,
        d_fixtures,
        num_simulations,
        seed,
        d_champion_counts,
        d_round_counts
    );


    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    auto end =
        std::chrono::high_resolution_clock::now();


    double runtime =
        std::chrono::duration<double>(
            end - start
        ).count();


    /*
     * --------------------------------------------------------
     * Copy results GPU -> CPU
     * --------------------------------------------------------
     */

    std::vector<int> champion_counts(
        WC_NUM_TEAMS,
        0
    );

    std::vector<int> round_counts(
        WC_NUM_TEAMS
        * WC_NUM_STAGES,
        0
    );


    CUDA_CHECK(
        cudaMemcpy(
            champion_counts.data(),
            d_champion_counts,
            WC_NUM_TEAMS * sizeof(int),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            round_counts.data(),
            d_round_counts,
            WC_NUM_TEAMS
            * WC_NUM_STAGES
            * sizeof(int),
            cudaMemcpyDeviceToHost
        )
    );


    /*
     * --------------------------------------------------------
     * Write results
     * --------------------------------------------------------
     */

    write_wc2026_results(
        data,
        champion_counts,
        round_counts,
        num_simulations
    );


    /*
     * --------------------------------------------------------
     * Performance
     * --------------------------------------------------------
     */

    double simulations_per_second =
        static_cast<double>(
            num_simulations
        )
        / runtime;


    std::cout
        << "\nSimulation complete.\n";

    std::cout
        << "Runtime: "
        << runtime
        << " seconds\n";

    std::cout
        << "Simulations/sec: "
        << simulations_per_second
        << "\n";


    /*
     * --------------------------------------------------------
     * Cleanup
     * --------------------------------------------------------
     */

    CUDA_CHECK(
        cudaFree(d_teams)
    );

    CUDA_CHECK(
        cudaFree(d_lambda_matrix)
    );

    CUDA_CHECK(
        cudaFree(d_fixtures)
    );

    CUDA_CHECK(
        cudaFree(d_champion_counts)
    );

    CUDA_CHECK(
        cudaFree(d_round_counts)
    );


    return 0;
}