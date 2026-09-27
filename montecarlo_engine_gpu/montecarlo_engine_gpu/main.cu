/*
 * main.cu -- GPU version of main.py. Loads the lambda matrix + groups CSV,
 * runs N_RUNS tournaments in parallel (one thread per tournament), prints
 * the championship-count and furthest-stage-reached summary, and writes a
 * champion_probabilities_cuda.csv (same shape as the earlier pi-project's
 * CSV pipeline).
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <curand_kernel.h>

#include "config.cuh"
#include "rng.cuh"
#include "loader/csv_loader.h"
#include "simulation/bracket.cuh"
#include "simulation/match_engine.cuh"
#include "simulation/tournament.cuh"

using namespace std;

int main(int argc, char **argv)
{
    // ---------------- LOAD DATA ----------------
    vector<string> team_names;
    map<string, int> team_to_id;
    static float h_lambda[NUM_TEAMS * NUM_TEAMS];

    cout << "Loading lambda matrix from: " << LAMBDA_MATRIX_CSV_PATH << endl;
    if (!load_lambda_matrix(LAMBDA_MATRIX_CSV_PATH, team_names, team_to_id, h_lambda))
    {
        cerr << "Failed to load lambda matrix. Exiting.\n";
        return 1;
    }

    int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP];
    cout << "Loading groups from: " << GROUPS_CSV_PATH << endl;
    if (!load_groups_csv(GROUPS_CSV_PATH, team_to_id, h_group_teams))
    {
        cerr << "Failed to load groups CSV. Exiting.\n";
        return 1;
    }

    cout << "Loaded " << NUM_TEAMS << " teams / " << NUM_GROUPS << " groups.\n\n";

    // ---------------- CONFIG ----------------
    // Trial count: ./montegoal_wc <N>  overrides config.cuh's N_RUNS at
    // runtime, so you don't need to edit the header and recompile just to
    // change how many tournaments to simulate.
    int total_threads = N_RUNS;
    if (argc > 1)
    {
        long parsed = atol(argv[1]);
        if (parsed <= 0)
        {
            cerr << "ERROR: trial count must be a positive integer, got '" << argv[1] << "'\n";
            cerr << "Usage: " << argv[0] << " [num_trials]\n";
            return 1;
        }
        total_threads = (int)parsed;
    }

    unsigned long long seed = 42;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    cout << "============================================\n";
    cout << "     CUDA FIFA WORLD CUP 2026 SIMULATION\n";
    cout << "============================================\n";
    cout << "RNG               : Philox\n";
    cout << "Seed              : " << seed << endl;
    cout << "Tournaments (runs): " << total_threads << endl;
    cout << "Threads/Block     : " << threads_per_block << endl;
    cout << "Blocks            : " << blocks << endl;
    cout << "============================================\n\n";

    // ---------------- UPLOAD CONSTANT DATA ----------------
    upload_bracket_constants();
    upload_lambda_matrix(h_lambda);
    upload_group_teams(h_group_teams);

    // ---------------- ALLOCATE GPU MEMORY ----------------
    curandStatePhilox4_32_10_t *d_states;
    cudaError_t err = cudaMalloc(&d_states, total_threads * sizeof(curandStatePhilox4_32_10_t));
    if (err != cudaSuccess)
    {
        cerr << "RNG Memory Allocation Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    int *d_champion, *d_semis, *d_finalists;
    cudaMalloc(&d_champion, total_threads * sizeof(int));
    cudaMalloc(&d_semis, total_threads * 4 * sizeof(int));
    cudaMalloc(&d_finalists, total_threads * 2 * sizeof(int));

    // ---------------- INIT RNG ----------------
    setup_rng<<<blocks, threads_per_block>>>(d_states, seed, total_threads);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "RNG Kernel Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    cudaDeviceSynchronize();

    // ---------------- RUN SIMULATION ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    world_cup_kernel<<<blocks, threads_per_block>>>(
        d_states, total_threads, d_champion, d_semis, d_finalists);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "World Cup Kernel Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // ---------------- COPY BACK ----------------
    vector<int> h_champion(total_threads);
    vector<int> h_semis(total_threads * 4);
    vector<int> h_finalists(total_threads * 2);

    cudaMemcpy(h_champion.data(), d_champion, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_semis.data(), d_semis, total_threads * 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_finalists.data(), d_finalists, total_threads * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------- AGGREGATE ----------------
    vector<int> champ_count(NUM_TEAMS, 0);
    vector<int> final_count(NUM_TEAMS, 0);
    vector<int> semi_count(NUM_TEAMS, 0);

    for (int i = 0; i < total_threads; i++)
    {
        champ_count[h_champion[i]]++;
        final_count[h_finalists[i * 2 + 0]]++;
        final_count[h_finalists[i * 2 + 1]]++;
        for (int s = 0; s < 4; s++)
            semi_count[h_semis[i * 4 + s]]++;
    }

    vector<int> champ_order(NUM_TEAMS);
    for (int i = 0; i < NUM_TEAMS; i++)
        champ_order[i] = i;
    sort(champ_order.begin(), champ_order.end(),
         [&](int a, int b) { return champ_count[a] > champ_count[b]; });

    // ---------------- PRINT SUMMARY ----------------
    cout << "\n============================================\n";
    cout << "              AGGREGATED SUMMARY\n";
    cout << "============================================\n";
    cout << "Total CUDA Threads      : " << total_threads << endl;
    cout << "GPU Execution Time      : " << fixed << setprecision(4) << gpu_ms << " ms\n";
    cout << "============================================\n\n";

    cout << "Champion probability (teams with at least one title):\n";
    for (int id : champ_order)
    {
        if (champ_count[id] == 0)
            continue;
        double pct = 100.0 * champ_count[id] / total_threads;
        cout << "   " << left << setw(24) << team_names[id]
             << right << setw(10) << champ_count[id] << " titles"
             << "  (" << fixed << setprecision(2) << setw(5) << pct << "%)\n";
    }

    vector<int> sf_order(NUM_TEAMS);
    for (int i = 0; i < NUM_TEAMS; i++)
        sf_order[i] = i;
    sort(sf_order.begin(), sf_order.end(),
         [&](int a, int b) { return semi_count[a] > semi_count[b]; });

    cout << "\nFurthest stage reached (teams with >=1 semifinal appearance):\n";
    cout << left << setw(24) << "Team" << right
         << setw(8) << "SF" << setw(8) << "Final" << setw(8) << "Champ" << endl;

    for (int id : sf_order)
    {
        if (semi_count[id] == 0)
            continue;
        cout << left << setw(24) << team_names[id] << right
             << setw(8) << semi_count[id]
             << setw(8) << final_count[id]
             << setw(8) << champ_count[id] << endl;
    }

    // ---------------- WRITE CSV ----------------
    ofstream csv("champion_probabilities_cuda.csv");
    csv << "team,champion_count,champion_probability\n";
    for (int id : champ_order)
    {
        double prob = (double)champ_count[id] / total_threads;
        csv << team_names[id] << "," << champ_count[id] << ","
            << fixed << setprecision(6) << prob << "\n";
    }
    csv.close();
    cout << "\nWrote champion_probabilities_cuda.csv\n";

    // ---------------- CLEANUP ----------------
    cudaFree(d_states);
    cudaFree(d_champion);
    cudaFree(d_semis);
    cudaFree(d_finalists);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
