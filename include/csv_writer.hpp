// =============================================================================
// csv_writer.hpp
// -----------------------------------------------------------------------------
// Implements "Push dataset: montecarlo_featuresX.csv" from your notes.
// Writes a host-side float array (already copied back from the GPU) to a CSV
// file with an auto-incrementing filename: montecarlo_features1.csv,
// montecarlo_features2.csv, ... so repeated runs never overwrite each other.
// =============================================================================
#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>

#include <iomanip>
#include "kernels/kernel_interface.cuh"

inline std::string next_available_filename(const std::string& dir,
                                            const std::string& prefix,
                                            const std::string& ext) {
    int idx = 1;
    std::string path;
    do {
        path = dir + "/" + prefix + std::to_string(idx) + ext;
        idx++;
    } while (std::ifstream(path).good());
    return path;
}

inline void write_csv(const std::string& out_dir,
                       const std::string& header,
                       const std::vector<SimulationResult>& results) {
    // Ensure the output directory exists.
    std::filesystem::create_directories(out_dir);

    std::string path = next_available_filename(out_dir, "montecarlo_features", ".csv");
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open " << path << " for writing.\n";
        return;
    }

    unsigned long long min_start_ns = 0;
    if (!results.empty()) {
        min_start_ns = results[0].start_ns;
        for (const auto& r : results) {
            if (r.start_ns < min_start_ns) min_start_ns = r.start_ns;
        }
    }

    file << header << "\n";
    for (size_t i = 0; i < results.size(); ++i) {
        double start_offset_us = (results[i].start_ns >= min_start_ns)
            ? static_cast<double>(results[i].start_ns - min_start_ns) / 1000.0
            : 0.0;
        double end_offset_us = (results[i].end_ns >= min_start_ns)
            ? static_cast<double>(results[i].end_ns - min_start_ns) / 1000.0
            : 0.0;
        file << i << ","
             << results[i].estimate << ","
             << results[i].error << ","
             << start_offset_us << ","
             << end_offset_us << ","
             << results[i].time_us << "\n";
    }
    file.close();

    std::cout << "[OK] Pushed dataset -> " << path
              << "  (" << results.size() << " rows)\n";
}

inline void write_csv(const std::string& out_dir,
                       const std::string& header,
                       const std::vector<float>& values) {
    // Ensure the output directory exists.
    std::filesystem::create_directories(out_dir);

    std::string path = next_available_filename(out_dir, "montecarlo_features", ".csv");
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open " << path << " for writing.\n";
        return;
    }

    file << header << "\n";
    for (size_t i = 0; i < values.size(); ++i) {
        file << i << "," << values[i] << "\n";
    }
    file.close();

    std::cout << "[OK] Pushed dataset -> " << path
              << "  (" << values.size() << " rows)\n";
}
