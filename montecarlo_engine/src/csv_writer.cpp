// #include "csv_writer.hpp"
// #include <fstream>
// #include <stdexcept>

// void write_csv(
//     const std::string& filename,
//     const std::vector<std::string>& headers,
//     const std::vector<std::vector<std::string>>& rows)
// {
//     std::ofstream file(filename);
//     if (!file)
//         throw std::runtime_error("Could not open CSV: " + filename);

//     for (size_t i = 0; i < headers.size(); ++i) {
//         if (i) file << ',';
//         file << headers[i];
//     }
//     file << '\n';

//     for (const auto& row : rows) {
//         for (size_t i = 0; i < row.size(); ++i) {
//             if (i) file << ',';
//             file << row[i];
//         }
//         file << '\n';
//     }
// }



#include "csv_writer.hpp"

#include <fstream>
#include <stdexcept>

void write_csv(
    const std::string& filename,
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
) {
    std::ofstream file(filename);

    if (!file.is_open()) {

        throw std::runtime_error(
            "Could not open CSV file: "
            + filename
        );
    }

    for (size_t i = 0;
         i < headers.size();
         ++i) {

        file << headers[i];

        if (i + 1 < headers.size()) {
            file << ",";
        }
    }

    file << "\n";

    for (const auto& row : rows) {

        for (size_t i = 0;
             i < row.size();
             ++i) {

            file << row[i];

            if (i + 1 < row.size()) {
                file << ",";
            }
        }

        file << "\n";
    }
}