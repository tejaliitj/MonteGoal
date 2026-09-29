// #pragma once
// #include <string>
// #include <vector>

// void write_csv(
//     const std::string& filename,
//     const std::vector<std::string>& headers,
//     const std::vector<std::vector<std::string>>& rows
// );



#pragma once

#include <string>
#include <vector>

void write_csv(
    const std::string& filename,
    const std::vector<std::string>& headers,
    const std::vector<std::vector<std::string>>& rows
);