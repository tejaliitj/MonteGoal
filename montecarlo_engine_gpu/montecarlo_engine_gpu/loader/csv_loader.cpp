#include "csv_loader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

static std::string trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
        out.push_back(trim(cell));
    return out;
}

bool load_lambda_matrix(
    const std::string &path,
    std::vector<std::string> &team_names,
    std::map<std::string, int> &team_to_id,
    float h_lambda[NUM_TEAMS * NUM_TEAMS])
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "ERROR: could not open lambda matrix CSV '" << path << "'\n";
        return false;
    }

    std::string header_line;
    if (!std::getline(f, header_line))
    {
        std::cerr << "ERROR: '" << path << "' is empty\n";
        return false;
    }

    std::vector<std::string> header = split_csv_line(header_line);

    // First column is the blank row-label header; the rest are team names,
    // in the order that fixes their integer IDs for the whole program.
    team_names.clear();
    for (size_t i = 1; i < header.size(); i++)
        team_names.push_back(header[i]);

    if ((int)team_names.size() != NUM_TEAMS)
    {
        std::cerr << "ERROR: expected " << NUM_TEAMS << " teams in header, found "
                   << team_names.size() << "\n";
        return false;
    }

    team_to_id.clear();
    for (int i = 0; i < NUM_TEAMS; i++)
        team_to_id[team_names[i]] = i;

    std::string line;
    int row_idx = 0;
    while (std::getline(f, line))
    {
        if (trim(line).empty())
            continue;

        std::vector<std::string> cells = split_csv_line(line);
        if ((int)cells.size() != NUM_TEAMS + 1)
        {
            std::cerr << "ERROR: row " << row_idx << " has " << cells.size()
                       << " columns, expected " << (NUM_TEAMS + 1) << "\n";
            return false;
        }

        std::string row_team = cells[0];
        auto it = team_to_id.find(row_team);
        if (it == team_to_id.end())
        {
            std::cerr << "ERROR: row label '" << row_team
                       << "' not found among header teams\n";
            return false;
        }
        int home_id = it->second;

        for (int col = 0; col < NUM_TEAMS; col++)
        {
            float val;
            try
            {
                val = std::stof(cells[col + 1]);
            }
            catch (...)
            {
                std::cerr << "ERROR: bad lambda value at row '" << row_team
                           << "', column " << col << "\n";
                return false;
            }
            h_lambda[home_id * NUM_TEAMS + col] = val;
        }

        row_idx++;
    }

    if (row_idx != NUM_TEAMS)
    {
        std::cerr << "ERROR: expected " << NUM_TEAMS << " data rows, found "
                   << row_idx << "\n";
        return false;
    }

    std::cout << "Loaded " << NUM_TEAMS << "x" << NUM_TEAMS
               << " lambda matrix from '" << path << "'\n";
    return true;
}

bool load_groups_csv(
    const std::string &path,
    const std::map<std::string, int> &team_to_id,
    int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP])
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "ERROR: could not open groups CSV '" << path << "'\n";
        return false;
    }

    std::string line;
    bool first_line = true;

    // std::map<char, ...> iterates in ascending key order, which conveniently
    // gives us A..L order for free -- same order the bracket module assumes
    // (letter index 0=A .. 11=L).
    std::map<char, std::vector<int>> by_group;

    while (std::getline(f, line))
    {
        if (trim(line).empty())
            continue;

        std::vector<std::string> cells = split_csv_line(line);

        if (first_line)
        {
            first_line = false;
            std::string lower0 = cells.empty() ? "" : cells[0];
            std::transform(lower0.begin(), lower0.end(), lower0.begin(), ::tolower);
            if (lower0.find("team") != std::string::npos)
                continue; // header row, skip
        }

        if (cells.size() < 2)
            continue;

        std::string team = cells[0];
        std::string group = cells[1];
        if (team.empty() || group.empty())
            continue;

        auto it = team_to_id.find(team);
        if (it == team_to_id.end())
        {
            std::cerr << "ERROR: team '" << team
                       << "' in groups CSV not found in the lambda matrix\n";
            return false;
        }

        char group_letter = group[0];
        by_group[group_letter].push_back(it->second);
    }

    if ((int)by_group.size() != NUM_GROUPS)
    {
        std::cerr << "ERROR: found " << by_group.size()
                   << " groups, expected " << NUM_GROUPS << "\n";
        return false;
    }

    int g = 0;
    for (auto &kv : by_group)
    {
        if ((int)kv.second.size() != TEAMS_PER_GROUP)
        {
            std::cerr << "ERROR: group '" << kv.first << "' has " << kv.second.size()
                       << " teams, expected " << TEAMS_PER_GROUP << "\n";
            return false;
        }
        for (int t = 0; t < TEAMS_PER_GROUP; t++)
            h_group_teams[g][t] = kv.second[t];
        g++;
    }

    std::cout << "Loaded " << NUM_GROUPS << " groups from '" << path << "'\n";
    return true;
}
