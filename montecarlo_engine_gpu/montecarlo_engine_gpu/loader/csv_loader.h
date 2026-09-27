/*
 * csv_loader.h -- reads and validates the two input CSVs, same role as
 * csv_loader.py but outputting flat arrays instead of dicts, since GPU
 * code can't do string keys.
 *
 *     load_lambda_matrix(path, team_names, team_to_id, h_lambda)
 *     load_groups_csv(path, team_to_id, h_group_teams)
 *
 * Nothing here runs a simulation -- this module's only job is turning CSV
 * rows into clean flat data structures, with clear errors if something's
 * wrong (mirrors csv_loader.py's error-reporting style).
 */
#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <string>
#include <vector>
#include <map>

#include "../config.cuh"

// Loads the dense NUM_TEAMS x NUM_TEAMS lambda matrix (row = home team,
// column = away team). The header row's column order fixes each team's
// integer ID 0..47 for the rest of the program.
// Returns true on success.
bool load_lambda_matrix(
    const std::string &path,
    std::vector<std::string> &team_names,      // out: id -> name, size 48
    std::map<std::string, int> &team_to_id,    // out: name -> id
    float h_lambda[NUM_TEAMS * NUM_TEAMS]      // out: flattened row-major
);

// Loads Team,Group rows and resolves each team name to the ID assigned by
// load_lambda_matrix. h_group_teams[g][0..3] = team ids for group g
// (g=0 -> Group A ... g=11 -> Group L, alphabetical order).
bool load_groups_csv(
    const std::string &path,
    const std::map<std::string, int> &team_to_id,
    int h_group_teams[NUM_GROUPS][TEAMS_PER_GROUP]
);

#endif
