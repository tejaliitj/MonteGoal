#include "bracket_data.cuh"
#include "cuda_check.cuh"

// Generated from wc2026_bracket_structure.json by a Python script, not
// hand-transcribed -- match_id 0..31 corresponds to original match numbers
// 73..104. Order matches the JSON's own bottom-up layout (round_of_32,
// round_of_16, quarterfinals, semifinals, third_place_match, final), which is
// already dependency-safe: every winner_of_match/loser_of_match reference
// points to a strictly earlier match_id (verified at generation time).
// match_id, team_a_type, team_a_arg, team_b_type, team_b_arg
const int h_bracket_team_a_type[NUM_KNOCKOUT_MATCHES] = {
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3
};

const int h_bracket_team_a_arg[NUM_KNOCKOUT_MATCHES] = {
  0, 4, 5, 2, 8, 4, 0, 11, 3, 6, 10, 7, 1, 9, 10, 3, 1, 0, 3, 6, 10, 8, 13, 12, 16, 20, 18, 22, 24, 26, 28, 28
};

const int h_bracket_team_b_type[NUM_KNOCKOUT_MATCHES] = {
  1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3
};

const int h_bracket_team_b_arg[NUM_KNOCKOUT_MATCHES] = {
  1, 3, 2, 5, 5, 8, 0, 7, 2, 4, 11, 9, 1, 7, 6, 6, 4, 2, 5, 7, 11, 9, 15, 14, 17, 21, 19, 23, 25, 27, 29, 29
};

__constant__ int d_bracket_team_a_type[NUM_KNOCKOUT_MATCHES];
__constant__ int d_bracket_team_a_arg[NUM_KNOCKOUT_MATCHES];
__constant__ int d_bracket_team_b_type[NUM_KNOCKOUT_MATCHES];
__constant__ int d_bracket_team_b_arg[NUM_KNOCKOUT_MATCHES];

void upload_bracket_data() {
    CUDA_CHECK(cudaMemcpyToSymbol(d_bracket_team_a_type, h_bracket_team_a_type,
                                   sizeof(h_bracket_team_a_type)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bracket_team_a_arg, h_bracket_team_a_arg,
                                   sizeof(h_bracket_team_a_arg)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bracket_team_b_type, h_bracket_team_b_type,
                                   sizeof(h_bracket_team_b_type)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_bracket_team_b_arg, h_bracket_team_b_arg,
                                   sizeof(h_bracket_team_b_arg)));
}
