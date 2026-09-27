#include "bracket.cuh"
#include <cuda_runtime.h>

__constant__ int d_r32_kind[16][2];
__constant__ int d_r32_letter[16][2];
__constant__ int d_r16_pairs[8][2];
__constant__ int d_qf_pairs[4][2];
__constant__ int d_sf_pairs[2][2];
__constant__ int d_third_slot_order[8];

// Letter index: A=0 B=1 C=2 D=3 E=4 F=5 G=6 H=7 I=8 J=9 K=10 L=11.
// Transcribed 1:1 from bracket.py's R32_FIXTURES (match numbers in the
// comments match the FIFA match numbers referenced there).
static const int h_r32_kind[16][2] = {
    {ROLE_R, ROLE_R}, // Match 73: R:A R:B
    {ROLE_W, ROLE_T}, // Match 74: W:E T:E
    {ROLE_W, ROLE_R}, // Match 75: W:F R:C
    {ROLE_W, ROLE_R}, // Match 76: W:C R:F
    {ROLE_W, ROLE_T}, // Match 77: W:I T:I
    {ROLE_R, ROLE_R}, // Match 78: R:E R:I
    {ROLE_W, ROLE_T}, // Match 79: W:A T:A
    {ROLE_W, ROLE_T}, // Match 80: W:L T:L
    {ROLE_W, ROLE_T}, // Match 81: W:D T:D
    {ROLE_W, ROLE_T}, // Match 82: W:G T:G
    {ROLE_R, ROLE_R}, // Match 83: R:K R:L
    {ROLE_W, ROLE_R}, // Match 84: W:H R:J
    {ROLE_W, ROLE_T}, // Match 85: W:B T:B
    {ROLE_W, ROLE_R}, // Match 86: W:J R:H
    {ROLE_W, ROLE_T}, // Match 87: W:K T:K
    {ROLE_R, ROLE_R}, // Match 88: R:D R:G
};

static const int h_r32_letter[16][2] = {
    {0, 1},   // A,  B
    {4, 4},   // E,  E
    {5, 2},   // F,  C
    {2, 5},   // C,  F
    {8, 8},   // I,  I
    {4, 8},   // E,  I
    {0, 0},   // A,  A
    {11, 11}, // L,  L
    {3, 3},   // D,  D
    {6, 6},   // G,  G
    {10, 11}, // K,  L
    {7, 9},   // H,  J
    {1, 1},   // B,  B
    {9, 7},   // J,  H
    {10, 10}, // K,  K
    {3, 6},   // D,  G
};

// R16 pairs index into the 16 R32 winners (0=M73 .. 15=M88).
static const int h_r16_pairs[8][2] = {
    {1, 4}, {0, 2}, {3, 5}, {6, 7}, {10, 11}, {8, 9}, {13, 15}, {12, 14}
};

// QF pairs index into the 8 R16 winners.
static const int h_qf_pairs[4][2] = {
    {0, 1}, {4, 5}, {2, 3}, {6, 7}
};

// SF pairs index into the 4 QF winners.
static const int h_sf_pairs[2][2] = {
    {0, 1}, {2, 3}
};

// slot_order = ["E","I","A","L","D","G","B","K"] from bracket.py, as letter indices.
static const int h_third_slot_order[8] = {4, 8, 0, 11, 3, 6, 1, 10};

void upload_bracket_constants()
{
    cudaMemcpyToSymbol(d_r32_kind, h_r32_kind, sizeof(h_r32_kind));
    cudaMemcpyToSymbol(d_r32_letter, h_r32_letter, sizeof(h_r32_letter));
    cudaMemcpyToSymbol(d_r16_pairs, h_r16_pairs, sizeof(h_r16_pairs));
    cudaMemcpyToSymbol(d_qf_pairs, h_qf_pairs, sizeof(h_qf_pairs));
    cudaMemcpyToSymbol(d_sf_pairs, h_sf_pairs, sizeof(h_sf_pairs));
    cudaMemcpyToSymbol(d_third_slot_order, h_third_slot_order, sizeof(h_third_slot_order));
}

__device__ int resolve_role_device(
    int kind, int letter,
    const int group_winner[NUM_GROUPS],
    const int group_runner_up[NUM_GROUPS],
    const int third_team_by_group[NUM_GROUPS],
    const int slot_to_third_group[NUM_GROUPS])
{
    if (kind == ROLE_W)
        return group_winner[letter];
    if (kind == ROLE_R)
        return group_runner_up[letter];

    // ROLE_T: 'letter' here is the *slot* letter -- look up which group's
    // third-place team was assigned to that slot.
    int src_group = slot_to_third_group[letter];
    return third_team_by_group[src_group];
}

__device__ void assign_third_place_slots_device(
    const int qualifying_group_letters[8],
    int slot_to_third_group[NUM_GROUPS])
{
    for (int i = 0; i < NUM_GROUPS; i++)
        slot_to_third_group[i] = -1;

    // sorted3 = sorted(qualifying_groups) -- ascending letter-index sort, 8 elements.
    int sorted3[8];
    for (int i = 0; i < 8; i++)
        sorted3[i] = qualifying_group_letters[i];

    for (int i = 1; i < 8; i++)
    {
        int key = sorted3[i];
        int j = i - 1;
        while (j >= 0 && sorted3[j] > key)
        {
            sorted3[j + 1] = sorted3[j];
            j--;
        }
        sorted3[j + 1] = key;
    }

    for (int i = 0; i < 8; i++)
    {
        int slot = d_third_slot_order[i];
        slot_to_third_group[slot] = sorted3[i];
    }

    // Swap-avoidance pass, mirrors bracket.py: if a group's own third ended
    // up in its own slot letter, swap with another slot whose assigned
    // group isn't this slot and isn't stuck on its own slot either.
    for (int i = 0; i < 8; i++)
    {
        int slot = d_third_slot_order[i];
        if (slot_to_third_group[slot] == slot)
        {
            for (int j = 0; j < 8; j++)
            {
                if (j == i)
                    continue;
                int other_slot = d_third_slot_order[j];
                if (slot_to_third_group[other_slot] != slot &&
                    slot_to_third_group[other_slot] != other_slot)
                {
                    int tmp = slot_to_third_group[slot];
                    slot_to_third_group[slot] = slot_to_third_group[other_slot];
                    slot_to_third_group[other_slot] = tmp;
                    break;
                }
            }
        }
    }
}
