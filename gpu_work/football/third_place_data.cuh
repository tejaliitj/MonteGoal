#pragma once

#define NUM_THIRDPLACE_SLOTS 8
#define NUM_THIRDPLACE_COMBOS 495

// Slot order everywhere in this table and in resolve_third_place()'s output is
// fixed: 1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L -- matching third_place.py's
// THIRD_PLACE_SLOTS and wc2026_bracket_structure.json's third_place_lookup slots.
extern __constant__ int d_thirdplace_combo_mask[NUM_THIRDPLACE_COMBOS];
extern __constant__ int d_thirdplace_slot_group[NUM_THIRDPLACE_COMBOS][NUM_THIRDPLACE_SLOTS];

// Uploads the table (baked in from wc2026_r32_thirdplace_lookup.csv, see
// third_place_data.cu) to device constant memory. Call once at startup,
// alongside upload_tournament_data().
void upload_thirdplace_table();
