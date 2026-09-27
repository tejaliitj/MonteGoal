# montecarlo_engine_gpu

CUDA port of the CPU World Cup 2026 simulator. One GPU thread = one full
48-team tournament; `N_RUNS` (in `config.cuh`) tournaments run in parallel,
and the results are aggregated into a championship-probability table, same
shape as the CPU version's output.

## Module map (Python -> CUDA)

| Python file       | CUDA equivalent                              | Notes |
|--------------------|-----------------------------------------------|-------|
| `config.py`         | `config.cuh`                                   | same constants, `N_RUNS` is now also the thread count |
| `csv_loader.py`      | `loader/csv_loader.cpp` / `.h`                | loads the dense `lambda_matrix.csv` instead of the sparse xG CSV; builds flat arrays + a `team name -> id (0..47)` map instead of dicts |
| `match_engine.py`   | `simulation/match_engine.cu` / `.cuh`         | Poisson goals via your existing `distributions/poisson.cuh`; the `(home,away)`/`(away,home)` fallback is gone since the lambda matrix is dense and directional (home advantage is baked into the asymmetry) |
| `group_stage.py`    | `simulation/group_stage.cu` / `.cuh`          | Python's `sorted()`/`cmp_to_key` becomes fixed-size insertion sorts (4 teams for group ranking, 12 for third-place ranking) since GPU code can't call into Python's generic sort |
| `bracket.py`        | `simulation/bracket.cu` / `.cuh`              | roles (`"W:A"`, `"R:B"`, `"T:E"`) become `(kind, letter)` int pairs; all fixture/pairing tables and the simplified third-place slot logic are transcribed 1:1 |
| `tournament.py`     | `simulation/tournament.cu` / `.cuh`           | `run_tournament()` becomes `world_cup_kernel`, the `__global__` entry point, one call per thread |
| `main.py`            | `main.cu`                                      | loads data, uploads it to GPU constant memory, launches the kernel, aggregates results on the host, prints the same two summary tables, and additionally writes `champion_probabilities_cuda.csv` |
| `rng.cuh/.cu`, `distributions/*` | unchanged, reused as-is from the earlier pi-estimation project |

## Known simplification carried over from the Python version

Same caveat as `bracket.py`: FIFA's real third-place-qualifier slotting
depends on a published 495-row combination table that wasn't embedded here.
`assign_third_place_slots_device` uses the same documented simplified
substitute the Python version used. Everything else in the bracket (who
plays whom by role, and the match graph through R16/QF/SF/Final) is the
real FIFA structure.

The third-place playoff is not simulated on GPU (the Python version plays
it but never folds its result into the aggregated `stats` dict either, so
this doesn't change any of the output).

## Data flow

- `data/lambda_matrix.csv`: 48x48 dense matrix, row = home team, column =
  away team, value = that match's expected goals (Poisson lambda) for the
  home team. The header's column order fixes each team's integer ID
  (0..47) for the whole program. Uploaded once to `__constant__` GPU
  memory (~9KB, fits easily in the 64KB constant cache).
- `data/teams_groups_2026.csv`: `Team,Group` rows, resolved to team IDs and
  grouped A..L (group index 0..11), uploaded to `__constant__` memory.
- Both matrices/tables are read-only for every thread -- ideal fit for
  constant memory, which is cached and broadcasts to all threads reading
  the same address in a warp.

## Build & run

Requires `nvcc` (CUDA toolkit) on the machine.

```bash
make
./montegoal_wc
```

or just:

```bash
make run
```

Change `N_RUNS` in `config.cuh` to run more or fewer tournaments (10,000 by
default). Output: a console summary (championship counts + furthest-stage
table) and `champion_probabilities_cuda.csv` written to the working
directory in the same `team,champion_count,champion_probability` shape as
before.

## Validation note

Python's Mersenne Twister and CUDA's Philox are different RNGs, so results
won't be bit-identical between the two versions even with "the same seed."
This port was validated structurally instead: every simulated tournament's
champion is always one of its own two finalists (sanity-checked over
20,000 runs with zero violations), and the resulting championship
probabilities land in the same ballpark and ranking as the CPU version and
the earlier hand-built CUDA prototype (France/Spain/Portugal/Argentina/
Belgium/Brazil/Colombia clustering at the top, in that rough order).
