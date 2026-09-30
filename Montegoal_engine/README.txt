MonteGoal - simple Windows run folder
=====================================

What this version reports:
  - probability of reaching the Quarterfinals
  - probability of reaching the Semifinals
  - probability of reaching the Final
  - probability of winning the tournament

Default run:
  Double-click run_main.bat

Or from PowerShell:
  .\run_main.bat

Custom number of simulations and seed:
  .\run_main.bat 1000000 42

Custom CUDA block size:
  .\run_main.bat 1000000 42 --block 256

The program uses one CUDA thread per simulated tournament. Each thread runs
one complete World Cup and records the teams appearing in the QF, SF, Final,
and the champion. The GPU aggregates four 48-team histograms and only those
192 counters are copied back to the CPU.

Required:
  - NVIDIA GPU
  - CUDA toolkit with nvcc on PATH
  - A supported MSVC toolchain for nvcc on Windows


Choosing the random number generator (separate file per RNG)
============================================================
Each generator lives in its own file inside rng\ :

    rng\rng_philox.cuh     Philox4x32-10  (default; counter-based, cheapest to start)
    rng\rng_xorwow.cuh     XORWOW         (cuRAND classic default)
    rng\rng_mrg32k3a.cuh   MRG32k3a       (L'Ecuyer combined recursive generator)
    rng\rng.cuh            picks one of the above from a compile flag

Nothing else in the project names a generator; they all use RngState / rng_init /
RNG_NAME from rng.cuh. To add another RNG: create rng\rng_<name>.cuh that defines
those three things and add one #elif to rng.cuh.

Build + run with a specific RNG (each gets its own exe in build\):

    .\run_philox.bat   1000000 42
    .\run_xorwow.bat   1000000 42
    .\run_mrg32k3a.bat 1000000 42 --block 256
    .\run_rng.bat xorwow 1000000 42         (same thing, RNG name as first argument)

run_main.bat is unchanged (Philox, build\montegoal.exe).

Compare all three RNGs (same trials + seed, output saved to results\<rng>.txt):

    .\run_compare_rngs.bat 100000000 42
    python tools\compare_rng_outputs.py     (checks they agree within noise)

Notes:
  * The RNGs produce different random numbers, so the exact percentages differ a
    little between RNGs; the probabilities should agree within Monte Carlo noise.
  * XORWOW and MRG32k3a must "skip ahead" inside curand_init to give every trial
    its own stream, which is more expensive than Philox, and their state is larger.
    Expect them to be slower here; the kernel time / tournaments-per-second lines
    in each output show by how much.
