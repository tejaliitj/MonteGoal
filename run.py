#!/usr/bin/env python3
"""
Monte Carlo Engine CLI Runner
Interfaces with the compiled CUDA shared library (montecarlo_engine.dll)
"""

import os
import sys
import ctypes
import argparse

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo CUDA Engine Runner")
    parser.add_argument("app", nargs="?", default="pi", choices=["pi", "poisson"],
                        help="Application kernel to run: 'pi' or 'poisson' (default: pi)")
    parser.add_argument("threads", nargs="?", type=int, default=1048576,
                        help="Number of simulations / GPU threads (default: 1048576)")
    parser.add_argument("trials", nargs="?", type=int, default=1000,
                        help="Trials per simulation (default: 1000)")
    parser.add_argument("seed", nargs="?", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(base_dir, "montecarlo_engine.dll")

    if not os.path.exists(dll_path):
        print(f"[ERROR] Engine DLL not found at: {dll_path}", file=sys.stderr)
        sys.exit(1)

    try:
        engine_lib = ctypes.CDLL(dll_path)
        engine_lib.run_engine.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulonglong
        ]
        engine_lib.run_engine.restype = ctypes.c_int

        app_bytes = args.app.encode("utf-8")
        status = engine_lib.run_engine(app_bytes, args.threads, args.trials, args.seed)
        sys.exit(status)
    except Exception as e:
        print(f"[ERROR] Failed to execute Monte Carlo Engine: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
