#!/usr/bin/env python3
import sys
from nanogpt.nanogpt_module import NanoGptPlayer

MODEL = "lichess_200k_bins_16layers_ckpt_with_optimizer.pt"

def main():
    try:
        player = NanoGptPlayer(model_name=MODEL)
    except Exception as e:
        print(f"info string Failed to load NanoGPT player: {e}")
        sys.stdout.flush()
        sys.exit(1)

    while True:
        line = sys.stdin.readline()
        if not line:
            continue
        line = line.strip()

        if line == "uci":
            print("id name ChessGPT Handshake Test")
            print("id author Repo")
            print("uciok")
            sys.stdout.flush()
        elif line == "isready":
            print("readyok")
            sys.stdout.flush()
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
