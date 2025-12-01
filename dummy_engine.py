#!/usr/bin/env python3

import sys
import threading

def main():
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        if line == "uci":
            print("id name DummyPythonEngine")
            print("id author You")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line.startswith("position"):
            # ignore actual position for dummy engine
            pass

        elif line.startswith("go"):
            # always play a stupid move like "e2e4"
            print("bestmove e2e4")
            sys.stdout.flush()

        elif line == "quit":
            break


if __name__ == "__main__":
    main()
