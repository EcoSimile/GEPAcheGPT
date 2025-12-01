#!/usr/bin/env python3
import sys
import chess
import random

NAME = "RandomPythonEngine"
AUTHOR = "Chris + Theo"

class RandomEngine:
    def __init__(self):
        self.board = chess.Board()

    def position(self, command: str):
        """
        Handle 'position' commands like:
        - position startpos
        - position startpos moves e2e4 e7e5
        - position fen <fen> moves ...
        """
        tokens = command.split()

        if "startpos" in tokens:
            self.board = chess.Board()
            # If there are moves after 'moves', push them.
            if "moves" in tokens:
                idx = tokens.index("moves") + 1
                for mv in tokens[idx:]:
                    self.board.push_uci(mv)

        elif "fen" in tokens:
            # position fen <fenstring> [moves ...]
            fen_index = tokens.index("fen") + 1
            # FEN is 6 fields; join them and split later
            fen = " ".join(tokens[fen_index:fen_index + 6])
            self.board = chess.Board(fen)
            if "moves" in tokens:
                idx = tokens.index("moves") + 1
                for mv in tokens[idx:]:
                    self.board.push_uci(mv)

    def go(self, command: str):
        """
        Handle 'go' commands. For now we ignore all search parameters
        and just pick a random legal move.
        """
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            # No legal moves: either checkmate or stalemate
            print("bestmove 0000")
            sys.stdout.flush()
            return

        move = random.choice(legal_moves)
        print(f"bestmove {move.uci()}")
        sys.stdout.flush()


def main():
    engine = RandomEngine()

    while True:
        line = sys.stdin.readline()
        if not line:
            continue
        line = line.strip()

        if line == "uci":
            print(f"id name {NAME}")
            print(f"id author {AUTHOR}")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line.startswith("ucinewgame"):
            engine.board = chess.Board()

        elif line.startswith("position"):
            engine.position(line)

        elif line.startswith("go"):
            engine.go(line)

        elif line == "quit":
            break

        # you can handle 'setoption' here later if needed


if __name__ == "__main__":
    main()
