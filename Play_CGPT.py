#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import chess

from nanogpt.nanogpt_module import NanoGptPlayer
from play_vs_chessgptCommandLine import GPTPlayer, LegalMoveResponse, get_legal_move

PROMPT_PATH = os.path.join(ROOT_DIR, "gpt_inputs", "prompt.txt")
DEFAULT_NANOGPT_MODEL = "lichess_200k_bins_16layers_ckpt_with_optimizer.pt"
FAILURE_LOG = os.path.join(ROOT_DIR, "logs", "uci_failures.log")
DEBUG_LOG = os.path.join(ROOT_DIR, "logs", "uci_debug.log")
PROMPT_LOG = os.path.join(ROOT_DIR, "logs", "nanogpt_prompts_uci.log")

logging.basicConfig(
    filename=DEBUG_LOG,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log_failure(
    reason: str,
    board: chess.Board,
    game_state: str,
    attempts: int,
    attempt_history: Optional[List[str]],
):
    os.makedirs(os.path.dirname(FAILURE_LOG), exist_ok=True)
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"Reason: {reason}\n")
        side = "white" if board.turn == chess.WHITE else "black"
        f.write(f"Attempts: {attempts}\n")
        if attempt_history:
            f.write("Attempt history:\n")
            for idx, mv in enumerate(attempt_history, 1):
                f.write(f"  {idx}: {mv}\n")
        f.write(f"Side to move: {side}\n")
        f.write(f"FEN: {board.fen()}\n")
        f.write(f"Transcript length (chars): {len(game_state)}\n")
        f.write(f"Transcript tokens: {len(game_state.split())}\n")
        f.write("Transcript:\n")
        f.write(game_state.strip() + "\n")
        f.write("=" * 40 + "\n\n")


def log_prompt_uci(stage: str, board: chess.Board, game_state: str, extra: Optional[str] = None, attempts: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(PROMPT_LOG), exist_ok=True)
    side = "white" if board.turn == chess.WHITE else "black"
    with open(PROMPT_LOG, "a", encoding="utf-8") as f:
        f.write("#" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"Side to move: {side}\n")
        f.write(f"FEN: {board.fen()}\n")
        f.write(f"Transcript length: {len(game_state)}\n")
        if extra:
            f.write(f"Extra: {extra}\n")
        if attempts:
            f.write("Attempts:\n")
            for i, a in enumerate(attempts, 1):
                f.write(f"  {i}: {a}\n")
        f.write("Transcript:\n")
        f.write(game_state + "\n")


class ChessGptUciEngine:
    """Minimal UCI bridge that wraps an existing ChessGPT player."""

    def __init__(self, player, engine_name: str, temperature: float = 0.0):
        self.player = player
        self.engine_name = engine_name
        self.temperature = temperature
        self.base_prompt = self._load_prompt()
        self.board = chess.Board()
        self.game_state = self.base_prompt
        # Track history for reconstruction across 'position' calls
        self.uci_history: List[str] = []
        self.pgn_tokens: List[str] = []  # e.g., ["1. e4 d6", "2. d4 c5"]
        self.history_path = os.path.join(ROOT_DIR, "logs", "active_game_uci.json")

    def _load_prompt(self) -> str:
        with open(PROMPT_PATH, "r") as f:
            return f.read()

    def reset(self):
        self.board.reset()
        self.game_state = self.base_prompt
        self.uci_history = []
        self.pgn_tokens = []
        # Don't delete history file here; some GUIs send 'ucinewgame' before replaying moves

    def handle_position(self, tokens: List[str]):
        if not tokens:
            self.reset()
            return

        idx = 0
        token = tokens[idx]
        start_from_fen = False
        if token == "startpos":
            self.reset()
            idx += 1
        elif token == "fen":
            fen_tokens = tokens[idx + 1 : idx + 7]
            fen = " ".join(fen_tokens)
            self.board.set_fen(fen)
            # Do NOT inject tags or non-training text into the NanoGPT prompt.
            # We'll try to resume from persisted history or build minimal tokens from moves only.
            self.game_state = self.base_prompt
            idx += 7
            start_from_fen = True
        else:
            self.reset()

        if idx < len(tokens) and tokens[idx] == "moves":
            moves = tokens[idx + 1 :]
            logging.info("Rebuilding from moves: %s", moves)
            self._apply_external_moves(moves, start_from_fen=start_from_fen)
        elif start_from_fen and not (idx < len(tokens) and tokens[idx] == "moves"):
            # FEN without moves – try to reconstruct just from saved history
            self._resume_from_saved_history_if_possible()

    def _apply_external_moves(self, moves: List[str], start_from_fen: bool = False):
        if start_from_fen:
            # Try to resume using saved history
            if not self._resume_from_saved_history_if_possible():
                # No history to reconstruct; build minimal transcript from FEN context
                logging.info("FEN resume without matching history; building minimal transcript from FEN context")
                transcript_tokens: List[str] = []
                current_pair: Optional[str] = None
                for move_str in moves:
                    move = chess.Move.from_uci(move_str)
                    san = self.board.san(move)
                    if current_pair is None:
                        # Always use training-style: 'n. SAN' even if it's Black's move
                        current_pair = f"{self.board.fullmove_number}. {san}"
                    else:
                        current_pair += f" {san}"
                        transcript_tokens.append(current_pair)
                        current_pair = None
                    self.board.push(move)
                if current_pair:
                    transcript_tokens.append(current_pair)
                if transcript_tokens:
                    self.game_state = self.base_prompt + " " + " ".join(transcript_tokens)
                    self.uci_history = moves.copy()
                    self.pgn_tokens = transcript_tokens
                    self._persist_history()
                return

        else:
            # startpos case: rebuild from scratch
            self.board.reset()
            self.game_state = self.base_prompt
            self.uci_history = []
            self.pgn_tokens = []

        # At this point, either we've resumed history into self.board/self.pgn_tokens
        # or we're at startpos with empty tokens. Append incoming moves to both.
        current_pair: Optional[str] = None
        if self.pgn_tokens:
            # If last token has both moves? We detect based on side to move
            if self.board.turn == chess.WHITE:
                current_pair = None
            else:
                # Start appending to existing last token
                current_pair = self.pgn_tokens.pop()

        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            san = self.board.san(move)
            if current_pair is None:
                current_pair = f"{self.board.fullmove_number}. {san}"
            else:
                current_pair += f" {san}"
                self.pgn_tokens.append(current_pair)
                current_pair = None
            self.board.push(move)
            self.uci_history.append(move_str)

        if current_pair:
            self.pgn_tokens.append(current_pair)

        # Rebuild game_state from tokens
        self.game_state = self.base_prompt + (" " + " ".join(self.pgn_tokens) if self.pgn_tokens else "")
        self._persist_history()

    def _persist_history(self):
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            import json
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump({"uci_history": self.uci_history, "pgn_tokens": self.pgn_tokens}, f)
        except Exception as e:
            logging.warning("Failed to persist history: %s", e)

    def _load_history(self) -> bool:
        try:
            import json
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.uci_history = data.get("uci_history", [])
            self.pgn_tokens = data.get("pgn_tokens", [])
            return bool(self.uci_history)
        except Exception:
            return False

    def _resume_from_saved_history_if_possible(self) -> bool:
        # Attempt to rebuild from saved in-memory or persisted history so that
        # board.fen() before applying incoming moves matches the GUI-provided FEN
        saved_moves = list(self.uci_history)
        if not saved_moves and not self._load_history():
            return False
        # Recreate board from startpos using saved moves
        temp = chess.Board()
        try:
            for m in saved_moves:
                temp.push_uci(m)
        except Exception:
            return False
        # If FENs match current board FEN, accept and rebuild state from saved tokens
        if temp.fen() == self.board.fen():
            self.board = temp
            # If no pgn_tokens loaded, rebuild from moves
            if not self.pgn_tokens:
                self._rebuild_tokens_from_moves(saved_moves)
            self.game_state = self.base_prompt + (" " + " ".join(self.pgn_tokens) if self.pgn_tokens else "")
            return True
        return False

    def _rebuild_tokens_from_moves(self, moves: List[str]):
        self.pgn_tokens = []
        temp = chess.Board()
        current_pair: Optional[str] = None
        for m in moves:
            move = chess.Move.from_uci(m)
            san = temp.san(move)
            if current_pair is None:
                current_pair = f"{temp.fullmove_number}. {san}"
            else:
                current_pair += f" {san}"
                self.pgn_tokens.append(current_pair)
                current_pair = None
            temp.push(move)
        if current_pair:
            self.pgn_tokens.append(current_pair)

    def _append_move_number_if_needed(self):
        if self.board.turn == chess.WHITE:
            if self.board.fullmove_number != 1:
                self.game_state += " "
            self.game_state += f"{self.board.fullmove_number}."
        else:
            self.game_state += " "

    def _append_san_to_game_state(self, san: str):
        self.game_state += san

    def _get_best_move(self) -> Tuple[Optional[str], Optional[str]]:
        self._append_move_number_if_needed()
        # Log the exact transcript we are about to send to NanoGPT
        log_prompt_uci("PROMPT", self.board, self.game_state)
        result: LegalMoveResponse = get_legal_move(
            self.player,
            self.board,
            self.game_state,
            player_one=self.board.turn == chess.WHITE,
        )
        if result.move_uci is None:
            if result.is_resignation:
                message = "model resigned (returned game result token)"
            elif result.is_illegal_move:
                message = (
                    f"model produced illegal moves for {result.attempts} attempts"
                )
            else:
                message = "model failed to return a move"
            log_failure(
                message,
                self.board.copy(),
                self.game_state,
                result.attempts,
                result.attempt_history,
            )
            print(f"info string {message}")
            sys.stdout.flush()
            log_prompt_uci("FAILURE", self.board, self.game_state, extra=message, attempts=result.attempt_history)
            return None, None
        self._append_san_to_game_state(result.move_san)
        self.board.push(result.move_uci)
        logging.info(
            "Engine move selected: %s (SAN %s) with transcript length %d",
            result.move_uci.uci(),
            result.move_san,
            len(self.game_state),
        )
        log_prompt_uci("RESULT", self.board, self.game_state, extra=f"bestmove {result.move_uci.uci()} SAN {result.move_san}")
        return result.move_uci.uci(), result.move_san

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            logging.info("RECV: %s", line)
            if not line:
                continue
            tokens = line.split()
            command = tokens[0]

            if command == "uci":
                print(f"id name {self.engine_name}")
                print("id author ChessGPT")
                print("uciok")
                sys.stdout.flush()
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
            elif command == "ucinewgame":
                self.reset()
            elif command == "position":
                self.handle_position(tokens[1:])
            elif command == "go":
                bestmove, _ = self._get_best_move()
                if bestmove is None:
                    print("bestmove 0000")
                else:
                    print(f"bestmove {bestmove}")
                sys.stdout.flush()
            elif command == "quit":
                logging.info("Received quit command")
                break
            # Ignoring other commands (stop, ponder, etc.) for this minimal implementation.


def create_player(opponent: str, gpt_model: str, nanogpt_checkpoint: str):
    if opponent == "gpt":
        player = GPTPlayer(model=gpt_model)
        label = gpt_model
    else:
        player = NanoGptPlayer(model_name=nanogpt_checkpoint)
        label = nanogpt_checkpoint
    return player, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expose ChessGPT as a minimal UCI engine for GUI play."
    )
    parser.add_argument(
        "--opponent",
        choices=["nanogpt", "gpt"],
        default="nanogpt",
        help="Which ChessGPT backend to use.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo-instruct",
        help="OpenAI model name when --opponent=gpt.",
    )
    parser.add_argument(
        "--nanogpt-checkpoint",
        default=DEFAULT_NANOGPT_MODEL,
        help="Checkpoint filename under nanogpt/out for NanoGPT play.",
    )
    parser.add_argument(
        "--engine-name",
        default="ChessGPT",
        help="Name reported to the GUI via the UCI 'id name' command.",
    )
    args = parser.parse_args()

    player, label = create_player(args.opponent, args.model, args.nanogpt_checkpoint)
    engine = ChessGptUciEngine(player=player, engine_name=f"{args.engine_name} ({label})")
    engine.loop()
