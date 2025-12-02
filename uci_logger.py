#!/usr/bin/env python3
"""Proxy Play_CGPT through a light-weight UCI logger that PyChess accepts."""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import threading
from typing import Iterable, List, Optional

DEFAULT_ENGINE_CMD = [
    "/usr/bin/python3",
    "/home/chriskar/chess_gpt_eval/Play_CGPT.py",
    "--opponent",
    "nanogpt",
]
DEFAULT_LOG_PATH = "/home/chriskar/chess_gpt_eval/logs/pychess_traffic.log"
DEFAULT_ENGINE_NAME = "ChessGPT Logger"
DEFAULT_ENGINE_AUTHOR = "ChessGPT"

log_lock = threading.Lock()


def log_line(log_path: str, prefix: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    if not line.endswith("\n"):
        line = f"{line}\n"
    with log_lock, open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {prefix} {line}")


class UciLoggingProxy:
    """Small wrapper that spoofs the UCI handshake and rewrites stdout if needed."""

    def __init__(
        self,
        engine_cmd: Iterable[str],
        log_path: str,
        engine_name: str,
        engine_author: str,
        spoof_handshake: bool = True,
    ) -> None:
        self.engine_cmd: List[str] = list(engine_cmd)
        self.log_path = log_path
        self.engine_name = engine_name
        self.engine_author = engine_author
        self.spoof_handshake = spoof_handshake
        self.stdout_lock = threading.Lock()
        self.stopped = threading.Event()
        self.engine_ready = threading.Event()

        self.engine = subprocess.Popen(
            self.engine_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.stdout_thread = threading.Thread(
            target=self._pump_engine_stdout, daemon=True
        )
        self.stdout_thread.start()

    def run(self) -> None:
        try:
            for raw_line in sys.stdin:
                if not raw_line:
                    continue
                self._handle_gui_command(raw_line)
        finally:
            self.shutdown()

    def _handle_gui_command(self, raw_line: str) -> None:
        log_line(self.log_path, ">>>", raw_line)
        stripped = raw_line.strip()
        if not stripped:
            return

        lower = stripped.lower()
        if lower == "uci":
            self._send_fake_handshake()
        elif lower == "xboard":
            self._write_gui("info string GUI requested xboard, but only UCI is supported")

        self._send_to_engine(raw_line)

    def _send_fake_handshake(self) -> None:
        if not self.spoof_handshake:
            return
        for msg in (
            f"id name {self.engine_name}",
            f"id author {self.engine_author}",
            "uciok",
        ):
            self._write_gui(msg)

    def _send_to_engine(self, data: str) -> None:
        if not self.engine.stdin:
            return
        try:
            self.engine.stdin.write(data)
            if not data.endswith("\n"):
                self.engine.stdin.write("\n")
            self.engine.stdin.flush()
        except BrokenPipeError:
            self._write_gui("info string Backend engine stdin pipe is closed")

    def _pump_engine_stdout(self) -> None:
        if not self.engine.stdout:
            return
        for line in self.engine.stdout:
            if not line:
                continue
            sanitized = self._sanitize_engine_output(line.rstrip("\n"))
            if sanitized is None:
                continue
            self._write_gui(sanitized)
        self.engine_ready.set()

    def _sanitize_engine_output(self, text: str) -> Optional[str]:
        stripped = text.strip()
        if not stripped:
            return ""

        lowered = stripped.lower()
        if lowered == "uciok":
            self.engine_ready.set()
            if self.spoof_handshake:
                return None
        if lowered.startswith("id ") and self.spoof_handshake:
            return None

        passthrough_prefixes = (
            "bestmove",
            "info",
            "option",
            "readyok",
            "id ",
        )
        if lowered.startswith(passthrough_prefixes):
            return stripped

        return f"info string {stripped}"

    def _write_gui(self, text: str) -> None:
        if text is None:
            return
        line = text if text.endswith("\n") else f"{text}\n"
        with self.stdout_lock:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                self.stopped.set()
                return
        log_line(self.log_path, "<<<", line)

    def shutdown(self) -> None:
        if self.stopped.is_set():
            return
        self.stopped.set()
        try:
            if self.engine.stdin and not self.engine.stdin.closed:
                self.engine.stdin.close()
        except Exception:
            pass
        try:
            self.engine.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.engine.kill()
        if self.engine.stdout and not self.engine.stdout.closed:
            self.engine.stdout.close()
        if self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log and proxy a UCI engine")
    parser.add_argument(
        "--engine-cmd",
        nargs="+",
        default=DEFAULT_ENGINE_CMD,
        help="Command used to launch the backend engine",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        help="File used to store the transcript",
    )
    parser.add_argument(
        "--engine-name",
        default=DEFAULT_ENGINE_NAME,
        help="Name reported to the GUI during the UCI handshake",
    )
    parser.add_argument(
        "--engine-author",
        default=DEFAULT_ENGINE_AUTHOR,
        help="Author field reported to the GUI",
    )
    parser.add_argument(
        "--passthrough-handshake",
        action="store_true",
        help="Forward the backend's UCI handshake instead of spoofing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy = UciLoggingProxy(
        engine_cmd=args.engine_cmd,
        log_path=args.log_path,
        engine_name=args.engine_name,
        engine_author=args.engine_author,
        spoof_handshake=not args.passthrough_handshake,
    )
    proxy.run()


if __name__ == "__main__":
    main()
