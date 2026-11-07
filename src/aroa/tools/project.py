"""File and shell tools confined to one project folder under PROJECTS_DIR."""

import fnmatch
import re
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from .base import ToolContext, ToolError

MAX_OUTPUT = 12000
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ToolError(f"{relative!r} is outside the project.")
    return path


def _cap(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n[truncated at {MAX_OUTPUT} characters]"


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def agents_md(root: Path) -> str:
    for name in ("AGENTS.md", "CLAUDE.md", "README.md"):
        candidate = root / name
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")[:12000]
    return ""


SHELL_SYNTAX = re.compile(r"[;&|<>`$]|\(|\)")
FIND_ACTIONS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"}


def check_command(command: str, allowed: str) -> list[str]:
    """The argv of a command the folder tools may run, or a ToolError saying why not.

    The first word must be on the allowlist, nothing may chain or redirect (the command never goes
    through a shell), paths cannot climb out of the folder, and `find` cannot execute or delete."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"Cannot parse the command: {exc}") from exc
    if not argv:
        raise ToolError("Empty command.")
    if argv[0] not in allowed.split():
        raise ToolError(f"'{argv[0]}' is not an allowed command. Allowed: {allowed}.")
    if SHELL_SYNTAX.search(command):
        raise ToolError("Pipes, redirections, substitutions and chaining are not allowed: one plain command.")
    for arg in argv[1:]:
        if arg.startswith(("/", "~")) or ".." in arg.split("/"):
            raise ToolError(f"Paths must stay inside the project folder: {arg!r}.")
        if argv[0] == "find" and arg in FIND_ACTIONS:
            raise ToolError(f"find {arg} is not allowed.")
        if argv[0] == "sed" and arg.startswith("-i"):
            raise ToolError("sed -i is not allowed; use edit_file to change a file.")
    return argv


def make_tools(ctx: ToolContext) -> list:
    @tool
    def list_files(pattern: str = "*") -> str:
        """List the project's files; pattern is a glob like "*.py" or "src/**"."""
        root = ctx.project_root
        names = [str(p.relative_to(root)) for p in iter_files(root)]
        matched = [n for n in names if fnmatch.fnmatch(n, pattern) or fnmatch.fnmatch(Path(n).name, pattern)]
        return _cap("\n".join(matched[:500])) if matched else f"No files match {pattern!r}."

    @tool
    def read_file(path: str, offset: int = 1, limit: int = 300) -> str:
        """Read a text file with line numbers, starting at line `offset`, up to `limit` lines."""
        target = safe_path(ctx.project_root, path)
        if not target.is_file():
            raise ToolError(f"{path} does not exist.")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[max(0, offset - 1) : max(0, offset - 1) + limit]
        if not selected:
            return f"{path} has {len(lines)} lines; nothing at offset {offset}."
        return _cap("\n".join(f"{i:>5}\t{line}" for i, line in enumerate(selected, start=max(1, offset))))

    @tool
    def grep(pattern: str, glob: str = "*") -> str:
        """Search a regular expression in the project's files. Returns file:line: text."""
        root = ctx.project_root
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Bad regular expression: {exc}") from exc
        hits = []
        for file in iter_files(root):
            name = str(file.relative_to(root))
            if not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(file.name, glob)):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            hits += [f"{name}:{i}: {line}" for i, line in enumerate(text.splitlines(), 1) if regex.search(line)]
            if len(hits) > 200:
                break
        return _cap("\n".join(hits[:200])) if hits else f"No matches for {pattern!r}."

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file. Keep it short; extend long files with edit_file."""
        target = safe_path(ctx.project_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {path} ({len(content.splitlines())} lines)."

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a file. `old` must appear exactly once."""
        target = safe_path(ctx.project_root, path)
        if not target.is_file():
            raise ToolError(f"{path} does not exist.")
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count != 1:
            raise ToolError(f"`old` appears {count} times in {path}; it must appear exactly once.")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Edited {path}."

    @tool
    def run_command(command: str) -> str:
        """Run one read-only command inside the project folder (ls, cat, head, tail, wc, grep, find, sort,
        uniq, diff, cut, tr, stat, file). No pipes, redirections or chaining; paths stay inside the folder."""
        root = ctx.project_root
        argv = check_command(command, ctx.settings.allowed_commands)
        try:
            done = subprocess.run(
                argv,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=ctx.settings.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"Command timed out after {ctx.settings.command_timeout_seconds}s.") from exc
        output = (done.stdout + ("\n" + done.stderr if done.stderr else "")).strip()
        if done.returncode != 0:
            raise ToolError(f"exit {done.returncode}\n{_cap(output)}")
        return _cap(output) or "(no output)"

    @tool
    def redact_file(path: str) -> str:
        """Write a copy of a text file with every name and identifier blacked out, for sharing outside the firm.
        The copy is saved next to the original as <name>.redacted<extension>."""
        if ctx.anonymizer is None:
            raise ToolError("Redaction is not available: privacy mode is off.")
        target = safe_path(ctx.project_root, path)
        if not target.is_file():
            raise ToolError(f"{path} does not exist.")
        if target.suffix.lower() == ".pdf":
            raise ToolError("Only text files can be redacted; convert the PDF to text first.")
        text = target.read_text(encoding="utf-8", errors="replace")
        redacted = ctx.anonymizer.redact(text)
        out = target.with_name(f"{target.stem}.redacted{target.suffix}")
        out.write_text(redacted, encoding="utf-8")
        boxes = redacted.count("████████")
        return f"Wrote {out.relative_to(ctx.project_root)} with {boxes} redactions."

    return [list_files, read_file, grep, write_file, edit_file, run_command, redact_file]
