"""Client-side backend for Anthropic's memory tool (memory_20250818).

Return strings follow the reference behavior in the memory tool docs.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class MemoryFiles:
    def __init__(self, root: Path, fresh: bool = True) -> None:
        # A new run starts empty; a later session keeps what an earlier one wrote.
        if fresh and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def _path(self, p: str) -> Path:
        if not p or not p.startswith("/memories"):
            raise ValueError(f"The path {p} is outside /memories")
        real = (self.root / p.removeprefix("/memories").lstrip("/")).resolve()
        real.relative_to(self.root)  # raises on traversal
        return real

    def run(self, inp: dict) -> tuple[str, bool]:
        try:
            return self._run(inp)
        except ValueError as e:
            return f"Error: {e}", True

    def _run(self, inp: dict) -> tuple[str, bool]:
        cmd = inp.get("command")
        if cmd == "view":
            path = self._path(inp["path"])
            if path.is_dir():
                lines = [f"{_size(path)}\t{inp['path'].rstrip('/') or '/memories'}"]
                for child in sorted(path.rglob("*")):
                    rel = child.relative_to(self.root)
                    if len(rel.parts) > 2 or any(part.startswith(".") for part in rel.parts):
                        continue
                    lines.append(f"{_size(child)}\t/memories/{rel}")
                return (
                    f"Here're the files and directories up to 2 levels deep in {inp['path']}, "
                    "excluding hidden items and node_modules:\n" + "\n".join(lines)
                ), False
            if not path.is_file():
                return f"The path {inp['path']} does not exist. Please provide a valid path.", True
            lines = path.read_text().split("\n")
            start, end = 1, len(lines)
            if inp.get("view_range"):
                start, end = inp["view_range"]
                end = len(lines) if end == -1 else end
            body = "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))
            return f"Here's the content of {inp['path']} with line numbers:\n{body}", False
        if cmd == "create":
            path = self._path(inp["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inp.get("file_text", ""))  # create overwrites, as the tool description says
            return f"File created successfully at: {inp['path']}", False
        if cmd == "str_replace":
            path = self._path(inp["path"])
            if not path.is_file():
                return f"Error: The path {inp['path']} does not exist. Please provide a valid path.", True
            text, old = path.read_text(), inp["old_str"]
            n = text.count(old)
            if n == 0:
                return f"No replacement was performed, old_str `{old}` did not appear verbatim in {inp['path']}.", True
            if n > 1:
                lines = [i + 1 for i, line in enumerate(text.split("\n")) if old.split("\n")[0] in line]
                return (f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines: "
                        f"{lines}. Please ensure it is unique"), True
            path.write_text(text.replace(old, inp.get("new_str", ""), 1))
            return "The memory file has been edited.", False
        if cmd == "insert":
            path = self._path(inp["path"])
            if not path.is_file():
                return f"Error: The path {inp['path']} does not exist", True
            lines = path.read_text().split("\n")
            at = inp["insert_line"]
            if at < 0 or at > len(lines):
                return (f"Error: Invalid `insert_line` parameter: {at}. It should be within the range of "
                        f"lines of the file: [0, {len(lines)}]"), True
            lines.insert(at, inp["insert_text"].rstrip("\n"))
            path.write_text("\n".join(lines))
            return f"The file {inp['path']} has been edited.", False
        if cmd == "delete":
            path = self._path(inp["path"])
            if path == self.root:
                return "Error: cannot delete the /memories directory", True
            if not path.exists():
                return f"Error: The path {inp['path']} does not exist", True
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            return f"Successfully deleted {inp['path']}", False
        if cmd == "rename":
            old, new = self._path(inp["old_path"]), self._path(inp["new_path"])
            if old == self.root:
                return "Error: cannot rename the /memories directory", True
            if not old.exists():
                return f"Error: The path {inp['old_path']} does not exist", True
            if new.exists():
                return f"Error: The destination {inp['new_path']} already exists", True
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            return f"Successfully renamed {inp['old_path']} to {inp['new_path']}", False
        return f"Error: unknown command {cmd}", True


def _size(p: Path) -> str:
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return f"{n / 1024:.1f}K" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f}M"
