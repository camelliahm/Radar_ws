#!/usr/bin/env python3
"""把 colcon 各包生成的 compile_commands.json 合并成工作空间级的一份。

colcon 只会把编译数据库写到 build/<pkg>/compile_commands.json，
不会生成根目录汇总文件，所以 VS Code 的 C++ 扩展如果只读根目录那一份，
就只能看到最后一次单独构建的那个包，其余包的所有 #include 都会报错。

用法（在仓库根目录执行）：
    python3 tools/merge_compile_commands.py

默认输出到 build/compile_commands.json（即根目录软链接的目标）。
每次 colcon build 之后重跑一次即可。
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "compile_commands.json")


def load_pkg_databases() -> tuple[list[dict], dict[str, int]]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stats: dict[str, int] = {}

    for path in sorted(glob(os.path.join(ROOT, "build", "*", "compile_commands.json"))):
        pkg = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {pkg}: {exc}", file=sys.stderr)
            continue

        kept = 0
        for entry in entries:
            key = (entry.get("file", ""), entry.get("directory", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
            kept += 1
        stats[pkg] = kept

    return merged, stats


def include_dirs(entry: dict) -> list[str]:
    cmd = entry.get("command")
    args = entry.get("arguments")
    try:
        tokens = shlex.split(cmd) if args is None else list(args)
    except ValueError:
        return []

    dirs: list[str] = []
    for idx, token in enumerate(tokens):
        if token.startswith("-I") and len(token) > 2:
            dirs.append(token[2:])
        elif token == "-isystem" and idx + 1 < len(tokens):
            dirs.append(tokens[idx + 1])
        elif token.startswith("-isystem") and len(token) > len("-isystem"):
            dirs.append(token[len("-isystem") :])
    return dirs


def report_missing_includes(entries: list[dict]) -> None:
    checked: set[str] = set()
    missing: set[str] = set()
    for entry in entries:
        for path in include_dirs(entry):
            if path in checked:
                continue
            checked.add(path)
            if not os.path.isdir(path):
                missing.add(path)

    print(f"include 目录: {len(checked)} 个, 不存在: {len(missing)} 个")
    for path in sorted(missing):
        print(f"  [缺] {path}")


def main() -> int:
    entries, stats = load_pkg_databases()
    if not entries:
        print("没有找到任何 build/*/compile_commands.json，请先 colcon build。", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"已写入 {os.path.relpath(OUT, ROOT)}: {len(entries)} 条编译命令")
    for pkg, count in stats.items():
        print(f"  {pkg:<22} {count:>4}")
    report_missing_includes(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
