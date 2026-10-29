#!/usr/bin/env python3
"""把 colcon 各包生成的 compile_commands.json 合并成工作空间级的一份。

colcon 只会把编译数据库写到 build/<pkg>/compile_commands.json，
不会生成根目录汇总文件，所以 VS Code 的 C++ 扩展如果只读根目录那一份，
就只能看到最后一次单独构建的那个包，其余包的所有 #include 都会报错。

用法（在仓库根目录执行）：
    python3 tools/merge_compile_commands.py

默认输出到 build/compile_commands.json（即根目录软链接的目标）。
每次 colcon build 之后重跑一次即可。

统计口径：以 build/ 下存在的包目录为全集，没有 compile_commands.json
的包（例如 livox_sdk_vendor 只安装 cmake 配置，不含 C/C++ 编译单元）
按 0 条列出，避免“条目只覆盖 12 个包”被误读成漏构建。
"""

from __future__ import annotations

import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")
OUT = os.path.join(BUILD_DIR, "compile_commands.json")

# 统计表的包名列宽，与 build.py 的结果表保持一致
NAME_WIDTH = 22


def package_databases() -> dict[str, str]:
    """返回 build/ 下每个包的数据库路径；未产生数据库的包映射到空串。

    以构建目录（含 compile_commands.json 或 CMakeCache.txt）作为包的全集，
    这样没有编译单元的包也会出现在统计里，而不是被静默省略。
    """
    found: dict[str, str] = {}
    if not os.path.isdir(BUILD_DIR):
        return found

    for name in sorted(os.listdir(BUILD_DIR)):
        path = os.path.join(BUILD_DIR, name)
        if not os.path.isdir(path):
            continue
        db = os.path.join(path, "compile_commands.json")
        if os.path.isfile(db):
            found[name] = db
        elif os.path.isfile(os.path.join(path, "CMakeCache.txt")):
            found[name] = ""
    return found


def load_pkg_databases() -> tuple[list[dict], dict[str, int]]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stats: dict[str, int] = {}

    for pkg, path in package_databases().items():
        if not path:
            stats[pkg] = 0
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {pkg}: {exc}", file=sys.stderr)
            stats[pkg] = 0
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

    if not missing:
        print(f"include 目录: {len(checked)} 个，全部存在")
        return

    print(f"include 目录: {len(checked)} 个, 不存在: {len(missing)} 个")
    for path in sorted(missing):
        print(f"  [缺] {path}")


def report_stats(stats: dict[str, int]) -> None:
    """打印每包条目数；0 条的包显式标注原因，避免被当成漏构建。"""
    empty = [pkg for pkg, count in stats.items() if count == 0]
    head = f"覆盖 {len(stats) - len(empty)}/{len(stats)} 个包"
    if empty:
        head += f"，{len(empty)} 个包无编译单元（列于末尾，条目为 0）"
    print(head)

    for pkg in sorted(stats, key=lambda name: (-stats[name], name)):
        note = "  <- 无 C/C++ 编译单元（只安装配置/脚本）" if stats[pkg] == 0 else ""
        print(f"  {pkg:<{NAME_WIDTH}} {stats[pkg]:>4}{note}")


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
    report_stats(stats)
    report_missing_includes(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
