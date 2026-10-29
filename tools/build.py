#!/usr/bin/env python3
"""radar_ws 一键构建脚本。

封装 colcon build 的常规流程，避免每次手敲环境变量和冗长的 cmake 参数：

  1. 校验工作空间、ROS 环境；
  2. source /opt/ros/<distro>/setup.bash 与 install/setup.bash 后执行 colcon build；
  3. 解析 log/latest_build 得到每个包的耗时与结果，输出汇总表；
  4. 构建失败时从各包的 stderr/stdout 日志中提取报错行并高亮；
  5. 构建成功后调用 tools/merge_compile_commands.py 更新根目录 compile_commands.json。

用法（在任意目录均可执行）：
    python3 tools/build.py                  # 全量增量构建（默认 Release，并行）
    python3 tools/build.py -p radar_vision  # 只构建指定包（可逗号分隔多个）
    python3 tools/build.py -c               # 清 build/ 后重建（强制重新 CMake 配置）
    python3 tools/build.py --clean-all      # 清 build/ 与 install/ 后重建（依赖库版本变了才需要）
    python3 tools/build.py --debug          # 使用 Debug 构建类型
    python3 tools/build.py --no-merge       # 跳过 compile_commands.json 合并
    python3 tools/build.py -j 8             # 限制并行度为 8
    python3 tools/build.py -v               # 透传 colcon 原始输出，不做精简

退出码：0 表示全部成功，非 0 表示存在构建失败的包。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 工作空间根目录：本文件位于 <root>/tools/build.py
ROOT = Path(__file__).resolve().parent.parent

# colcon 构建产物目录，均被根 .gitignore 忽略
BUILD_DIR = ROOT / "build"
INSTALL_DIR = ROOT / "install"
LOG_DIR = ROOT / "log"

# 默认 ROS 发行版；可用 --ros-distro 或环境变量 ROS_DISTRO 覆盖
DEFAULT_ROS_DISTRO = "jazzy"

# 构建失败后，从包日志中提取报错行时最多输出的行数，避免刷屏
MAX_ERROR_LINES = 40

# 判定为“报错行”的特征。构建机的 make/cmake 会输出中文，因此中英文都要覆盖：
#   gcc/g++/nvcc : error: / fatal error: / 错误：
#   CMake        : Error N / 找不到 / Could not find
#   make         : *** No rule / *** 没有规则 / 错误 N
#   链接器       : undefined reference to / cannot find -l / 未定义的引用
ERROR_PATTERNS = re.compile(
    r"(?:error:|fatal error:|错误：|Error\s+\d+"
    r"|undefined reference to|未定义的引用|cannot find -l"
    r"|No rule to make target|没有规则可制作目标"
    r"|Could not find|找不到"
    r"|\*\*\*.*(?:stop|停止)|\*\*\*)",
    re.IGNORECASE,
)

# 纯警告行不算错误，即使命中了上面的模式也剔除，避免掩盖真正的失败原因
NOISE_PATTERNS = re.compile(r"\b(?:warning|警告)\b", re.IGNORECASE)

# make 只在真正编译源文件时输出这类行；用它区分“编过了”和“增量空转”
COMPILE_MARKERS = re.compile(r"Building (?:CXX|C|CUDA) object")

# 低于该耗时且没有任何编译动作，才提示“增量空转”：接口包即使不编译，
# 也要跑 rosidl 生成与 symlink 安装，耗时可达分钟级，不能一并归为空转。
IDLE_HINT_SECONDS = 10.0


def color(text: str, code: str) -> str:
    """在终端支持颜色时给文本着色；输出被重定向时自动降级为纯文本。"""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color(text, "32")


def red(text: str) -> str:
    return color(text, "31")


def yellow(text: str) -> str:
    return color(text, "33")


def bold(text: str) -> str:
    return color(text, "1")


def human_time(seconds: float) -> str:
    """把秒数格式化成 12.3s 或 2m04s 这类易读形式。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def step(message: str) -> None:
    """打印带序号的阶段标题。"""
    print(f"\n{bold('==>')} {message}")


def warn(message: str) -> None:
    print(f"{yellow('[warn]')} {message}")


def fatal(message: str) -> int:
    print(f"{red('[error]')} {message}", file=sys.stderr)
    return 1


@dataclass
class PkgResult:
    """单个包在本次构建中的结果。"""

    name: str
    rc: int | str | None = None
    duration: float = 0.0
    log_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return self.rc == 0 or self.rc == "0"

    @property
    def rc_display(self) -> str:
        """rc 可能是数字，也可能是 'SIGINT' 这类信号名，统一成可打印文本。"""
        return "-" if self.rc is None else str(self.rc)


@dataclass
class Summary:
    """整次构建的汇总数据。"""

    packages: list[PkgResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def failed(self) -> list[PkgResult]:
        return [p for p in self.packages if not p.ok]

    @property
    def succeeded(self) -> list[PkgResult]:
        return [p for p in self.packages if p.ok]


def find_ros_setup(distro: str) -> Path | None:
    """定位指定 ROS 发行版的 setup.bash。"""
    candidate = Path("/opt/ros") / distro / "setup.bash"
    return candidate if candidate.is_file() else None


def colcon_available() -> bool:
    return shutil.which("colcon") is not None


def clean_build(remove_install: bool = False) -> None:
    """清理构建缓存，用于全量重建。

    默认只清 build/：它存放 CMake 配置结果与链接规则，是版本不一致时
    最需要重建的一层；install/ 里的产物只要源码与依赖没变就仍然有效，
    清掉它意味着所有包都要重新安装，徒增耗时。

    remove_install=True 时连 install/ 一起清，适用于依赖库版本变更、
    .so 已指向不存在路径的情况。两个目录都被 .gitignore 忽略，删除不影响源码。
    """
    targets = [BUILD_DIR]
    if remove_install:
        targets.append(INSTALL_DIR)

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            print(f"  已删除 {target.relative_to(ROOT)}/")
        else:
            print(f"  跳过 {target.relative_to(ROOT)}/（不存在）")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="radar_ws 一键构建：source 环境 -> colcon build -> 汇总结果 -> 合并编译数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--packages",
        metavar="PKG[,PKG...]",
        help="只构建指定包，逗号分隔；留空表示全量构建",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="构建前删除 build/，强制重新执行 CMake 配置",
    )
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="构建前删除 build/ 与 install/（依赖库版本变更时使用，耗时更长）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="使用 Debug 构建类型（默认 Release）",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="并行构建的包数量（默认等于 CPU 核数 %(default)s）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="透传 colcon 原始输出，不做精简",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="构建成功后不合并 compile_commands.json",
    )
    parser.add_argument(
        "--ros-distro",
        default=os.environ.get("ROS_DISTRO", DEFAULT_ROS_DISTRO),
        help="ROS 发行版名称（默认 %(default)s）",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个包失败后仍继续构建其余包（默认 colcon 已如此，此处仅作显式说明）",
    )
    return parser.parse_args(argv)


def system_python() -> str:
    """返回系统 Python 解释器路径，供 CMake 配置阶段使用。

    ament_cmake 与 rosidl 的 .cmake 脚本会在配置阶段把 Python 解释器路径
    写死进 execute_process()，之后调用时使用的是绝对路径，因此光调 PATH
    没有用。当前环境若在 PATH 上安装了另一套 Python（不会带 apt 装的
    catkin_pkg、em 等模块），必须在此处显式指定系统解释器。
    """
    for candidate in ("/usr/bin/python3.12", "/usr/bin/python3"):
        if Path(candidate).is_file():
            return candidate
    return "/usr/bin/python3"


def build_command(args: argparse.Namespace) -> list[str]:
    """拼装 colcon build 命令行。

    --symlink-install 让 Python/launch/config 以软链接安装，改完无需重新构建；
    --event-handlers console_cohesion+ 把同一个包的输出聚在一起，便于定位报错。
    Python3_EXECUTABLE 显式指向系统解释器，避免配置阶段选中外部 Python。
    """
    cmd = ["colcon", "build"]

    install_type = "Debug" if args.debug else "Release"
    cmd += [
        "--symlink-install",
        "--event-handlers",
        "console_cohesion+",
        "--cmake-args",
        f"-DCMAKE_BUILD_TYPE={install_type}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
        f"-DPython3_EXECUTABLE={system_python()}",
    ]

    if args.jobs and args.jobs > 0:
        cmd += ["--parallel-workers", str(args.jobs)]

    if args.packages:
        names = [p.strip() for p in args.packages.split(",") if p.strip()]
        if names:
            cmd += ["--packages-select", *names]

    return cmd


def shell_command(cmd: list[str], args: argparse.Namespace) -> list[str]:
    """把 colcon 命令包装成“先 source 两个 setup.bash，再执行”的 bash 脚本。

    通过 bash -c 在子进程内 source，避免污染当前 Python 进程的环境；
    set -e 让 source 失败时直接退出，不至于拿脏环境去构建。

    关键一步是把 /usr/bin 前置到 PATH：ROS 的 setup.bash 会在 PATH 上放
    python3，如果当前 shell 里另有更高优先级的 Python（如独立安装的 3.13），
    构建过程中调用的 python3 就会换成那个解释器，从而 import 不到
    catkin_pkg、em 等 apt 安装的模块。固定用系统解释器可消除这一差异。
    """
    setup = find_ros_setup(args.ros_distro)
    prelude = ['export PATH="/usr/bin:/bin:$PATH"']  # 让 python3 解析到系统解释器
    if setup:
        prelude.append(f'source "{setup}"')
    if (INSTALL_DIR / "setup.bash").is_file():
        prelude.append(f'source "{INSTALL_DIR / "setup.bash"}"')

    parts = ["set -e", *prelude, " ".join(cmd)]
    return ["bash", "-c", " && ".join(parts)]


def run_build(cmd: list[str], args: argparse.Namespace) -> int:
    """执行构建，返回 colcon 的退出码；-v 时直接透传输出。"""
    full = shell_command(cmd, args)

    if args.verbose:
        return subprocess.call(full, cwd=ROOT)

    # 精简模式：只保留 colcon 的 Starting/Finished/Summary 等关键行
    proc = subprocess.Popen(
        full,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        if re.match(r"^(Starting >>>|Finished <<<|Aborted <<<|Summary:|Failed\s)", stripped.strip()):
            print("  " + stripped.strip())
    return proc.wait()


def latest_build_dir() -> Path | None:
    """找到最近一次构建的日志目录（log/latest_build 是 colcon 维护的软链接）。"""
    link = LOG_DIR / "latest_build"
    if link.is_symlink() or link.is_dir():
        resolved = link.resolve()
        if resolved.is_dir():
            return resolved

    # 兜底：按目录名（含时间戳）排序取最新的一个
    candidates = sorted(p for p in LOG_DIR.glob("build_*") if p.is_dir())
    return candidates[-1] if candidates else None


def normalize_rc(raw: str) -> int | str:
    """把 events.log 里的 rc 字段归一化：数字保持 int，信号名保持原字符串。"""
    value = raw.strip().strip("'\"")
    try:
        return int(value)
    except ValueError:
        return value


def parse_event_log(events: Path) -> tuple[list[PkgResult], float]:
    """从 events.log 解析每个包的 returncode 与耗时。

    colcon 的 events.log 是逐行 Python dict，我们只关心两行关键字：
      (JobStarted ... 'start_time': <epoch>)   -> 记录起始时间
      (JobEnded ... 'identifier': 'pkg', 'rc': 0) -> 记录结束时间与返回码
    时间取该行行首的 [秒数]，表示相对构建开始的偏移量，精度足够做汇总。
    """
    started: dict[str, float] = {}
    results: dict[str, PkgResult] = {}
    last_ts = 0.0

    # 行格式：[76.265391] (kalman_filter) JobEnded: {'identifier': 'kalman_filter', 'rc': 0}
    # rc 可能是数字、带引号的数字，也可能是 'SIGINT' 这类信号名，因此按任意值捕获。
    # 时间戳是相对构建开始的偏移秒数，用它做耗时统计无需解析 epoch。
    line_re = re.compile(r"^\[(\d+\.\d+)\]")
    job_started_re = re.compile(r"^\[[\d.]+\] \(([\w-]+)\) JobStarted:")
    job_ended_re = re.compile(r"^\[[\d.]+\] \(([\w-]+)\) JobEnded:.*'rc': ([^,}]+)")

    with events.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            ts_match = line_re.match(raw)
            if ts_match:
                last_ts = float(ts_match.group(1))

            started_match = job_started_re.match(raw)
            if started_match:
                started[started_match.group(1)] = last_ts
                continue

            ended_match = job_ended_re.match(raw)
            if ended_match:
                name = ended_match.group(1)
                rc = normalize_rc(ended_match.group(2))
                begin = started.get(name, last_ts)
                results[name] = PkgResult(name=name, rc=rc, duration=max(last_ts - begin, 0.0))

    return list(results.values()), last_ts


def collect_summary(log_path: Path) -> Summary:
    """读取日志目录得到构建汇总；解析不到时返回空汇总而不是报错。"""
    events = log_path / "events.log"
    summary = Summary()
    if not events.is_file():
        return summary

    packages, total = parse_event_log(events)
    summary.packages = sorted(packages, key=lambda p: p.duration, reverse=True)
    summary.total_time = total

    # 为每个包挂上自己的日志子目录，失败时用于提取报错
    for pkg in summary.packages:
        pkg_dir = log_path / pkg.name
        if pkg_dir.is_dir():
            pkg.log_dir = pkg_dir
    return summary


def extract_errors(pkg: PkgResult) -> list[str]:
    """从单个包的日志里筛出报错行，去重后返回。

    优先看 stderr.log（编译器与 make 的错误主要写在这里）；
    若其中没有任何命中，再退到合并流和 stdout，兼容把错误打到标准输出的工具。
    """
    if pkg.log_dir is None:
        return []

    for name in ("stderr.log", "stdout_stderr.log", "stdout.log"):
        log_file = pkg.log_dir / name
        if not log_file.is_file():
            continue
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = [
            line.strip()
            for line in content.splitlines()
            if ERROR_PATTERNS.search(line) and not NOISE_PATTERNS.search(line)
        ]
        if not lines:
            continue

        # 保持原始顺序去重，避免同一报错在多个日志里重复刷屏
        seen: set[str] = set()
        unique = [ln for ln in lines if not (ln in seen or seen.add(ln))]
        return unique[:MAX_ERROR_LINES]

    return []


def print_summary(summary: Summary) -> None:
    """打印构建结果表格与失败详情。"""
    if not summary.packages:
        warn("未能解析构建日志，跳过结果汇总")
        return

    step("构建结果")
    print(f"  {'包名':<22}{'状态':<8}{'耗时':>8}")
    print(f"  {'-' * 36}")
    for pkg in summary.packages:
        status = green("OK") if pkg.ok else red("FAIL")
        print(f"  {pkg.name:<22}{status:<8}{human_time(pkg.duration):>8}")

    ok_count = len(summary.succeeded)
    total = len(summary.packages)
    print(f"  {'-' * 36}")
    line = f"  合计 {total} 个包，成功 {ok_count} 个，耗时 {human_time(summary.total_time)}"
    print(green(line) if not summary.failed else line)

    for pkg in summary.failed:
        print(f"\n{red('--- ' + pkg.name + ' 构建失败 (rc=' + pkg.rc_display + ') ---')}")
        errors = extract_errors(pkg)
        if errors:
            for line in errors:
                print(f"  {line}")
        else:
            location = pkg.log_dir or LOG_DIR
            print(f"  未自动提取到报错行，请查看 {location}")


def compiled_targets(log_path: Path | None) -> int:
    """统计本次构建真正发生的编译动作数。

    0 表示 colcon 只是把所有包走了一遍、没有任何源文件需要重编（增量空转），
    此时耗时极短是正常的，并不代表漏构建了某些包。
    """
    if log_path is None:
        return 0

    count = 0
    for log_file in log_path.glob("*/stdout.log"):
        try:
            with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                count += sum(1 for line in fh if COMPILE_MARKERS.search(line))
        except OSError:
            continue
    return count


def merge_compile_commands() -> bool:
    """调用现有工具脚本合并各包的编译数据库。"""
    script = ROOT / "tools" / "merge_compile_commands.py"
    if not script.is_file():
        warn(f"未找到 {script.relative_to(ROOT)}，跳过合并")
        return False

    step("合并编译数据库")
    # 合并脚本只用到标准库，任何 Python 3 都可执行，沿用当前解释器即可
    proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, text=True)
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    began = time.time()

    # --- 前置校验 ---
    if not (ROOT / "src").is_dir():
        return fatal(f"{ROOT} 下没有 src/，请确认脚本位于工作空间内")

    if not colcon_available():
        return fatal("未找到 colcon，请先安装：sudo apt install python3-colcon-common-extensions")

    setup = find_ros_setup(args.ros_distro)
    if setup is None:
        return fatal(f"未找到 /opt/ros/{args.ros_distro}/setup.bash，请用 --ros-distro 指定发行版")
    if not os.environ.get("ROS_DISTRO"):
        warn(f"当前 shell 未 source ROS 环境，脚本将自动 source {setup}")

    build_type = "Debug" if args.debug else "Release"
    scope = args.packages or "全部包"
    print(bold(f"radar_ws 构建  |  {build_type}  |  {scope}  |  并行 {args.jobs}"))

    # --- 可选清理 ---
    if args.clean or args.clean_all:
        step("清理构建缓存")
        clean_build(remove_install=args.clean_all)

    # --- 执行构建 ---
    cmd = build_command(args)
    step("开始构建")
    print(f"  $ {' '.join(cmd)}")
    started = time.time()
    rc = run_build(cmd, args)
    elapsed = time.time() - started

    # --- 汇总（优先解析日志，日志缺失时用 colcon 退出码兜底） ---
    log_path = latest_build_dir()
    summary = collect_summary(log_path) if log_path else Summary()

    if not summary.packages:
        # 没有可解析的日志：只能用总退出码给结论
        step("构建结果")
        if rc == 0:
            print(green(f"  构建成功，耗时 {human_time(elapsed)}"))
        else:
            print(red(f"  构建失败 (rc={rc})，耗时 {human_time(elapsed)}"))
            if log_path:
                print(f"  日志目录：{log_path}")
    else:
        summary.total_time = summary.total_time or elapsed
        print_summary(summary)

    # 说明“为什么这么快就全 OK”：纯增量空转，不是漏包
    if rc == 0 and elapsed < IDLE_HINT_SECONDS and compiled_targets(log_path) == 0:
        print(
            f"\n{yellow('[info]')} 本次未重新编译任何源文件（增量空转），包数并未减少；"
            "需要全量验证请加 -c"
        )

    # --- 合并编译数据库：仅在全量/含接口包的构建后执行 ---
    if rc == 0 and not args.no_merge:
        merge_compile_commands()
    elif rc != 0:
        print(f"\n{yellow('[skip]')} 构建未全部通过，跳过 compile_commands.json 合并")

    print(f"\n总耗时 {human_time(time.time() - began)}")

    if rc == 0 and not summary.failed:
        print(green("构建完成，无错误。"))
        print(f"提示：新开终端后执行 source {INSTALL_DIR / 'setup.bash'} 以加载工作空间")
        return 0

    print(red("构建未全部通过，请根据上方报错修复后重试。"))
    return rc if rc != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
