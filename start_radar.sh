#!/usr/bin/env bash
# =============================================================================
# start_radar.sh —— 雷达站一键启动
#
# 把「source ROS 2 → source 工作空间 → 激活虚拟环境 → 启动节点」这一串动作
# 收成一个命令。每个组件各占一个终端标签，互相独立，Ctrl-C 只停自己。
#
# 用法：
#   ./start_radar.sh              # 交互菜单
#   ./start_radar.sh all          # 相机 + 雷达 + 视觉 + UI 全起
#   ./start_radar.sh ui           # 只开 PyQt5 界面（会激活虚拟环境）
#   ./start_radar.sh camera       # 只开海康相机
#   ./start_radar.sh lidar        # 只开 Livox 雷达
#   ./start_radar.sh vision       # 只开视觉检测
#   ./start_radar.sh map          # 只开 debug_map 可视化
#   ./start_radar.sh shell        # 新开一个终端，停在「环境已就绪」的 shell 里
#   ./start_radar.sh check        # 环境自检（不启动任何东西）
#   ./start_radar.sh stop         # 停掉本脚本启动过的组件（按 pid 文件，不碰其它进程）
#
# 选项：
#   --no-term     不开新终端，全部后台运行，日志写到 log/start_radar/
#   -h | --help   显示本说明
#
# =============================================================================
set -uo pipefail

WS="$HOME/radar_ws"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
VENV="${VENV:-$HOME/radar_venv}"
LOGDIR="$WS/log/start_radar"
RUNDIR="$LOGDIR/run"

USE_TERM=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-term) USE_TERM=0 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        *) ARGS+=("$a") ;;
    esac
done
TARGET="${ARGS[0]:-menu}"

c_info()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
c_ok()    { printf '\033[1;32m  ✔ %s\033[0m\n' "$*"; }
c_warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
c_err()   { printf '\033[1;31m  ✘ %s\033[0m\n' "$*" >&2; }

# 必须在当前 shell 里 source，函数形式才能把环境带进后续命令
load_env() {
    local want_venv="${1:-no}"
    [ -f "$ROS_SETUP" ] || { c_err "找不到 $ROS_SETUP"; return 1; }
    # ROS 的 setup.bash 会引用未定义变量，本脚本开了 set -u，必须先临时关掉
    set +u
    # shellcheck disable=SC1090
    source "$ROS_SETUP" || { c_err "source ROS 失败"; set -u; return 1; }
    if [ -f "$WS/install/setup.bash" ]; then
        # shellcheck disable=SC1091
        source "$WS/install/setup.bash"
    else
        c_warn "install/setup.bash 不存在，先跑 colcon build"
    fi
    if [ "$want_venv" = "venv" ]; then
        if [ -f "$VENV/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$VENV/bin/activate"
        else
            c_err "找不到 $VENV，请检查 VENV 变量"
            set -u
            return 1
        fi
    fi
    set -u
    cd "$WS" || return 1
}

# 给新终端用的环境前缀（子 shell 里要自己 source 一遍）
env_prefix() {
    local need_venv="${1:-no}"
    local p="cd '$WS' && source '$ROS_SETUP' && { [ -f '$WS/install/setup.bash' ] && source '$WS/install/setup.bash' || true; }"
    [ "$need_venv" = "venv" ] && p="$p && source '$VENV/bin/activate'"
    printf '%s' "$p"
}

# radar_spawn <标签> <venv|no> <命令...>
radar_spawn() {
    local title="$1" need_venv="$2"; shift 2
    local cmd="$*"
    local prefix; prefix="$(env_prefix "$need_venv")"
    mkdir -p "$RUNDIR"
    local pidfile="$RUNDIR/$title.pid"
    # 节点放后台并记录真实 PID（$!）；退出码可读，标签保留可回看日志
    local inner="$prefix; $cmd & echo \$! > '$pidfile'; wait \$!; ret=\$?; \
                echo; echo \"[$title] 退出码 \$ret，回车关闭这个标签\"; read -r"

    if [ "$USE_TERM" = "1" ] && command -v gnome-terminal >/dev/null 2>&1; then
        if gnome-terminal --tab --title="$title" -- bash -c "$inner" 2>/dev/null; then
            c_ok "$title → 新终端标签（pid 记在 $pidfile）"
            return 0
        fi
        c_warn "gnome-terminal 起不来，改为后台运行"
    fi

    local log="$LOGDIR/$title-$(date +%H%M%S).log"
    setsid bash -c "$inner" >"$log" 2>&1 &
    c_ok "$title → 后台运行，日志 $log"
}

cmd_check() {
    c_info "环境自检"
    [ -f "$ROS_SETUP" ] && c_ok "ROS 2: $ROS_SETUP" || c_err "缺 ROS 2: $ROS_SETUP"
    [ -d "$WS/install" ] && c_ok "工作空间已编译: $WS/install" || c_err "未编译，先 colcon build"
    if [ -f "$VENV/bin/activate" ]; then
        c_ok "虚拟环境: $VENV ($("$VENV/bin/python3" -V 2>&1))"
    else
        c_err "缺虚拟环境: $VENV"
    fi
    if lsusb 2>/dev/null | grep -qi "hikrobot\|Hikvision"; then
        c_ok "海康相机已识别: $(lsusb | grep -i 'hikrobot\|Hikvision' | head -1 | cut -c1-60)"
    else
        c_warn "未在 lsusb 里看到海康相机（可能没插或没上电）"
    fi
    if ip -4 addr show 2>/dev/null | grep -q "192.168.96.50"; then
        c_ok "雷达网口已配 192.168.96.50"
    else
        c_warn "没有 192.168.96.50 这个地址，雷达可能连不上"
    fi
    if ping -c1 -W1 192.168.96.100 >/dev/null 2>&1; then
        c_ok "雷达在线: 192.168.96.100"
    else
        c_warn "ping 不通 192.168.96.100（雷达未上电或网线未接）"
    fi
}

cmd_stop() {
    c_info "停止本脚本启动的组件"
    local found=0 f name pid
    for f in "$RUNDIR"/*.pid; do
        [ -e "$f" ] || continue
        name="$(basename "$f" .pid)"
        pid="$(cat "$f" 2>/dev/null)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.2
            done
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
            c_ok "已结束: $name (pid $pid)"
            found=1
        else
            c_warn "$name 已不在运行（pid $pid 失效）"
        fi
        rm -f "$f"
    done
    if [ "$found" = 0 ]; then
        c_warn "没有正在运行的组件；如果是手动启动的，请在对应标签里 Ctrl-C"
    fi
    c_warn "终端标签不会自动关闭，右键关掉即可"
}

do_camera() {
    radar_spawn camera no \
        ros2 run hik_camera hik_camera_node --ros-args -p exposure_time:=50000.0 -p gain:=10.0
}
do_lidar() {
    radar_spawn lidar no ros2 launch livox_ros2_driver livox_lidar_launch.py
}
do_vision() {
    # detect.cpp 用相对路径读 ./config/detect_params.yaml 和 model/，必须以 radar_core 为工作目录
    radar_spawn vision no \
        "cd '$WS/src/radar_core' && ros2 run radar_vision radar_detect_node --ros-args -r camera_image:=/image"
}
do_map() {
    radar_spawn map no ros2 run debug_map debug_map
}
do_ui() {
    radar_spawn ui venv python3 camera_ui.py
}

do_shell() {
    c_info "打开就绪终端（ROS 2 + install + 虚拟环境 已 source）"
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --tab --title="radar-ready" -- bash -c \
            "$(env_prefix venv); clear; echo '环境已就绪：ROS 2 + install + radar_venv'; \
             echo \"python3 -> \$(command -v python3)\"; echo; exec bash"
    else
        c_warn "没有图形终端可用；请在当前 shell 执行："
        printf '  source %s\n  source %s/install/setup.bash\n  source %s/bin/activate\n' \
            "$ROS_SETUP" "$WS" "$VENV"
    fi
}

do_menu() {
    printf '\n\033[1m雷达站一键启动\033[0m\n'
    printf '  1) 全部启动（相机 + 雷达 + 视觉 + UI）\n'
    printf '  2) 只开 UI\n'
    printf '  3) 只开相机\n'
    printf '  4) 只开雷达\n'
    printf '  5) 只开视觉检测\n'
    printf '  6) 只开 debug_map\n'
    printf '  7) 打开环境就绪的终端\n'
    printf '  8) 环境自检\n'
    printf '  9) 停止全部\n'
    printf '  q) 退出\n\n'
    printf '选择: '
    local c; read -r c
    case "$c" in
        1) do_camera; do_lidar; do_vision; do_ui ;;
        2) do_ui ;;
        3) do_camera ;;
        4) do_lidar ;;
        5) do_vision ;;
        6) do_map ;;
        7) do_shell ;;
        8) load_env no >/dev/null; cmd_check ;;
        9) cmd_stop ;;
        q|Q|"") exit 0 ;;
        *) c_err "无效选择: $c"; exit 2 ;;
    esac
}

case "$TARGET" in
    menu)      do_menu ;;
    all)       load_env no >/dev/null; cmd_check; echo
               do_camera; do_lidar; do_vision; do_ui
               c_warn "相机/雷达出图需要 10 秒左右，UI 起来后等一下画面" ;;
    ui)        do_ui ;;
    camera)    do_camera ;;
    lidar)     do_lidar ;;
    vision)    do_vision ;;
    map)       do_map ;;
    shell|env) do_shell ;;
    check)     load_env no >/dev/null; cmd_check ;;
    stop)      cmd_stop ;;
    *)         c_err "未知参数: $TARGET（用 -h 看用法）"; exit 2 ;;
esac

