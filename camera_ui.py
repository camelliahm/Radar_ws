import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter as ParamMsg
from rcl_interfaces.msg import ParameterValue, ParameterType
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                              QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
                              QGridLayout, QPushButton, QCheckBox, QSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import QTimer, Qt
import numpy as np


# ===== 柔和白色主题 =====
DARK_BG = "#f2f2f2"
PANEL_BG = "#ffffff"
BORDER = "#e0e0e0"
TEXT = "#2c2c2c"
TEXT_DIM = "#9e9e9e"
GREEN = "#2e7d32"
RED = "#c62828"
YELLOW = "#b8860b"
BLUE = "#1976d2"

FONT_PANEL_TITLE = 26
FONT_CONN = 8
FONT_STATUS = 22
FONT_BODY = 24
FONT_SMALL = 20
FONT_BIG = 32

SPINBOX_STYLE = f"""
    QSpinBox {{
        color: {YELLOW};
        background-color: #ffffff;
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 10px;
        padding-right: 40px;
        font-size: {FONT_BODY}px;
        min-height: 48px;
    }}
    QSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 32px;
        background-color: #e8e8e8;
        border-left: 1px solid {BORDER};
        border-top-right-radius: 3px;
    }}
    QSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 32px;
        background-color: #e8e8e8;
        border-left: 1px solid {BORDER};
        border-bottom-right-radius: 3px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
        background-color: #d8d8d8;
    }}
    QSpinBox::up-arrow {{
        width: 0; height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-bottom: 8px solid {TEXT};
    }}
    QSpinBox::down-arrow {{
        width: 0; height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 8px solid {TEXT};
    }}
"""


class RosNode(Node):
    def __init__(self):
        super().__init__('camera_ui_node')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_cam_time = 0.0
        self.frame_count = 0
        self.fps_window_start = time.time()
        self.current_fps = 0.0
        self.subscription = self.create_subscription(
            Image, '/image', self.image_callback, 10)

        self.latest_lidar_time = 0.0
        self.latest_lidar_points = None
        self.lidar_history = []       # 累积多帧点云
        self.max_history = 3          # 默认保留 3 帧（约 0.3s）
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/livox/lidar', self.lidar_callback, 10)

        # kalman_filter 节点的参数服务，用于读写己方颜色
        self.kf_client = self.create_client(
            SetParameters, '/kalman_filter_node/set_parameters')
        self.kf_get_client = self.create_client(
            GetParameters, '/kalman_filter_node/get_parameters')

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.last_cam_time = time.time()
            self.frame_count += 1
            now = time.time()
            if now - self.fps_window_start >= 1.0:
                self.current_fps = self.frame_count / (now - self.fps_window_start)
                self.frame_count = 0
                self.fps_window_start = now
        except Exception as e:
            self.get_logger().error(f'转换图像失败: {e}')

    def lidar_callback(self, msg):
        self.latest_lidar_time = time.time()
        try:
            pts = list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True))
            if len(pts) == 0:
                return
            arr = np.array([[float(p[0]), float(p[1]), float(p[2])]
                            for p in pts], dtype=np.float32)
            if len(arr) > 4000:
                idx = np.random.choice(len(arr), 4000, replace=False)
                arr = arr[idx]

            # 累积历史帧
            self.lidar_history.append(arr)
            while len(self.lidar_history) > self.max_history:
                self.lidar_history.pop(0)

            all_pts = np.vstack(self.lidar_history)
            if len(all_pts) > 20000:
                idx = np.random.choice(len(all_pts), 20000, replace=False)
                all_pts = all_pts[idx]
            self.latest_lidar_points = all_pts
        except Exception as e:
            self.get_logger().error(f'点云解析失败: {e}')

    def set_max_history(self, n):
        self.max_history = max(1, int(n))
        while len(self.lidar_history) > self.max_history:
            self.lidar_history.pop(0)

    def camera_connected(self):
        return (time.time() - self.last_cam_time) < 1.0

    def lidar_connected(self):
        return (time.time() - self.latest_lidar_time) < 1.0

    def set_camera_params(self, exposure, gain):
        client = self.create_client(SetParameters, '/hik_camera/set_parameters')
        if not client.wait_for_service(timeout_sec=1.0):
            return False
        req = SetParameters.Request()
        req.parameters = [
            ParamMsg(name='exposure_time',
                     value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                          double_value=float(exposure))),
            ParamMsg(name='gain',
                     value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                          double_value=float(gain))),
        ]
        client.call_async(req)
        return True

    def set_self_color(self, color):
        """写入 kalman_filter_node 的 self_color 参数，0 蓝 2 红"""
        if not self.kf_client.service_is_ready():
            # 服务刚启动时可能尚未发现，短暂等待一次
            self.kf_client.wait_for_service(timeout_sec=0.3)
        if not self.kf_client.service_is_ready():
            return False, 'kalman_filter 节点未运行'
        req = SetParameters.Request()
        req.parameters = [
            ParamMsg(name='self_color',
                     value=ParameterValue(
                         type=ParameterType.PARAMETER_INTEGER,
                         integer_value=int(color))),
        ]
        future = self.kf_client.call_async(req)
        # 参数服务应毫秒级返回，超时即视为失败
        deadline = time.time() + 0.5
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            return False, '参数服务超时'
        results = future.result().results
        if not results or not results[0].successful:
            reason = results[0].reason if results else '无返回'
            return False, f'拒绝：{reason}'
        return True, ''

    def get_self_color(self):
        """回读节点当前己方颜色，失败返回 None"""
        if not self.kf_get_client.service_is_ready():
            return None
        req = GetParameters.Request()
        req.names = ['self_color']
        future = self.kf_get_client.call_async(req)
        deadline = time.time() + 0.3
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            return None
        values = future.result().values
        if not values:
            return None
        if values[0].type != ParameterType.PARAMETER_INTEGER:
            return None
        return int(values[0].integer_value)


class StatusDot(QWidget):
    def __init__(self, text, initial=False):
        super().__init__()
        self.text = text
        self.connected = initial
        self.setFixedHeight(28)

    def set_connected(self, ok):
        if self.connected != ok:
            self.connected = ok
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(GREEN) if self.connected else QColor(RED)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 8, 12, 12)
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Ubuntu", FONT_CONN))
        painter.drawText(24, 20, self.text)


class CameraPanel(QFrame):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER};")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("图像显示")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: {FONT_PANEL_TITLE}px; font-weight: bold;")
        layout.addWidget(title)

        # ===== 己方阵营选择 =====
        color_row = QWidget()
        ccl = QHBoxLayout(color_row)
        ccl.setContentsMargins(0, 0, 0, 0)
        ccl.setSpacing(12)

        side_l = QLabel("己方阵营")
        side_l.setStyleSheet(f"color: {TEXT}; font-size: {FONT_BODY}px;")
        ccl.addWidget(side_l)

        self.blue_btn = QPushButton("蓝方")
        self.blue_btn.setMinimumHeight(52)
        self.blue_btn.setCursor(Qt.PointingHandCursor)
        self.blue_btn.clicked.connect(lambda: self.set_side(0))
        ccl.addWidget(self.blue_btn)

        self.red_btn = QPushButton("红方")
        self.red_btn.setMinimumHeight(52)
        self.red_btn.setCursor(Qt.PointingHandCursor)
        self.red_btn.clicked.connect(lambda: self.set_side(2))
        ccl.addWidget(self.red_btn)

        ccl.addStretch()

        self.side_label = QLabel("当前：未同步")
        self.side_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: {FONT_BODY}px;")
        ccl.addWidget(self.side_label)

        layout.addWidget(color_row)

        # 节点当前颜色，None 表示尚未回读到
        self.current_side = None
        # 距上次回读的帧数，用于限制回读频率
        self.sync_ticks = 0
        self._render_side()

        self.image_label = QLabel("等待相机数据...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            f"background-color: #1a1a1a; color: #888; font-size: {FONT_BIG}px;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label, 1)

        hsv_frame = QFrame()
        hsv_frame.setStyleSheet(
            f"background-color: {PANEL_BG}; border: 1px solid {BORDER};")
        hsv_layout = QGridLayout(hsv_frame)
        hsv_layout.setContentsMargins(24, 20, 24, 20)
        hsv_layout.setHorizontalSpacing(30)
        hsv_layout.setVerticalSpacing(16)

        params = [
            ("H下限", 95, 0, 180), ("H上限", 130, 0, 180),
            ("S下限", 80, 0, 255), ("S上限", 255, 0, 255),
            ("V下限", 80, 0, 255), ("V上限", 255, 0, 255),
            ("曝光", 50000, 100, 1000000), ("增益", 10, 0, 20),
        ]

        self.hsv_spinboxes = {}
        for i, (name, default, min_v, max_v) in enumerate(params):
            l = QLabel(name)
            l.setStyleSheet(f"color: {TEXT}; font-size: {FONT_BODY}px;")
            spin = QSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default)
            spin.setMinimumWidth(150)
            spin.setMinimumHeight(52)
            spin.setStyleSheet(SPINBOX_STYLE)
            spin.setButtonSymbols(QSpinBox.UpDownArrows)
            self.hsv_spinboxes[name] = spin
            hsv_layout.addWidget(l, i // 2, (i % 2) * 2)
            hsv_layout.addWidget(spin, i // 2, (i % 2) * 2 + 1)

        color_l = QLabel("颜色")
        color_l.setStyleSheet(f"color: {TEXT}; font-size: {FONT_BODY}px;")
        self.color_spin = QSpinBox()
        self.color_spin.setRange(0, 1)
        self.color_spin.setValue(0)
        self.color_spin.setMinimumWidth(150)
        self.color_spin.setMinimumHeight(52)
        self.color_spin.setStyleSheet(SPINBOX_STYLE)
        self.color_spin.setButtonSymbols(QSpinBox.UpDownArrows)
        hsv_layout.addWidget(color_l, 4, 0)
        hsv_layout.addWidget(self.color_spin, 4, 1)
        hsv_layout.addWidget(QLabel(""), 4, 2, 1, 2)

        layout.addWidget(hsv_frame)

        apply_btn = QPushButton("应用相机参数")
        apply_btn.setStyleSheet(
            f"background-color: #43a047; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        apply_btn.clicked.connect(self.apply_params)
        layout.addWidget(apply_btn)

        self.state_label = QLabel("当前：等待")
        self.state_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: {FONT_BODY}px;")
        layout.addWidget(self.state_label)

    def apply_params(self):
        exposure = self.hsv_spinboxes["曝光"].value()
        gain = self.hsv_spinboxes["增益"].value()
        ok = self.ros_node.set_camera_params(exposure, gain)
        if ok:
            self.state_label.setText(f"已发送：曝光={exposure}us 增益={gain}")
        else:
            self.state_label.setText("失败：相机节点未运行")

    def set_side(self, color):
        """把己方阵营写入滤波节点，0 蓝 2 红"""
        ok, err = self.ros_node.set_self_color(color)
        if ok:
            # 以节点实际接受的值为准
            self.current_side = color
        else:
            self.side_label.setText(f"失败：{err}")
        self._render_side()

    def sync_side(self):
        """定时回读节点颜色，保证按钮状态与节点一致"""
        self.sync_ticks += 1
        # 每 15 帧（约 0.5 秒）回读一次，避免频繁阻塞
        if self.sync_ticks % 15 != 0:
            return
        color = self.ros_node.get_self_color()
        if color is None:
            # 节点未运行或回读失败，标记为未同步
            self.current_side = None
        else:
            self.current_side = color
        self._render_side()

    def _render_side(self):
        """按当前颜色高亮选中按钮，未同步时两者置灰"""
        # 蓝方与红方按钮的着色规则：颜色值、选中底色、未选中描边色
        sides = (
            (self.blue_btn, 0, BLUE, "#1565c0"),
            (self.red_btn, 2, RED, "#8e0000"),
        )
        for btn, color, base, active in sides:
            if self.current_side is None:
                # 未同步：灰色提示，但按钮仍可点击
                btn.setStyleSheet(
                    f"background-color: #ececec; color: {TEXT_DIM}; "
                    f"padding: 10px 28px; font-size: {FONT_BODY}px; "
                    f"border-radius: 5px; border: 1px solid {BORDER};")
            elif self.current_side == color:
                # 选中：实心主题色 + 白字
                btn.setStyleSheet(
                    f"background-color: {base}; color: white; "
                    f"padding: 10px 28px; font-size: {FONT_BODY}px; "
                    f"font-weight: bold; border-radius: 5px; border: none;")
            else:
                # 未选中：白底 + 主题色描边
                btn.setStyleSheet(
                    f"background-color: white; color: {active}; "
                    f"padding: 10px 28px; font-size: {FONT_BODY}px; "
                    f"border-radius: 5px; border: 1px solid {active};")
        if self.current_side == 0:
            self.side_label.setText("当前：蓝方（敌方为红）")
            self.side_label.setStyleSheet(
                f"color: {BLUE}; font-size: {FONT_BODY}px;")
        elif self.current_side == 2:
            self.side_label.setText("当前：红方（敌方为蓝）")
            self.side_label.setStyleSheet(
                f"color: {RED}; font-size: {FONT_BODY}px;")
        else:
            self.side_label.setText("当前：未同步")
            self.side_label.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: {FONT_BODY}px;")

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        self.state_label.setText(f"当前：{w}×{h}  RGB8")


class FieldView(QWidget):
    """中间面板：雷达点云 2D 俯视图"""
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.points = None
        self.range_m = 15.0
        # 倒置相关：默认正常朝向
        self.flip_x = False
        self.flip_y = False

    def set_points(self, pts):
        self.points = pts
        self.update()

    def set_orientation(self, flip_x, flip_y):
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor("#fafafa"))

        painter.setPen(QPen(QColor("#e5e5e5"), 1))
        for x in range(0, w, 60):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 60):
            painter.drawLine(0, y, w, y)

        margin = 50
        painter.setPen(QPen(QColor("#b0b0b0"), 2))
        painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)

        cx, cy = w // 2, h // 2
        size = min(w, h) // 8
        painter.setPen(QPen(QColor("#a0a0a0"), 2))
        painter.drawLine(cx, cy - size, cx + size, cy)
        painter.drawLine(cx + size, cy, cx, cy + size)
        painter.drawLine(cx, cy + size, cx - size, cy)
        painter.drawLine(cx - size, cy, cx, cy - size)

        if self.points is not None and len(self.points) > 0:
            scale = min(w, h) / (2 * self.range_m)
            painter.setPen(Qt.NoPen)
            for p in self.points:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                # 根据朝向翻转
                if self.flip_x:
                    x = -x
                if self.flip_y:
                    y = -y
                if abs(x) > self.range_m or abs(y) > self.range_m:
                    continue
                px = int(cx - y * scale)
                py = int(cy - x * scale)
                if 0 <= px < w and 0 <= py < h:
                    z_norm = max(0.0, min(1.0, (z + 0.5) / 2.5))
                    r = int(30 + 200 * z_norm)
                    g = int(200 - 150 * z_norm)
                    b = int(255 - 200 * z_norm)
                    painter.setBrush(QColor(r, g, b))
                    painter.drawEllipse(px - 1, py - 1, 3, 3)

            painter.setPen(QColor("#666"))
            painter.setFont(QFont("Ubuntu Mono", FONT_SMALL))
            painter.drawText(20, 40, f"点数: {len(self.points)}")
        else:
            painter.setPen(QColor("#b0b0b0"))
            painter.setFont(QFont("Ubuntu", FONT_BIG))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待雷达数据接入...")

        painter.setPen(QColor("#9e9e9e"))
        painter.setFont(QFont("Ubuntu Mono", FONT_SMALL))
        painter.drawText(w - 120, h - 24, f"{self.range_m:.0f}.0m")


class FieldPanel(QFrame):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER};")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("场地实时投影")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: {FONT_PANEL_TITLE}px; font-weight: bold;")
        layout.addWidget(title)

        self.field = FieldView()
        layout.addWidget(self.field, 1)

        # ===== 雷达参数行 =====
        ctrl_row = QWidget()
        cl = QHBoxLayout(ctrl_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(20)

        # 倒置复选框
        self.invert_cb = QCheckBox("雷达倒置（Y 轴翻转）")
        self.invert_cb.setChecked(False)
        self.invert_cb.setStyleSheet(f"color: {TEXT}; font-size: {FONT_SMALL}px;")
        self.invert_cb.stateChanged.connect(self._on_invert_changed)
        cl.addWidget(self.invert_cb)

        # 累积帧数
        l1 = QLabel("累积帧数:")
        l1.setStyleSheet(f"color: {TEXT}; font-size: {FONT_SMALL}px;")
        cl.addWidget(l1)

        self.decay_spin = QSpinBox()
        self.decay_spin.setRange(1, 20)
        self.decay_spin.setValue(3)
        self.decay_spin.setStyleSheet(SPINBOX_STYLE)
        self.decay_spin.setMinimumWidth(100)
        self.decay_spin.setMaximumHeight(60)
        self.decay_spin.valueChanged.connect(self._on_decay_changed)
        cl.addWidget(self.decay_spin)

        # 显示范围
        l2 = QLabel("显示范围(m):")
        l2.setStyleSheet(f"color: {TEXT}; font-size: {FONT_SMALL}px;")
        cl.addWidget(l2)

        self.range_spin = QSpinBox()
        self.range_spin.setRange(5, 50)
        self.range_spin.setValue(15)
        self.range_spin.setStyleSheet(SPINBOX_STYLE)
        self.range_spin.setMinimumWidth(100)
        self.range_spin.setMaximumHeight(60)
        self.range_spin.valueChanged.connect(self._on_range_changed)
        cl.addWidget(self.range_spin)

        cl.addStretch()
        layout.addWidget(ctrl_row)

        # ===== 显示选项复选框 =====
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(24)

        self.checkboxes = []
        for txt in ["显示敌方车辆", "显示己方车辆", "显示精确点", "显示雷达点",
                    "显示投影点", "显示信息波坐标", "显示追踪坐标"]:
            cb = QCheckBox(txt)
            cb.setChecked(False)
            self.checkboxes.append(cb)
            bl.addWidget(cb)

        bl.addStretch()
        layout.addWidget(bottom)
        self.set_checkboxes_enabled(False)

    def _on_invert_changed(self, state):
        # 倒置：翻转 Y；若雷达是绕不同轴旋转的，可改这里
        self.field.set_orientation(flip_x=False, flip_y=(state == Qt.Checked))

    def _on_decay_changed(self, value):
        self.ros_node.set_max_history(value)

    def _on_range_changed(self, value):
        self.field.range_m = float(value)
        self.field.update()

    def set_checkboxes_enabled(self, enabled):
        for cb in self.checkboxes:
            cb.setEnabled(enabled)
            if enabled:
                cb.setStyleSheet(f"color: {TEXT}; font-size: {FONT_SMALL}px;")
            else:
                cb.setStyleSheet(f"color: {TEXT_DIM}; font-size: {FONT_SMALL}px;")


class RightPanel(QFrame):
    def __init__(self, ros_node, on_exit):
        super().__init__()
        self.ros_node = ros_node
        self.on_exit = on_exit
        self.setStyleSheet(f"background-color: {PANEL_BG}; border: 1px solid {BORDER};")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        layout.addWidget(self._section_title("设备状态"))
        self.dot_camera = StatusDot("相机连接：未连接")
        self.dot_radar = StatusDot("雷达接收数据：未连接")
        self.dot_sentry = StatusDot("哨兵接收数据：未连接")
        self.dot_judge = StatusDot("裁判系统连接：未连接")
        for d in [self.dot_camera, self.dot_radar, self.dot_sentry, self.dot_judge]:
            layout.addWidget(d)

        self.fps_labels = {}
        for name, val in [("相机帧率", "0.0 FPS"), ("定位帧率", "0.0 FPS"),
                          ("定位发送频率", "0.0 FPS")]:
            row, value_label = self._info_row_with_label("● " + name, val)
            self.fps_labels[name] = value_label
            layout.addWidget(row)

        layout.addWidget(self._divider())
        layout.addWidget(self._section_title("校准控制"))

        btn1 = QPushButton("开始外参校准")
        btn1.setStyleSheet(
            f"background-color: #43a047; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        layout.addWidget(btn1)

        btn2 = QPushButton("直接输入外参")
        btn2.setStyleSheet(
            f"background-color: #1e88e5; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        layout.addWidget(btn2)

        layout.addWidget(self._divider())
        layout.addWidget(self._section_title("系统状态"))

        self.state_label = QLabel("双倍易伤状态机：IDLE")
        self.state_label.setStyleSheet(
            f"color: {RED}; font-size: {FONT_BODY}px; padding: 6px 0;")
        layout.addWidget(self.state_label)

        self.trigger_label = QLabel("可触发次数：0 / 2")
        self.trigger_label.setStyleSheet(f"color: {TEXT}; font-size: {FONT_BODY}px;")
        layout.addWidget(self.trigger_label)

        layout.addWidget(self._section_title("机器人标记状态"))

        robot_grid = QWidget()
        rg = QGridLayout(robot_grid)
        rg.setContentsMargins(0, 0, 0, 0)
        rg.setHorizontalSpacing(40)
        rg.setVerticalSpacing(16)
        robots = ["1英雄", "2工程", "3步兵", "4步兵", "7哨兵"]
        for i, name in enumerate(robots):
            l = QLabel(
                f'<span style="color:{TEXT}; font-size:{FONT_BODY}px;">{name}</span>'
                f'  <span style="color:{RED}; font-size:{FONT_BODY}px;">●</span>')
            rg.addWidget(l, i // 2, i % 2)
        layout.addWidget(robot_grid)

        layout.addStretch()

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(12)

        recv_btn = QPushButton("接收数据")
        recv_btn.setStyleSheet(
            f"background-color: #8e44ad; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        br.addWidget(recv_btn)

        send_btn = QPushButton("发送数据")
        send_btn.setStyleSheet(
            f"background-color: #1e88e5; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        br.addWidget(send_btn)

        exit_btn = QPushButton("退出")
        exit_btn.setStyleSheet(
            f"background-color: {RED}; color: white; padding: 14px; "
            f"font-size: {FONT_BODY}px; border-radius: 5px; border: none;")
        exit_btn.clicked.connect(self.on_exit)
        br.addWidget(exit_btn)

        layout.addWidget(btn_row)

    def _section_title(self, text):
        l = QLabel(text)
        l.setStyleSheet(
            f"color: {TEXT}; font-size: {FONT_PANEL_TITLE}px; font-weight: bold; "
            f"margin-top: 10px;")
        return l

    def _info_row_with_label(self, name, value):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        l = QLabel(name)
        l.setStyleSheet(f"color: {TEXT}; font-size: {FONT_STATUS}px;")
        v = QLabel(value)
        v.setStyleSheet(f"color: {YELLOW}; font-size: {FONT_STATUS}px;")
        rl.addWidget(l)
        rl.addStretch()
        rl.addWidget(v)
        return row, v

    def _divider(self):
        d = QFrame()
        d.setFrameShape(QFrame.HLine)
        d.setStyleSheet(
            f"color: {BORDER}; background-color: {BORDER}; max-height: 1px;")
        return d

    def refresh(self):
        cam_ok = self.ros_node.camera_connected()
        self.dot_camera.text = f"相机连接：{'已连接' if cam_ok else '未连接'}"
        self.dot_camera.set_connected(cam_ok)

        radar_ok = self.ros_node.lidar_connected()
        self.dot_radar.text = f"雷达接收数据：{'已连接' if radar_ok else '未连接'}"
        self.dot_radar.set_connected(radar_ok)

        fps = getattr(self.ros_node, 'current_fps', 0.0)
        self.fps_labels["相机帧率"].setText(f"{fps:.1f} FPS")


class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("RoboMaster 2027 破浪笃行雷达站控制界面")
        self.setStyleSheet(f"background-color: {DARK_BG};")

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        self.camera_panel = CameraPanel(ros_node)
        main_layout.addWidget(self.camera_panel, 5)

        self.field_panel = FieldPanel(ros_node)
        main_layout.addWidget(self.field_panel, 6)

        self.right_panel = RightPanel(ros_node, on_exit=self.close)
        main_layout.addWidget(self.right_panel, 4)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

    def update_ui(self):
        try:
            rclpy.spin_once(self.ros_node, timeout_sec=0)
        except Exception:
            return

        cam_ok = self.ros_node.camera_connected()
        if self.ros_node.latest_frame is not None and cam_ok:
            self.camera_panel.update_frame(self.ros_node.latest_frame)

        # 同步滤波节点的己方阵营，保证按钮高亮与节点一致
        self.camera_panel.sync_side()

        radar_ok = self.ros_node.lidar_connected()
        if radar_ok and self.ros_node.latest_lidar_points is not None:
            self.field_panel.field.set_points(self.ros_node.latest_lidar_points)

        self.field_panel.set_checkboxes_enabled(radar_ok)
        self.right_panel.refresh()


def main():
    rclpy.init()
    ros_node = RosNode()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Ubuntu", 16))

    window = MainWindow(ros_node)
    window.showMaximized()

    def on_close(event):
        window.timer.stop()
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()

    window.closeEvent = on_close
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()