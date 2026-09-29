import os
from launch import LaunchDescription
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # 相机（用 hik_camera 组件，替代不存在的 NodeCamera）
    hik_camera_node = ComposableNode(
        package='hik_camera',
        plugin='hik_camera::HikCameraNode',
        name='hik_camera',
        parameters=[{'exposure_time': 25000.0, 'gain': 10.0}],
        #remappings=[('/image', 'camera_image')],
        remappings = [('camera_image', '/compressed_image')],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # 检测
    radar_detect_node = ComposableNode(
        package='tdt_vision',
        plugin='tdt_radar::Detect',
        name='radar_detect_node',
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # 坐标解算
    radar_resolve_node = ComposableNode(
        package='tdt_vision',
        plugin='tdt_radar::Resolve',
        name='radar_resolve_node',
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # Foxglove 可视化
    foxglove_node = ComposableNode(
        package='foxglove_bridge',
        plugin='foxglove_bridge::FoxgloveBridge',
        name='foxglove_bridge_node',
        parameters=[{'send_buffer_limit': 1000000000}],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # 容器
    container = ComposableNodeContainer(
        name='camera_detector_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            hik_camera_node,
            radar_detect_node,
            radar_resolve_node,
            foxglove_node,
        ],
        output='both',
        emulate_tty=True,
    )

    # 地图服务
    map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('tdt_vision'), 'launch',
            'map_server_launch.py')]),
    )

    return LaunchDescription([
        container,
        # map_launch,   # 需要地图可视化时再打开
    ])