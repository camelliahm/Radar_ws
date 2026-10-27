#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H
#include <iostream>
#include <string>
#include "vector"
#include <opencv2/opencv.hpp>

namespace radar_core
{
    // 单个场地平面区域的像素->世界映射器。
    // 每个区域的点位由 RM2025_Points.yaml 给出（图像上看到的四边形顶点对应的世界坐标），
    // 通过外参投影回像素后，用「点是否落在该多边形内」判断点击像素属于哪个区域，
    // 从而得到该区域装甲板的高度 Height。
    class Parser_Points
    {
    public:
        Parser_Points(const std::string &points_name);
        // 判断像素点是否落在本区域多边形内：是则返回 Height，否则返回 0
        float return_height(cv::Point2f &input_point);
        // 外参更新后重新从 yaml 读取并重投影
        void Update();
        // 把 Points_3D 投影到像素平面得到 Points_2D
        void World2Camera();
        std::vector<cv::Point3f> ReadPoints(const std::string &points_name);
        std::vector<cv::Point> Float2Int(std::vector<cv::Point2f> &FloatPoint);
        std::vector<cv::Point3f> Points_3D;  // 区域顶点的世界坐标
        std::vector<cv::Point> Points_2D;    // 区域顶点的像素坐标（投影结果）
        float Height=0;                      // 该区域目标（装甲板）离地高度，单位 m
    private:
        cv::Mat world_rvec;    // 世界->相机 旋转向量（外参）
        cv::Mat world_tvec;    // 世界->相机 平移向量（外参）
        cv::Mat camera_matrix; // 相机内参 3x3
        cv::Mat dist_coeffs;   // 畸变系数
    };
    // 像素坐标 -> 场地世界坐标的解算入口。
    // 思路：先按区域确定目标高度，再用该高度上的水平面做单应反投影。
    class parser
    {
    public:
        parser();
        // 标定结束后重读外参，并让所有子区域刷新投影点
        void Change_Matrix();
        // 输入检测框中心像素，返回场地坐标（单位 m）
        cv::Point2f parse(cv::Point2f &input_point);
        // 调试用：在图上画出各区域多边形
        void draw_ui(cv::Mat &img);
        float get_height(cv::Point2f &input_point);
        // 在指定高度的水平面上，把像素点反投影为世界坐标
        cv::Point2f get_2d(cv::Point2f &input_point,float height);
        cv::Mat world_rvec;
        cv::Mat world_tvec;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        // 区域名 -> 区域映射器（Middle_Line / Left_Road / Right_Road /
        // Enemy_Buff / Self_Fortress / Enemy_Fortress）
        std::map <std::string,Parser_Points*> points_map;
    };
}
#endif //RADAR_UTILS_H
