#ifndef INITIALIZER_H
#define INITIALIZER_H
#include <opencv2/core.hpp>
#include <map>
#include <set>
#include <aruco/aruco.h>

class Initializer
{

    std::vector<std::vector<std::vector<aruco::Marker>>> detections;
    enum transform_type{camera,marker};

    std::map<int, std::map<int, std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> transformation_sets_cam, transformation_sets_marker;

    struct Node{
        int id;
        mutable double distance;
        mutable int parent;
        bool operator < (const Node &n) const{
            return this->id<n.id;
        }
    };

    void fill_transformation_set(const std::map<int, std::map<int, std::vector<std::pair<cv::Mat,double>>>> &pose_estimations, const std::map<int, cv::Mat>& transforms_to_root_cam, const std::map<int, cv::Mat>& transforms_to_root_marker, std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>> &transformation_set);
    void fill_transformation_sets(transform_type tt, const std::map<int, std::map<int, std::vector<std::pair<cv::Mat,double>>>> &pose_estimations, std::map<int, std::map<int, std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> &transformation_sets);
    int find_best_transformation_min(double marker_size, const std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>>& solutions, double& min_error);
    int find_best_transformation(double marker_size, const std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>>& solutions, double& weight);
    void find_best_transformations(double marker_size, const std::map<int, std::map<int, std::vector<std::tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> &transformation_sets, std::map<int, std::map<int, std::pair<cv::Mat,double>>> &best_transformations);
//    double get_reprojection_error(double marker_size, aruco::Marker marker, cv::Mat r, cv::Mat t, cv::Mat cam_mat, cv::Mat dist_coeffs);
    void make_mst(int starting_node, std::set<int> node_ids, const std::map<int, std::map<int, std::pair<cv::Mat,double>>>& adjacency, std::map<int, std::set<int>> &children);
    void find_transforms_to_root(int root_node, const std::map<int,std::set<int>> &children, const std::map<int, std::map<int, std::pair<cv::Mat,double>>> &best_transforms, std::map<int, cv::Mat> &transforms_to_root);

public:
    Initializer();


};

#endif // INITIALIZER_H
