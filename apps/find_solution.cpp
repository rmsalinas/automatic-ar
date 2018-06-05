#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/calib3d.hpp>
#include <aruco/markerdetector.h>
#include <aruco/ippe.h>
#include <aruco/cvdrawingutils.h>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <iterator>
#include <chrono>
#include "sparselevmarq.h"
#include "multicam_mapper.h"

using namespace std;

int print_usage(char* argv0){
    cout<<"Usage: "<<argv0<<" <folder_path> <marker_size> [-subseqs] [-exclude-cam]"<<endl;
    cout<<"\toptions:"<<endl
        <<"\t         -subseqs: evaluates the results only in the intervals of the subsequences in subseqs.txt"<<endl;
    return -1;
}

void fill_transformation_set(const map<int, map<int, vector<pair<cv::Mat,double>>>> &pose_estimations, const map<int, cv::Mat>& transforms_to_root_cam, const map<int, cv::Mat>& transforms_to_root_marker, vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>> &transformation_set){
//for object pose
    for(auto marker_it=pose_estimations.begin();marker_it!=pose_estimations.end();marker_it++){//loop on marker
        int marker_id=marker_it->first;
        auto marker_cams=marker_it->second;
        for(auto cam_it=marker_cams.begin();cam_it!=marker_cams.end();cam_it++){//loop on camera
            int cam_id=cam_it->first;
            auto pose_ests = cam_it->second;

            cv::Mat T_mr,T_rm,T_cr,T_rc;
            if(transforms_to_root_marker.find(marker_id)!=transforms_to_root_marker.end()){
                T_mr=transforms_to_root_marker.at(marker_id);
                T_rm=T_mr.inv();
            }
            else{
                T_mr=cv::Mat::eye(4,4,CV_64FC1);
                T_rm=cv::Mat::eye(4,4,CV_64FC1);
            }

            if(transforms_to_root_cam.find(cam_id)!=transforms_to_root_cam.end()){
                T_cr=transforms_to_root_cam.at(cam_id);
                T_rc=T_cr.inv();
            }
            else{
                T_cr=cv::Mat::eye(4,4,CV_64FC1);
                T_rc=cv::Mat::eye(4,4,CV_64FC1);
            }


            for(size_t i=0;i<pose_ests.size();i++){
                cv::Mat T_mc;
                pose_ests[i].first.convertTo(T_mc,CV_64FC1);
                cv::Mat T_cm=T_mc.inv();
                double error=pose_ests[i].second;
                transformation_set.push_back(make_tuple(T_cr*T_mc*T_rm, T_mr*T_cm, T_rc, error));
                //T_rc*(T_cr*T_mc*T_rm)*T_mr should transform points onto themeselves
            }

        }
    }
}

enum transform_type{camera,marker};

void fill_transformation_sets(transform_type tt, const map<int, map<int, vector<pair<cv::Mat,double>>>> &pose_estimations, map<int, map<int, vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> &transformation_sets){
//for cameras or markers
    for(auto it=pose_estimations.begin();it!=pose_estimations.end();it++){
        auto objects=it->second;
        if(objects.size()>1)//e.g. if more than one camera has seen the marker or more than one marker is seen in the camera
            for(auto it1=objects.begin();it1!=objects.end();++it1){//for every camera or for every marker
                int id1=it1->first;//id of the first object
                auto poses_1=it1->second;//candidate poses of the first object
                for(size_t i=0;i<poses_1.size();i++){
                    double error1=poses_1[i].second;
                    for(auto it2=next(it1);it2!=objects.end();it2++){
                        int id2=it2->first;//id of the second object
                        auto poses_2=it2->second;//candidate poses of the second object
                        for(size_t j=0;j<poses_2.size();j++){
                            double error2=poses_2[j].second;
                            switch(tt){
                            case camera:
                                transformation_sets[id1][id2].push_back(make_tuple(poses_2[j].first*poses_1[i].first.inv(), poses_1[i].first, poses_2[j].first.inv(), error1*error2));
                                break;
                            case marker:
                                transformation_sets[id1][id2].push_back(make_tuple(poses_2[j].first.inv()*poses_1[i].first, poses_1[i].first.inv(), poses_2[j].first, error1*error2));
                                break;
                            }
                        }
                    }
                }
            }
    }
}

int find_best_transformation_min(double marker_size, const vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>>& solutions, double& min_error){
    std::pair<int,double> bestMin(-1,std::numeric_limits<double>::max());
    for(size_t i=0;i< solutions.size();i++){
        double v =get<3>(solutions[i]);
        if (v<bestMin.second)
            bestMin={i,v};
    }
    min_error=bestMin.second;
    return bestMin.first;
}


int find_best_transformation(double marker_size, const vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>>& solutions, double& weight){
    double half_msize=marker_size/2;
    cv::Mat points(4,4,CV_64FC1);
    points.at<double>(0,0)=-half_msize;
    points.at<double>(1,0)=half_msize;
    points.at<double>(2,0)=0;
    points.at<double>(3,0)=1;
    points.at<double>(0,1)=half_msize;
    points.at<double>(1,1)=half_msize;
    points.at<double>(2,1)=0;
    points.at<double>(3,1)=1;
    points.at<double>(0,2)=half_msize;
    points.at<double>(1,2)=-half_msize;
    points.at<double>(2,2)=0;
    points.at<double>(3,2)=1;
    points.at<double>(0,3)=-half_msize;
    points.at<double>(1,3)=-half_msize;
    points.at<double>(2,3)=0;
    points.at<double>(3,3)=1;

    double min_error=numeric_limits<double>::max();
    int min_index=-1;

//    cout<<"solutions size: "<<solutions.size()<<endl;

    for(size_t i=0;i<solutions.size();i++){
        cv::Mat T=get<0>(solutions[i]);
        double curr_error=0;
        for(size_t j=0;j<solutions.size();j++){
            cv::Mat T1_inv=get<1>(solutions[j]);
            cv::Mat T2_inv=get<2>(solutions[j]);
            cv::Mat p2=T2_inv*T*T1_inv*points;
            cv::Mat diff=points-p2;
            diff=diff.rowRange(cv::Range(0,3));
            cv::Mat diff_sq=diff.mul(diff);
            cv::reduce(diff_sq,diff_sq,0,cv::REDUCE_SUM);
            cv::sqrt(diff_sq,diff_sq);
            curr_error += sum(diff_sq)[0];
        }
        if(curr_error<min_error){
            min_index=i;
            min_error=curr_error;
            weight=min_error;//get<3>(solutions[i]);
        }
    }
    return min_index;
}

void find_best_transformations(double marker_size, const map<int, map<int, vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> &transformation_sets, map<int, map<int, pair<cv::Mat,double>>> &best_transformations){

    //loop on the first item
    for(auto it1=transformation_sets.begin();it1!=transformation_sets.end();it1++){
        int id1=it1->first;
        cout<<"id1: "<<id1<<endl;
        //loop on the second item
        for(auto it2=it1->second.begin();it2!=it1->second.end();it2++){
            int id2=it2->first;
            cout<<"id2: "<<id2<<endl;
            const vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>> &solutions=it2->second;

//            if (1){
//            double Confidence;
//            int best_index=find_best_transformation_min(marker_size, solutions, Confidence);
//            best_transformations[id1][id2].first=get<0>(solutions[best_index]);
//            best_transformations[id1][id2].second=Confidence;
//            }
//            else{
//                double reproj_error;
//                int best_index=find_best_transformation(marker_size, solutions, reproj_error);
//                best_transformations[id1][id2].first=get<0>(solutions[best_index]);
//                best_transformations[id1][id2].second=reproj_error;
//            }

            double min_error;
            int min_index=find_best_transformation(marker_size, solutions, min_error);
            double reproj_error=min_error;//get<3>(solutions[min_index]);
            best_transformations[id1][id2].first=get<0>(solutions[min_index]);
            best_transformations[id1][id2].second=reproj_error;


        }
    }
}

//void make_stg_matrix(int num_nodes, map<int, map<int, pair<cv::Mat,double>>>& best_transformations, aruco_mm::StgMatrix<aruco_mm::Pose_Error>& sm){
//    sm.resize(num_nodes,num_nodes);
//    for(int i=0;i<num_nodes;i++){
//        for(int j=0;j<num_nodes;j++)
//            sm(i,j).repj_error=numeric_limits<double>::max();
//        sm(i,i).repj_error=numeric_limits<double>::max();
//        sm(i,i).pose=cv::Mat::eye(4,4,CV_64FC1);
//    }

//    for(auto it1=best_transformations.begin();it1!=best_transformations.end();it1++){
//        int ind1=it1->first;
//        for(auto it2=it1->second.begin();it2!=it1->second.end();it2++){
//            int ind2=it2->first;
//            cv::Mat T=it2->second.first;
//            double error=it2->second.second;

//            sm(ind1,ind2).pose=T;
//            sm(ind1,ind2).repj_error=error;
//            sm(ind2,ind1).pose=T.inv();
//            sm(ind2,ind1).repj_error=error;
//        }
//    }
//}

//void copy_stg2map(aruco_mm::StgMatrix<aruco_mm::Pose_Error>& sm, map<int, map<int, pair<cv::Mat,double>>>& transformations){
//    transformations.clear();
//    for(int i=0;i<sm.rows();i++)
//        for(int j=i+1;j<sm.cols();j++){
//            transformations[i][j].first=sm(i,j).pose.convert();
//            transformations[i][j].second=sm(i,j).repj_error;
//        }
//}

struct Node{
    int id;
    mutable double distance;
    mutable int parent;
    bool operator < (const Node &n) const{
        return this->id<n.id;
    }
};

double get_reprojection_error(double marker_size, aruco::Marker marker, cv::Mat r, cv::Mat t, cv::Mat cam_mat, cv::Mat dist_coeffs){

    vector<cv::Point2f> projected_points;
    cv::projectPoints(marker.get3DPoints(marker_size), r, t, cam_mat, dist_coeffs, projected_points);

    double error=0;
    for(int i=0;i<4;i++){
        error+=(projected_points[i].x-marker[i].x)*(projected_points[i].x-marker[i].x);
        error+=(projected_points[i].y-marker[i].y)*(projected_points[i].y-marker[i].y);
    }
    return error;
}

void make_mst(int starting_node, set<int> node_ids, const map<int, map<int, pair<cv::Mat,double>>>& adjacency, map<int, set<int>> &children){

    set<Node> nodes_outside_tree;
    for(auto it = node_ids.begin(); it != node_ids.end() ; it++){//fill the set of nodes with undetermined distances
        Node n;
        n.id=*it;
        n.parent=-1;
        if(n.id==starting_node)
            n.distance=0;
        else
            n.distance=numeric_limits<double>::max();
        nodes_outside_tree.insert(n);
    }

    while(!nodes_outside_tree.empty()){
        //find the undetermined node with the smallest distance
        auto min_node_it=nodes_outside_tree.begin();
        for(auto node_it=nodes_outside_tree.begin();node_it!=nodes_outside_tree.end();node_it++)
            if(node_it->distance<min_node_it->distance)
                min_node_it=node_it;

        //update the distance for the neighbours of the node with the minimum distance
        for(set<Node>::iterator node_it=nodes_outside_tree.begin();node_it!=nodes_outside_tree.end();node_it++){

            cv::Mat transform;
            double error=numeric_limits<double>::max();

            if(min_node_it->id<node_it->id){
                if(adjacency.find(min_node_it->id)!=adjacency.end())
                    if(adjacency.at(min_node_it->id).find(node_it->id)!=adjacency.at(min_node_it->id).end()){
                        transform=adjacency.at(min_node_it->id).at(node_it->id).first;
                        error=adjacency.at(min_node_it->id).at(node_it->id).second;
                    }
            }
            else if(adjacency.find(node_it->id)!=adjacency.end())
                    if(adjacency.at(node_it->id).find(min_node_it->id)!=adjacency.at(node_it->id).end()){
                        transform=adjacency.at(node_it->id).at(min_node_it->id).first;
                        error=adjacency.at(node_it->id).at(min_node_it->id).second;
                    }

            if(!transform.empty())//if they are neighbours
                if(error /*+ min_node_it->distance*/ < node_it->distance){//if (the edge weight + distance of the min node < distance of the current node)
                    node_it->distance=error /*+ min_node_it->distance*/;
                    if(node_it->parent != -1)
                        children[node_it->parent].erase(node_it->id);
                    children[min_node_it->id].insert(node_it->id);
                    node_it->parent=min_node_it->id;
                }
        }
        //remove the memeber with the minimum distance
        nodes_outside_tree.erase(min_node_it);
    }
}

void find_transforms_to_root(int root_node, const map<int,set<int>> &children, const map<int, map<int, pair<cv::Mat,double>>> &best_transforms, map<int, cv::Mat> &transforms_to_root){
    transforms_to_root[root_node] = cv::Mat::eye(4,4,CV_64FC1);
    queue<int> q;
    q.push(root_node);
    cout<<"finding best transforms"<<endl;
    while(!q.empty()){
        if(children.find(q.front()) != children.end())
        for(auto child_it=children.at(q.front()).begin();child_it!=children.at(q.front()).end();child_it++){
            int child_id=*child_it;
            int parent_id=q.front();

            if(child_id<parent_id)
                transforms_to_root[child_id] = best_transforms.at(child_id).at(parent_id).first.clone();
            else
                transforms_to_root[child_id] = best_transforms.at(parent_id).at(child_id).first.inv();

            if(parent_id!=root_node)
                transforms_to_root[child_id] = transforms_to_root[parent_id]*transforms_to_root[child_id];

            q.push(child_id);
        }
        q.pop();
    }
}

//void toy_error(const ucoslam::SparseLevMarq<float>::eVector& par, ucoslam::SparseLevMarq<float>::eVector& error){

//    if(error.size()!=1)
//        error.resize(1);
//    float x=par(0);
//    error(0) = (x*x+1);
//    cout<<"x: "<<x<<endl;
//    cout<<"error: "<<error(0)<<endl;
//}

void print_mst_standard(int root, vector<set<int>> &mst){

}

int main(int argc, char*argv[])
{
    auto start=chrono::system_clock::now();
    int min_detections=2;
    Eigen::initParallel();
    if(argc<3)
        return print_usage(argv[0]);

        set<int> excluded_cams;
        double marker_size=stod(argv[2]);
        bool optimize_graph=false;
        enum ArgFlag { NONE, excludeCams};
        ArgFlag arg_flag=NONE;
        string folder_path=argv[1];
        bool use_subseqs=false;
        bool tracking_only=false;
        bool with_huber=false;
        string detections_file_path=folder_path+"/aruco.detections";
        string tracking_input_solution_path=folder_path+"/../fixed/subseqs.solution";
        for(int i=4;i<argc;i++)
            if(string(argv[i])=="-subseqs"){
                use_subseqs=true;
            }
            else if(string(argv[i])=="-exclude-cams"){
                arg_flag=excludeCams;
            }
            else if(string(argv[i])=="-tracking-only"){
                tracking_only=true;
            }
            else if(string(argv[i])=="-with-huber"){
                with_huber=true;
            }
            else if(arg_flag==excludeCams){
                excluded_cams.insert(stoi(argv[i]));
            }

        string solution_file_name="";

        if(tracking_only)
            solution_file_name+="_tracking_only";

        if(use_subseqs)
            solution_file_name+="_subseqs";

        if(with_huber)
            solution_file_name+="_with_huber";

        if(excluded_cams.size()>0){
            solution_file_name+="_excluded_cams";
            for(int cam_id:excluded_cams)
                solution_file_name+="_"+to_string(cam_id);
        }

        solution_file_name+=".solution";

        string initial_solution_file_path = folder_path+"/initial"+solution_file_name;
        string final_solution_file_path = folder_path+"/final"+solution_file_name;

        vector<CamConfig> cam_configs=CamConfig::read_cam_configs(folder_path);

        map<int, map<int, vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>>>> transformation_sets_cam, transformation_sets_marker;

        vector<vector<vector<aruco::Marker>>> detections;
        if(use_subseqs){
            auto subseqs=MultiCamMapper::read_subseqs(folder_path+"/subseqs.txt");
            detections=MultiCamMapper::read_detections_file(detections_file_path,subseqs);
        }
        else
        detections=MultiCamMapper::read_detections_file(detections_file_path);

        set<int> marker_ids,cam_ids;
        map<int, map<int, map<int, vector<pair<cv::Mat,double>>>>> frame_detections;
        map<int, map<int, vector<aruco::Marker>>> frame_cam_markers;

        cout<<"Estimating camera poses from markers.."<<endl;
        for(int frame_num=0;frame_num<detections.size();frame_num++){
            cout<<"frame_num: "<<frame_num<<endl;
            map<int,map<int,vector<pair<cv::Mat,double>>>> pose_estimations_marker, pose_estimations_cam;
            //Find all of the solutions for all of markers in each camera

            //first check if the frame has more than one detections
            int num_detections=0;
            for(int cam=0;cam<detections[frame_num].size();cam++)
                if(excluded_cams.count(cam)==0)
                    num_detections += detections[frame_num][cam].size();

            if(!(num_detections >= min_detections))
                continue;

            //loop over all possible cameras
            for(int cam=0;cam<detections[frame_num].size();cam++)
                if(excluded_cams.count(cam)==0){
                    size_t num_cam_markers=detections[frame_num][cam].size();

                    if(num_cam_markers < 1)
                        continue;

                    cam_ids.insert(cam);

                    auto &cam_markers=frame_cam_markers[frame_num][cam];

                    for(int m=0;m<num_cam_markers;m++){
                        aruco::Marker &marker=detections[frame_num][cam][m];

                        marker_ids.insert(marker.id);

                        cam_markers.push_back(marker);

                        vector<pair<cv::Mat,double>> solutions=aruco::solvePnP_(marker_size,marker,cam_configs[cam].getCamMat(),cam_configs[cam].getDistCoeffs());
//                        double ambiguity=solutions[0].second/solutions[1].second;
//                        cout<<marker.getArea()<<"  m1="<<ambiguity/marker.getArea()<<" "<<ambiguity<<" "<<1/ambiguity<<" "<<solutions[0].second<<" "<<solutions[1].second<<endl;
//                        solutions[0].second=solutions[1].second=ambiguity/marker.getArea();
                        solutions[0].first.convertTo(solutions[0].first,CV_64FC1);
                        pose_estimations_cam[marker.id][cam].push_back(solutions[0]);
                        pose_estimations_marker[cam][marker.id].push_back(solutions[0]);


//                        if(solutions[1].second/solutions[0].second < 4){
//                            solutions[1].first.convertTo(solutions[1].first,CV_64FC1);
//                            pose_estimations_cam[marker.id][cam].push_back(solutions[1]);
//                            pose_estimations_marker[cam][marker.id].push_back(solutions[1]);
//                        }
                    }
                }

            frame_detections[frame_num]=pose_estimations_cam;
            if(!tracking_only){
                fill_transformation_sets(transform_type::camera,pose_estimations_cam,transformation_sets_cam);
                fill_transformation_sets(transform_type::marker,pose_estimations_marker,transformation_sets_marker);
            }
        }

        int root_cam, root_marker;
        map<int,cv::Mat> transforms_to_root_cam,transforms_to_root_marker;

#ifdef PCL
        pcl::PointCloud<pcl::PointXYZRGB> point_cloud_cameras,point_cloud_markers;
#endif

        if(tracking_only){
            MultiCamMapper mcm;
            mcm.read_solution_file(tracking_input_solution_path);
            root_cam=mcm.get_root_cam();
            root_marker=mcm.get_root_marker();

            MultiCamMapper::MatArrays ma=mcm.get_mat_arrays();
            auto mcm_ttrc=ma.transforms_to_root_cam;
            auto mcm_ttrm=ma.transforms_to_root_marker;
            for(auto &cam_id_index: mcm_ttrc.m){
                int cam_id=cam_id_index.first;
                int cam_index=cam_id_index.second;
                transforms_to_root_cam[cam_id]=mcm_ttrc.v[cam_index];
            }
            for(auto &marker_id_index: mcm_ttrm.m){
                int marker_id=marker_id_index.first;
                int marker_index=marker_id_index.second;
                transforms_to_root_marker[marker_id]=mcm_ttrm.v[marker_index];
            }
        }
        else{
            int num_cams=cam_configs.size();
            size_t num_markers=marker_ids.size();

            cout<<"marker_ids: ";
            for(auto it=marker_ids.begin();it!=marker_ids.end();it++)
                cout<<*it<<" ";
            cout<<endl;

            cout<<"cam ids: ";
            for(auto it=cam_ids.begin();it!=cam_ids.end();it++)
                cout<<*it<<" ";
            cout<<endl;

            cout<<"Finding the best transformations.."<<endl;
            map<int, map<int, pair<cv::Mat,double>>> best_transforms_cam;
            map<int, map<int, pair<cv::Mat,double>>> best_transforms_marker;
            find_best_transformations(marker_size,transformation_sets_cam,best_transforms_cam);
            find_best_transformations(marker_size,transformation_sets_marker,best_transforms_marker);


            map<int, map<int, pair<cv::Mat, double>>> transforms_cam,transforms_marker;


            transforms_cam=best_transforms_cam;
            transforms_marker=best_transforms_marker;
            root_cam=*cam_ids.begin();
            root_marker=*marker_ids.begin();


            map<int, set<int>> cam_tree,marker_tree;

            cout<<"Making the MST.."<<endl;

//            //debug
//            auto t_3_5=best_transforms_marker[3][5];
//            auto t_4_5=best_transforms_marker[4][5];
//            cout<<"t_3_5 weight"<<t_3_5.second<<endl;
//            cout<<"t_4_5 weight"<<t_4_5.second<<endl;
//            MultiCamMapper::visualize_marker(3,t_3_5.first,0.04,point_cloud_markers);
//            MultiCamMapper::visualize_marker(4,t_4_5.first,0.04,point_cloud_markers);
//            MultiCamMapper::visualize_marker(5,cv::Mat::eye(4,4,CV_64FC1),0.04,point_cloud_markers);
//            pcl::io::savePCDFile(folder_path + "/markers_cloud.pcd",point_cloud_markers);
//            return 0;

            make_mst(root_cam,cam_ids,transforms_cam,cam_tree);
            make_mst(root_marker,marker_ids,transforms_marker,marker_tree);

            cout<<"Finding the transformations to the reference camera.."<<endl;
            find_transforms_to_root(root_cam,cam_tree,transforms_cam,transforms_to_root_cam);
            cout<<"Finding the transformations to the reference marker.."<<endl;
            find_transforms_to_root(root_marker,marker_tree,transforms_marker,transforms_to_root_marker);

            cout<<"Visualizing the camera positions in the point cloud.."<<endl;
//            //debug
//            auto t_3_5=best_transforms_marker[3][5];
//            auto t_4_5=best_transforms_marker[4][5];
//            cout<<"t_3_5 weight"<<t_3_5.second<<endl;
//            cout<<"t_4_5 weight"<<t_4_5.second<<endl;
//            MultiCamMapper::visualize_marker(3,t_3_5.first,0.04,point_cloud_markers);
//            MultiCamMapper::visualize_marker(4,t_4_5.first,0.04,point_cloud_markers);
//            MultiCamMapper::visualize_marker(5,cv::Mat::eye(4,4,CV_64FC1),0.04,point_cloud_markers);
//            pcl::io::savePCDFile(folder_path + "/markers_cloud.pcd",point_cloud_markers);

//            cout<<"3_5_transfomrs:"<<transformation_sets_marker[3][5].size()<<endl;
//            cout<<"4_5_transfomrs:"<<transformation_sets_marker[4][5].size()<<endl;

//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
//            pcl::visualization::PCLVisualizer pclv;
//            pclv.addPointCloud(point_cloud);

//            for(auto t: transformation_sets_marker[4][5]){
//                point_cloud->clear();
//                auto t_3_5=best_transforms_marker[3][5];
//                auto t_4_5=get<0>(t);
//                MultiCamMapper::visualize_marker(3,t_3_5.first,0.04,*point_cloud);
//                MultiCamMapper::visualize_marker(4,t_4_5,0.04,*point_cloud);
//                MultiCamMapper::visualize_marker(5,cv::Mat::eye(4,4,CV_64FC1),0.04,*point_cloud);
//                pclv.updatePointCloud(point_cloud);
//                pclv.spin();
//            }
//            //debug
        }


        cout<<"root marker:"<<root_marker<<endl;
        cout<<"Initializing object poses.."<<endl;
        cout<<"frame:    ";
        map<int,cv::Mat> best_object_transforms;
        for(auto it=frame_detections.begin(); it != frame_detections.end(); it++){
            int frame = it->first;
            cout<<"\r"<<setw(3)<<frame;
            cout.flush();
            vector<tuple<cv::Mat,cv::Mat,cv::Mat,double>> transformation_set;
            fill_transformation_set(it->second,transforms_to_root_cam,transforms_to_root_marker,transformation_set);
            double min_err;
            int best_transform_index=find_best_transformation(marker_size,transformation_set,min_err);
            if(best_transform_index >= 0)
                best_object_transforms[frame]=get<0>(transformation_set[best_transform_index]);
        }

        //local optimization
        MultiCamMapper mcm(root_cam,transforms_to_root_cam,root_marker,transforms_to_root_marker,best_object_transforms,frame_cam_markers,marker_size,cam_configs);
        mcm.set_optmize_flag_cam_intrinsics(false);
        if(with_huber)
            mcm.set_with_huber(true);

        if(tracking_only){
            mcm.set_optmize_flag_cam_poses(false);
            mcm.set_optmize_flag_marker_poses(false);
            mcm.set_optmize_flag_object_poses(true);
        }

#ifdef PCL
//        mcm.visualize_sequence();
        mcm.write_solution_file(initial_solution_file_path);
        mcm.write_text_solution_file(initial_solution_file_path+".yaml");
        point_cloud_markers.clear();
        point_cloud_cameras.clear();
        mcm.visualize_markers(point_cloud_markers);
        pcl::io::savePCDFile(initial_solution_file_path+".markers.pcd",point_cloud_markers);
        mcm.visualize_cameras(point_cloud_cameras);
        pcl::io::savePCDFile(initial_solution_file_path+".cameras.pcd",point_cloud_cameras);
#endif

        mcm.solve();

#ifdef PCL
//        mcm.visualize_sequence();
        mcm.write_solution_file(final_solution_file_path);
        mcm.write_text_solution_file(final_solution_file_path+".yaml");
        point_cloud_markers.clear();
        point_cloud_cameras.clear();
        mcm.visualize_markers(point_cloud_markers);
        pcl::io::savePCDFile(final_solution_file_path+".markers.pcd",point_cloud_markers);
        mcm.visualize_cameras(point_cloud_cameras);
        pcl::io::savePCDFile(final_solution_file_path+".cameras.pcd",point_cloud_cameras);
#endif

        auto end=chrono::system_clock::now();
        std::chrono::duration<double> d=end-start;
        cout<<"the duration was: "<<d.count()/60<<" minutes "<<d.count()-floor(d.count()/60)*60<<" seconds"<<endl;
        cout<<"in total: "<<d.count()<<" seconds"<<endl;


    return 0;
}

