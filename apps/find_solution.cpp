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
    cout<<"Usage: "<<argv0<<" <folder_path> <marker_size> [-subseqs] [ -exclude-cam [<cam_1> [<cam2> [...] ] ]  ] ["<<endl;
    cout<<"\toptions:"<<endl
        <<"\t         -subseqs: evaluates the results only in the intervals of the subsequences in subseqs.txt."<<endl
        <<"\t         -exclude-cam: specifies a list of cameras that should be excluded from initialization and optimization."<<endl
        <<"\t         "<<endl;
    return -1;
}

int main(int argc, char*argv[])
{

    int min_detections = 2;
    double threshold = 2.0;
    Eigen::initParallel();
    if(argc<3)
        return print_usage(argv[0]);

        set<int> excluded_cams;
        double marker_size=stod(argv[2]);

        enum ArgFlag { NONE, ExcludeCams, Threshold};
        ArgFlag arg_flag=NONE;
        string folder_path=argv[1];
        bool use_subseqs=false;
        bool tracking_only=false;
        bool with_huber=false;
        bool set_threshold=false;
        string detections_file_path=folder_path+"/aruco.detections";
        string tracking_input_solution_path=folder_path+"/../fixed/subseqs.solution";
        for(int i=4;i<argc;i++)
            if(string(argv[i])=="-subseqs"){
                use_subseqs=true;
            }
            else if(string(argv[i])=="-exclude-cams"){
                arg_flag=ExcludeCams;
            }
            else if(string(argv[i])=="-tracking-only"){
                tracking_only=true;
                arg_flag=NONE;
            }
            else if(string(argv[i])=="-with-huber"){
                with_huber=true;
                arg_flag=NONE;
            }
            else if(string(argv[i])=="-thresh"){
                set_threshold=true;
                arg_flag=Threshold;
            }
            else if(arg_flag==ExcludeCams){
                excluded_cams.insert(stoi(argv[i]));
            }
            else if(arg_flag==Threshold){
                threshold=stod(argv[i]);
                arg_flag=NONE;
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

        if(set_threshold){
            char d[5];
            sprintf(d,"%.1f",threshold);
            solution_file_name+="_thresh_"+string(d);
        }

        solution_file_name+=".solution";

        string initial_solution_file_path = folder_path+"/initial"+solution_file_name;
        string final_solution_file_path = folder_path+"/final"+solution_file_name;

        vector<CamConfig> cam_configs=CamConfig::read_cam_configs(folder_path);

        vector<vector<vector<aruco::Marker>>> detections;

        if(use_subseqs){
            auto subseqs=MultiCamMapper::read_subseqs(folder_path+"/subseqs.txt");
            detections=MultiCamMapper::read_detections_file(detections_file_path,subseqs);
        }
        else
        detections=MultiCamMapper::read_detections_file(detections_file_path);

        auto start=chrono::system_clock::now();

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

                        solutions[0].first.convertTo(solutions[0].first,CV_64FC1);
                        pose_estimations_cam[marker.id][cam].push_back(solutions[0]);
                        pose_estimations_marker[cam][marker.id].push_back(solutions[0]);


                        if(solutions[1].second/solutions[0].second < threshold){
                            solutions[1].first.convertTo(solutions[1].first,CV_64FC1);
                            pose_estimations_cam[marker.id][cam].push_back(solutions[1]);
                            pose_estimations_marker[cam][marker.id].push_back(solutions[1]);
                        }
                    }
                }

            frame_detections[frame_num]=pose_estimations_cam;
            if(!tracking_only){
                //fill_transformation_sets(transform_type::camera,pose_estimations_cam,transformation_sets_cam);
                //fill_transformation_sets(transform_type::marker,pose_estimations_marker,transformation_sets_marker);
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
            //find_best_transformations(marker_size,transformation_sets_cam,best_transforms_cam);
            //find_best_transformations(marker_size,transformation_sets_marker,best_transforms_marker);


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

            //make_mst(root_cam,cam_ids,transforms_cam,cam_tree);
            //make_mst(root_marker,marker_ids,transforms_marker,marker_tree);

            cout<<"Finding the transformations to the reference camera.."<<endl;
            //find_transforms_to_root(root_cam,cam_tree,transforms_cam,transforms_to_root_cam);
            cout<<"Finding the transformations to the reference marker.."<<endl;
            //find_transforms_to_root(root_marker,marker_tree,transforms_marker,transforms_to_root_marker);

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
            //fill_transformation_set(it->second,transforms_to_root_cam,transforms_to_root_marker,transformation_set);
            double min_err;
            int best_transform_index;//=find_best_transformation(marker_size,transformation_set,min_err);
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

        std::chrono::duration<double> d=chrono::system_clock::now()-start;

        mcm.write_solution_file(initial_solution_file_path);
        mcm.write_text_solution_file(initial_solution_file_path+".yaml");
#ifdef PCL
        mcm.visualize_sequence();
        point_cloud_markers.clear();
        point_cloud_cameras.clear();
        mcm.visualize_markers(point_cloud_markers);
        pcl::io::savePCDFile(initial_solution_file_path+".markers.pcd",point_cloud_markers);
        mcm.visualize_cameras(point_cloud_cameras);
        pcl::io::savePCDFile(initial_solution_file_path+".cameras.pcd",point_cloud_cameras);
#endif
        start=chrono::system_clock::now();
        mcm.solve();
        d += chrono::system_clock::now()-start;

        mcm.write_solution_file(final_solution_file_path);
        mcm.write_text_solution_file(final_solution_file_path+".yaml");
#ifdef PCL
        mcm.visualize_sequence();
        point_cloud_markers.clear();
        point_cloud_cameras.clear();
        mcm.visualize_markers(point_cloud_markers);
        pcl::io::savePCDFile(final_solution_file_path+".markers.pcd",point_cloud_markers);
        mcm.visualize_cameras(point_cloud_cameras);
        pcl::io::savePCDFile(final_solution_file_path+".cameras.pcd",point_cloud_cameras);
#endif


        int minutes=d.count()/60;
        int seconds=lround(d.count()-minutes*60);
        cout<<"The algorithm took: "<<minutes<<" minutes "<<seconds<<" seconds"<<endl;


    return 0;
}

