#include <iostream>
#include <fstream>
#include <aruco/aruco.h>
#include <algorithm>
#include <set>
#include <opencv2/highgui.hpp>
#include <cctype>
#include <algorithm>
#include <cstdio>
#include "multicam_mapper.h"
#include "filesystem.h"

using namespace std;
using namespace filesystem;

int print_usage(char* argv0){
    cout<<"Usage: "<<argv0<<" <path_to_data_folder> [-d <dictionary>]"<<endl
        <<"The default values:"<<endl
        <<"\t<dictionary>: ARUCO_MIP_36h12"<<endl;
    return -1;
}

bool string_is_uint(string s){
    if(s.size()==0)
        return false;
    for(char c: s)
        if (!isdigit(c))
            return false;
    return true;
}

void get_frame_nums(size_t num_cams, string folder_path, vector<set<size_t>>& frame_nums, set<size_t>& all_frames){
    frame_nums.resize(num_cams);

    for(auto cam_num=0;cam_num<num_cams;cam_num++){
        string cam_dir_path=folder_path+"/"+to_string(cam_num);
        vector<string> files_list=get_files_list(cam_dir_path);
        sort(files_list.begin(),files_list.end());
        for(string file_name : files_list){
            size_t frame_num=0;
            if(sscanf(file_name.c_str(),"%lu.png",&frame_num)==1){
                frame_nums[cam_num].insert(frame_num);
                all_frames.insert(frame_num);
            }
        }
    }
}

void get_frame(size_t frame_num, string folder_path, vector<set<size_t>> &frame_nums, vector<cv::Mat> &frames){
    size_t num_cams=frame_nums.size();
    frames.resize(num_cams);

    for(size_t cam=0;cam<num_cams;cam++){
        string cam_dir_path=folder_path+"/"+to_string(cam);

        if(frame_nums[cam].count(frame_num)>0)//the cameras has that frame
            frames[cam]=cv::imread(cam_dir_path+"/"+to_string(frame_num)+".png");
    }
}

size_t get_num_cams(string folder_path){

    vector<string> dirs_list=get_dirs_list(folder_path);

    int num_cams=-1;
    for(size_t i=0;i<dirs_list.size();i++){
        cout<<dirs_list[i]<<endl;
        if(string_is_uint(dirs_list[i])){
            int dir_num=stoi(dirs_list[i]);
            if(dir_num>num_cams)
                num_cams=dir_num;
        }
    }

    num_cams++;
    return num_cams;
}

int main(int argc, char* argv[]){
    if(argc<2)
        return print_usage(argv[0]);

    string dictionary_name="ARUCO_MIP_36h12";

    if(argc>2){
        vector<string> optional_params;
        for(int i=3;i<argc;i++)
            optional_params.push_back(argv[i]);

        auto it=find(optional_params.begin(),optional_params.end(),"-d");
        if(it!=optional_params.end())
            dictionary_name=*(++it);
    }

    try{
        string folder_path = argv[1];

        size_t num_cams=get_num_cams(folder_path);

        string output_file_name=folder_path+"/aruco.detections";

        vector<aruco::MarkerDetector> detector(num_cams);
        for(size_t i=0;i<num_cams;i++){
            detector[i].getParameters().setDetectionMode(aruco::DetectionMode::DM_VIDEO_FAST,0);
            detector[i].setDictionary(dictionary_name);
            detector[i].getParameters().setCornerRefinementMethod(aruco::CornerRefinementMethod::CORNER_LINES);
        }

        vector<set<size_t>> frame_nums;
        set<size_t> all_frames;
        get_frame_nums(num_cams,folder_path,frame_nums,all_frames);
        cout<<"all frames: "<<all_frames.size()<<endl;
        ofstream output_file(output_file_name,ios_base::binary);
        if(!output_file.is_open())
            throw runtime_error("Could not open a file to write output at: "+output_file_name);

        output_file.write((char*)&num_cams,sizeof(num_cams));

        const int min_detections_per_marker=1;

        std::set<int> marker_ids;
        vector<cv::Mat> frames;
        for( size_t frame_num : all_frames){
            cout<<"frame num:"<<frame_num<<endl;
            auto start=chrono::system_clock::now();
            get_frame(frame_num,folder_path,frame_nums,frames);
            map<int,int> markers_count;
            vector<vector<aruco::Marker>> cam_markers(num_cams);
            for(size_t cam=0;cam<num_cams;cam++){
                cv::Mat tmp_img=frames[cam];
                detector[cam].detect(tmp_img,cam_markers[cam]);
                for(size_t m=0;m<cam_markers[cam].size();m++)
                    if(markers_count.find(cam_markers[cam][m].id)!=markers_count.end())
                        markers_count[cam_markers[cam][m].id]++;
                    else
                        markers_count[cam_markers[cam][m].id]=1;
            }
            auto end=chrono::system_clock::now();
            std::chrono::duration<double> d=end-start;

            vector<vector<aruco::Marker>> new_markers(num_cams);

            for(size_t cam=0;cam<num_cams;cam++){
                auto &markers=cam_markers[cam];

                for(size_t m=0;m<markers.size();m++){
                    if(markers_count[markers[m].id]>=min_detections_per_marker)//if there is more that one instance of that marker detected in the cameras
                        new_markers[cam].push_back(markers[m]);
                }
            }

            for(size_t cam=0;cam<num_cams;cam++){
                cv::Mat tmp_img=frames[cam];
                auto new_markers_size=new_markers[cam].size();
                output_file.write((char*)&new_markers_size,sizeof(new_markers_size));

                for(size_t m=0;m<new_markers_size;m++){
                    marker_ids.insert(new_markers[cam][m].id);
                    MultiCamMapper::serialize_marker(new_markers[cam][m],output_file);
                    new_markers[cam][m].draw(tmp_img);
                }
                imshow("cam_"+to_string(cam),tmp_img);
                cv::waitKey(1);
            }

        }
        cout<<"Detected marker ids: ";
        for(auto it=marker_ids.begin();it!=marker_ids.end();it++)
            cout<<*it<<" ";
        cout<<endl;
    }
    catch(cv::Exception e){
        cout<<e.err<<endl;
    }

}
