#include "dataset.h"
#include "multicam_mapper.h"
#include "image_array_detector.h"

using namespace std;

void print_usage(){
    cout<<"Usage: <data_folder_path> <path_to_solution_file>"<<endl;
}

int main(int argc, char *argv[]){

    if(argc<3){
        print_usage();
        return -1;
    }
    const int min_detections=2;
    string data_folder=argv[1];
    string solution_file_path=argv[2];

    size_t num_cams=dataset.get_num_cams();
    set<size_t> frame_nums;
    dataset.get_frame_nums(frame_nums);

    Dataset dataset(data_folder);
    ImageArrayDetector iad(num_cams);


    size_t frame_index=0;
    for(auto frame_num:frame_nums){
        vector<cv::Mat> frames;
        //retrieve the frame
        dataset.get_frame(frame_num,frames);
        //detect the markers
        iad.detect_markers(frames);
        //initiate the parameters
    }


    return 0;
}
