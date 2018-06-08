# automatic-ar

## Tested* prerequisites:

* CMake 2.8

* [Aruco](https://www.uco.es/investiga/grupos/ava/node/26) 3.0.0 (Included in the project)

* OpenCV 3.2.0

* PCL 1.7.2 (Optional, for point cloud I/O and visualization)

*The project might work with older dependencies however we have not tested.

## Hot to compile

Like a normal cmake project make a build library:

```shell
mkdir automatic-ar-build
```
change the current directory to the build directory:
```shell
cd automatic-ar-build
```
run cmake to configure the project and generate the makefiles:

```shell
cmake ../automatic-ar/trunk/
```
in case cmake does not find a library you have automatically, you can manually give cmake the path to where the library's cmake configuration file exists. For example
```shell
cmake ../automatic-ar/trunk/ -DOpenCV_DIR=~/local/share/OpenCV
```
Finally run make to build the binaries
```shell
make
```
You will find the executables to work with in the apps directory.
## Sample data
You can download sample data sets from [here](https://mega.nz/#F!riAgQY7J!7VbP7yOmsRKvFbkLtdUE1A).

## Usage
1. Unzip the desired data set.
```shell
unzip pentagonal.zip
```
2. Do the marker detection by giving the path of the data set to `detect_markers`:
```shell
detect_markers pentagonal
```
After unzipping you will find a file name `aruco.detections` in the data folder path.

3. Apply the algorithm by providing the data folder path and the physical size of the marker to `find_solution`:
```shell
find_solution pentagonal 0.04
```
Here `0.04` is the size of each marker in meters.

The output of `find_solutions` includes several files. Files with the name format 'initial*' store information when the solution after initialization and before optimization. The files with their names 'final*' store information resulted after doing the optimization.

## Visualization
If you compile with the PCL library you will have automatic visualization when you run `find_solution`. However, if not, you can still visualize the solution using the overlay app:
```shell
overlay pentagonal
```
You can also save overlayed visualization in a video in the dataset folder by using an extra option:
```shell
overlay pentagonal -save-video
```

## Dataset format

Each dataset is defined within a folder. In that folder each camera has a directory with its index as its name. In the cameras folders there is a "calib.xml" file specifying the camera matrix, distortion coefficients and the image size in OpenCV calibration output style.
