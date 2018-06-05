# automatic-ar

## Tested* prerequisites:

* CMake 2.8

* [Aruco](https://www.uco.es/investiga/grupos/ava/node/26) 3.0.0 

* OpenCV 3.2.0

* PCL 1.7.2 (Optional, for point cloud I/O and visualization)

*The project might work with older dependencies however we have not tested.

## Sample data
You can download sample data sets from [here](https://mega.nz/#F!riAgQY7J!7VbP7yOmsRKvFbkLtdUE1A).

## How to use the program with sample data
1. Unzip the desired data set.
```shell
unzip pentagonal.zip
```
2. Do the marker detection by giving the path of the data set to `detect_markers`:
```shell
detect_markers pentagonal
```
3. Apply the algorithm:
```shell
find_solution pentagonal 0.04
```
Here `0.04` is the size of each marker in meters.


