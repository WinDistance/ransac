cmake_minimum_required (VERSION 3.0)
project(ransac)


SET(CMAKE_C_COMPILER g++)

add_compile_options(-std=c++11)


add_executable(line_ransac src/line_ransac.cpp)

