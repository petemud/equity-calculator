find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(test test.cpp)
target_link_libraries(test Eigen3::Eigen)
