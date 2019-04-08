project(examples)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist libsnn)

add_executable(spiral spiral.cpp)
target_link_libraries(spiral libsnn)
