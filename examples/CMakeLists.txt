project(examples)

add_executable(spiral_test spiral_test.cpp)
target_link_libraries(spiral_test libsnn)

add_executable(demo_mnist demo_mnist.cpp)
target_link_libraries(demo_mnist libsnn)

add_executable(mnist_model mnist_model.cpp)
target_link_libraries(mnist_model libsnn)
