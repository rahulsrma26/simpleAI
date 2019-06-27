project(examples)

add_executable(spiral_test spiral_test.cpp)
target_link_libraries(spiral_test libsnn)

add_executable(demo_mnist demo_mnist.cpp)
target_link_libraries(demo_mnist libsnn)

add_executable(sequential_mnist sequential_mnist.cpp)
target_link_libraries(sequential_mnist libsnn)

add_executable(autoencoder_mnist autoencoder_mnist.cpp)
target_link_libraries(autoencoder_mnist libsnn)
