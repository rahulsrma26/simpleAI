project(tests)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor libsnn)

add_executable(test_activator test_activator.cpp)
target_link_libraries(test_activator libsnn)

add_executable(test_loss test_loss.cpp)
target_link_libraries(test_loss libsnn)

add_executable(test_kwargs test_kwargs.cpp)
target_link_libraries(test_kwargs libsnn)

add_executable(test_initializer test_initializer.cpp)
target_link_libraries(test_initializer libsnn)

add_executable(test_optimizer test_optimizer.cpp)
target_link_libraries(test_optimizer libsnn)

add_executable(test_variable test_variable.cpp)
target_link_libraries(test_variable libsnn)

add_executable(test_layer test_layer.cpp)
target_link_libraries(test_layer libsnn)

add_executable(test_model test_model.cpp)
target_link_libraries(test_model libsnn)

add_executable(test_tmath test_tmath.cpp)
target_link_libraries(test_tmath libsnn)
