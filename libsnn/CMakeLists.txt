project (libsnn)

file(GLOB_RECURSE SRC_FILES "src/**.cpp")
#file(GLOB SRC_FILES "src/activator/sigmoid.cpp")

add_library(${PROJECT_NAME} ${SRC_FILES})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
