project (libsnn)

file(GLOB SRC_FILES "src/*.cpp")

add_library(${PROJECT_NAME} ${SRC_FILES})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
