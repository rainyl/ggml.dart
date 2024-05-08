cmake_minimum_required (VERSION 3.12)
project(ggml_dart VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

set(BUILD_SHARED_LIBS OFF)
add_subdirectory(./ggml)
set(BUILD_SHARED_LIBS ON)

add_library(
    ${PROJECT_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml.dart.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/ggml.dart.h
)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ggml
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/ggml
    )
