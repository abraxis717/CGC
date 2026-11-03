cmake_minimum_required(VERSION 3.16)
project(cgc_ecsg_native VERSION 0.1.0 LANGUAGES C)
include(CTest)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_library(cgc_ecsg_native STATIC
    src/ecsg_math.c
    src/ecsg_state.c
    src/ecsg_executor.c)
target_include_directories(cgc_ecsg_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
if(UNIX)
    target_link_libraries(cgc_ecsg_native PUBLIC m)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cgc_ecsg_native PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
    foreach(test_name test_ecsg_math test_ecsg_state test_ecsg_executor)
        add_executable(${test_name} tests/${test_name}.c)
        target_link_libraries(${test_name} PRIVATE cgc_ecsg_native)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
