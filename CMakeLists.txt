cmake_minimum_required(VERSION 3.16)
project(cgc_native_research VERSION 0.1.0 LANGUAGES C)
include(CTest)
# Independent research components; no CGC/LAO cross-link or runtime admission.
add_subdirectory(native/ECS_G)
add_subdirectory(native/LAO)
