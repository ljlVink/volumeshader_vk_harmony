cmake_minimum_required(VERSION 3.22)
project(volume_entry LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

# This repository's root owns the Vulkan renderer core (volume_core) and the
# shared volume/ headers; build it out-of-tree so its generated shader header
# and per-platform linking (Vulkan + HMS graphics libs on OHOS) stay in one
# place.
add_subdirectory(${NATIVERENDER_ROOT_PATH}/../../../../
                 ${CMAKE_CURRENT_BINARY_DIR}/volume_core)

add_library(entry SHARED napi_init.cpp volume_app.cpp)
target_link_libraries(entry PUBLIC volume_core libace_napi.z.so libace_ndk.z.so qos dl)
