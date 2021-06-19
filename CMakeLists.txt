cmake_minimum_required(VERSION 3.15)

project (tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable ")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)


if(DEBUG)
    add_definitions(-DDEBUG)
endif()

if(TKDNN_PATH)
    message("SET TKDNN_PATH:"${TKDNN_PATH})
    add_definitions(-DTKDNN_PATH="${TKDNN_PATH}")
else()
    add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

find_package(CUDA REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

# compile
file(GLOB tkdnn_CUSRC "src/kernels/*.cu" "src/sorting.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})
target_link_libraries(kernels ${CUDA_CUBLAS_LIBRARIES})

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
message("Eigen DIR: " ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

# gives problems in cross-compiling, probably malformed cmake config
find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

if(BUILD_LIB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
    add_library(tkDNN SHARED ${tkdnn_SRC})
    target_link_libraries(tkDNN ${tkdnn_LIBS})

    if(INSTALL_LIB)
        message("install dir:" ${CMAKE_INSTALL_PREFIX})
        install(DIRECTORY include/ DESTINATION include/)
        install(TARGETS tkDNN kernels DESTINATION lib)
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/"
                DESTINATION "share/tkDNN/cmake/"
        )
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/" # source directory
                DESTINATION "share/tkDNN/tests" # target directory
        )
    endif()
endif()

add_executable(general_engine_test tests/ge_test.cpp ${tkdnn_SRC})
target_link_libraries(general_engine_test ${tkdnn_LIBS} nvinfer_plugin)