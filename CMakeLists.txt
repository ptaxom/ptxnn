cmake_minimum_required(VERSION 3.15)

project (tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE Release)
if(UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-unused-variable ")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)


if(DEBUG)
    set(CMAKE_BUILD_TYPE DEBUG)
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

# Python bindings
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp nvinfer_plugin)
set(python_LIBS pybind11::module pybind11::lto ${Python_LIBRARIES})

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

pybind11_add_module(ptxnn ${tkdnn_SRC})
target_link_libraries(ptxnn PUBLIC ${tkdnn_LIBS} ${python_LIBS})

if(INSTALL_PYTHON)
    install(TARGETS ptxnn
    COMPONENT python
    LIBRARY DESTINATION "/usr/lib/python3/dist-packages"
    )
endif()