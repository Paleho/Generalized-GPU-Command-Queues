cmake_minimum_required (VERSION 3.10)

## Define a name for the backend library wrappers/implementations used in this build, Currently implemented with CuCuBLAS(==CUDA+CUBLAS) and HIP
# set(BACKEND "CuCuBLAS")
message("Universal Helpers - BACKEND: " ${BACKEND})
add_definitions(-DBACKEND="${BACKEND}")

## Define a name for the testbed, to be used for the whole framework setup
set(TESTBED_NAME "silver1") # silver1
message("Universal Helpers - TESTBED_NAME: " ${TESTBED_NAME})
add_definitions(-DTESTBED="${TESTBED_NAME}")

## Define the (max) number of devices for utilization (0-num_devices)
set(NUM_LOCATIONS 4) # silver1
message("Universal Helpers - NUM_LOCATIONS: " ${NUM_LOCATIONS})
add_definitions(-DLOC_NUM=${NUM_LOCATIONS})

## Define the max GPU memory percentage (%) a problem can use ( not accounting for streams and other control structs; e.g. only the problem data ).
set(PROBLEM_GPU_PERCENTAGE 70)
message("Universal Helpers - PROBLEM_GPU_PERCENTAGE: " ${PROBLEM_GPU_PERCENTAGE})
add_definitions(-DPROBLEM_GPU_PERCENTAGE=${PROBLEM_GPU_PERCENTAGE})

### Flags

## CMAKE_CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -Wuninitialized")
message("Universal Helpers - CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

set (CUDA_PREFIX "/usr/local/cuda-11.6") # Silver1
add_definitions(-DCUDA_VER=1100)

set (ROCM_PREFIX "/opt/rocm-5.4.0") # Silver1

message("Universal Helpers - CXX_PREFIX: " ${CXX_PREFIX})

## CMAKE_CUDA_ARCHITECTURES
string(COMPARE EQUAL "${CMAKE_CUDA_ARCHITECTURES}" "" empty_cuda_arch)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR empty_cuda_arch)
	set(CMAKE_CUDA_ARCHITECTURES 70) #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70
endif()
message("Universal Helpers - CMAKE_CUDA_ARCHITECTURES: " ${CMAKE_CUDA_ARCHITECTURES} )


# VALUE_TYPE
set(VALUETYPEID 1)
if (${VALUETYPEID} EQUAL 0)
	add_definitions(-DVALUE_TYPE=float)
elseif(${VALUETYPEID} EQUAL 1)
	add_definitions(-DVALUE_TYPE=double)
else()
endif()


# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)
set (CMAKE_HIP_COMPILER ${ROCM_PREFIX}/bin/hipcc)

# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -O3")
message("Universal Helpers - CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

set (CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -w -O3")
message("Universal Helpers - CMAKE_HIP_FLAGS: " ${CMAKE_HIP_FLAGS})

# 	c) Includes
if (${BACKEND} STREQUAL "CuCuBLAS" OR ${BACKEND} STREQUAL "Pthreads")
	set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
	include_directories(${CUDA_INCLUDE_DIRS})
	message("Universal Helpers - CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

elseif(${BACKEND} STREQUAL "HIP")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_NVIDIA__ -D__HIP_PLATFORM_NVCC__") # for NVIDIA platform (can be changed to __HIP_PLATFORM_AMD__)

	set (HIP_INCLUDE_DIRS ${ROCM_PREFIX}/include )
	include_directories(${HIP_INCLUDE_DIRS})
	set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
	include_directories(${CUDA_INCLUDE_DIRS})
	message("Universal Helpers - HIP_INCLUDE_DIRS: " ${HIP_INCLUDE_DIRS})

else()
	message(FATAL_ERROR "Universal Helpers - BACKEND flag error: BACKEND=" ${BACKEND} )
endif()

set(OPENBLAS_PREFIX "/home/users/spoutas/Libs/OpenBLAS-install") # silver1
set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("Universal Helpers - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

if(${BACKEND} STREQUAL "HIP")
	set(HIPBLAS_PREFIX "/opt/rocm-5.4.0/hipblas") # silver1
	set (HIPBLAS_INCLUDE_DIRS ${HIPBLAS_PREFIX}/include)
	include_directories(${HIPBLAS_INCLUDE_DIRS})
	message("Universal Helpers - HIPBLAS_INCLUDE_DIRS: " ${HIPBLAS_INCLUDE_DIRS})

else()
endif()

# 	d) Link Libraries
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand -lcuda") # silver1
message("Universal Helpers - CUDA_LINK: " ${CUDA_LINK} )
set (CUDA_INTERNAL_LINK_LIB "${CUDA_LINK} -lrt -ldl")
message("Universal Helpers - CUDA_INTERNAL_LINK_LIB: " ${CUDA_INTERNAL_LINK_LIB})
set (PTHREADS_INTERNAL_LINK_LIB "-lpthread")
message("Universal Helpers - PTHREADS_INTERNAL_LINK_LIB: " ${PTHREADS_INTERNAL_LINK_LIB})
set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK  "-L${OPENBLAS_LD} -lopenblas" )
message("Universal Helpers - OPENBLAS_LINK: " ${OPENBLAS_LINK} )
set (HIP_LD ${ROCM_PREFIX}/hip/lib) 
set (HIP_LINK "-L${HIP_LD}") 
message("Universal Helpers - HIP_LINK: " ${HIP_LINK} )
set (HIPRAND_LINK  "-L${ROCM_PREFIX}/lib -lhiprand" )
message("Universal Helpers - HIPRAND_LINK: " ${HIPRAND_LINK} )
set (HIP_INTERNAL_LINK_LIB "${HIP_LINK}")
message("Universal Helpers - HIP_INTERNAL_LINK_LIB: " ${HIP_INTERNAL_LINK_LIB})
set (HIPBLAS_LD ${HIPBLAS_PREFIX}/lib)
set (HIPBLAS_LINK  "-L${HIPBLAS_LD} -lhipblas" )
message("Universal Helpers - HIPBLAS_LINK: " ${HIPBLAS_LINK} )

set(PROJECT_VERSION "1.0")
message("Universal Helpers - PROJECT_VERSION: " ${PROJECT_VERSION})
add_definitions(-DVERSION="${PROJECT_VERSION}")

### Initialize the project
project (CoCopeLia-Helpers VERSION ${PROJECT_VERSION} DESCRIPTION "Universal Helper function lib for CoCoPeLia" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Universal Helpers - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )

set ( UNIHELP_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
message( "Universal Helpers - UNIHELP_DIR: " ${UNIHELP_DIR} )

set (UNIHELP_DIR_INC ${UNIHELP_DIR}/include)
message( "Universal Helpers - Included DIR: " ${UNIHELP_DIR_INC} )
include_directories(${UNIHELP_DIR_INC})

if (${BACKEND} STREQUAL "HIP")
  set (UNIHELP_DIR_SRC ${UNIHELP_DIR}/HIP/src)
  message( "Universal Helpers - Source DIR: " ${UNIHELP_DIR_SRC} )

  set (BACKEND_UNIHELP_DIR_SRC ${UNIHELP_DIR}/HIP/hip_backend)
  message( "Universal Helpers - BACKEND DIR: " ${BACKEND_UNIHELP_DIR_SRC} )
  include_directories(${BACKEND_UNIHELP_DIR_SRC})

else ()
  set (UNIHELP_DIR_SRC ${UNIHELP_DIR}/src)
  message( "Universal Helpers - Source DIR: " ${UNIHELP_DIR_SRC} )

  set (BACKEND_UNIHELP_DIR_SRC ${UNIHELP_DIR}/${BACKEND})
  message( "Universal Helpers - BACKEND DIR: " ${BACKEND_UNIHELP_DIR_SRC} )
  include_directories(${BACKEND_UNIHELP_DIR_SRC})

endif()

### Install Dir
if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
		message("Universal Helpers - CMAKE_INSTALL_PREFIX not provided (with --install-prefix <directory>) - project will be installed at default build dir.")
		set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${TESTBED_NAME}-install)
endif()
set (CMAKE_INSTALL_PROJECT ${CMAKE_INSTALL_PREFIX})
message("Universal Helpers - CMAKE_INSTALL_PROJECT: " ${CMAKE_INSTALL_PROJECT})

set (UNIHELP_INSTALL ${CMAKE_INSTALL_PROJECT}/Universal_helpers)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL})

set (UNIHELP_INSTALL_LIB ${UNIHELP_INSTALL}/lib)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${UNIHELP_INSTALL_LIB})
message("Universal Helpers - CMAKE_LIBRARY_OUTPUT_DIRECTORY: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set (UNIHELP_INSTALL_INC ${UNIHELP_INSTALL}/include)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL_INC})

## Enable using lazy events; e.g. event backends are created on first record on the corresponding device.
#FIXME: Current CoCoPeLia version only work with lazy events.
add_definitions(-DENABLE_LAZY_EVENTS)

#TODO: Remove this set
set(BUILD_BLAS_WRAPPERS_FLAG 1)
if (${BACKEND} STREQUAL "CuCuBLAS")
	if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)
		add_definitions(-DBUILD_BLAS_WRAPPERS_FLAG)

		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_lib_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_routine_wrappers.cu
		)
	else()
		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp    
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
		)
	endif()

	target_link_libraries(unihelpers ${CUDA_INTERNAL_LINK_LIB} ${OPENBLAS_LINK})
elseif(${BACKEND} STREQUAL "Pthreads")
	if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)
		add_definitions(-DBUILD_BLAS_WRAPPERS_FLAG)

		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp
		${BACKEND_UNIHELP_DIR_SRC}/queues_per_device.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_lib_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_routine_wrappers.cu
		)
	else()
		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp
		${BACKEND_UNIHELP_DIR_SRC}/queues_per_device.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
		)
	endif()
	target_link_libraries(unihelpers ${PTHREADS_INTERNAL_LINK_LIB} ${CUDA_INTERNAL_LINK_LIB} ${OPENBLAS_LINK})

elseif(${BACKEND} STREQUAL "HIP")
	if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)
		add_definitions(-DBUILD_BLAS_WRAPPERS_FLAG)
		
		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_lib_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_routine_wrappers.cpp
		)
	else()
		add_library(unihelpers SHARED
		${UNIHELP_DIR_SRC}/general_utils.cpp
		${UNIHELP_DIR_SRC}/data_management.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cpp
		${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cpp
		)
	endif()

	target_link_libraries(unihelpers ${HIP_INTERNAL_LINK_LIB} ${OPENBLAS_LINK} ${CUDA_INTERNAL_LINK_LIB} ${HIPBLAS_LINK} ${HIPRAND_LINK})

else()
	message(FATAL_ERROR "Universal Helpers - BACKEND flag error: BACKEND=" ${BACKEND} )
endif()

configure_file(${UNIHELP_DIR_INC}/unihelpers.hpp ${UNIHELP_INSTALL_INC}/unihelpers.hpp COPYONLY)
## FIXME: This is a temporary cheat for deployment, must be removed at some point (?). It has to do with TransposeTranslate.
configure_file(${BACKEND_UNIHELP_DIR_SRC}/backend_wrappers.hpp ${UNIHELP_INSTALL_INC}/backend_wrappers.hpp COPYONLY)


set (UNIHELP_INSTALL_INC ${UNIHELP_INSTALL_INC} PARENT_SCOPE)
message( "->\nUniversal Helpers - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Universal Helpers - Shared Library headers: " ${UNIHELP_INSTALL_INC} )
message( "------------------------------------------------------------------------------------------------")

# Testing
enable_testing()
if (${BACKEND} STREQUAL "CuCuBLAS" OR ${BACKEND} STREQUAL "HIP")
	add_subdirectory(testing)
elseif(${BACKEND} STREQUAL "Pthreads")
	add_subdirectory(testing_pthreads)
else()
endif()

#set_target_properties(unihelpers PROPERTIES VERSION ${PROJECT_VERSION})
#set_target_properties(unihelpers PROPERTIES SOVERSION 0)
#set_target_properties(unihelpers PROPERTIES PUBLIC_HEADER ${PROJECT_BACKEND_LIB_DIR_INC}/unihelpers.hpp)
#target_include_directories(mylib PRIVATE .)
#target_include_directories(mylib PRIVATE include)
#target_include_directories(mylib PRIVATE src)

#include(GNUInstallDirs)
#install(TARGETS unihelpers
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#configure_file(unihelpers.pc.in unihelpers.pc @ONLY)
#install(FILES ${CMAKE_BINARY_DIR}/unihelpers.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)