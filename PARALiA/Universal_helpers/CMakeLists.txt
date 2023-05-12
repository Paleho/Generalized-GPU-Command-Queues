cmake_minimum_required (VERSION 3.10)

# 	d) Link Libraries
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

if (${BACKEND} STREQUAL "CuCuBLAS")
	if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)

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

	message("Universal Helpers - INTERNAL_LINK_LIB: " ${INTERNAL_LINK_LIB} )
	message("Universal Helpers - OPENBLAS_LINK: " ${OPENBLAS_LINK} )
	target_link_libraries(unihelpers ${INTERNAL_LINK_LIB} ${OPENBLAS_LINK})

elseif(${BACKEND} STREQUAL "HIP")
	if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)
		
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

# # Testing
# enable_testing()
# if (${BACKEND} STREQUAL "CuCuBLAS" OR ${BACKEND} STREQUAL "HIP")
# 	add_subdirectory(testing)
# else()
# endif()

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
