# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(Integrals)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify where to place shared libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pyTalbot/lib)

# Add source files to compile
add_library(integrals_quad SHARED
    pyTalbot/src/integrals_quad.cpp
)

add_library(integrals_levin SHARED
    pyTalbot/src/integrals_levin.cpp
)

# Links the compiled files to GSL 
find_package(GSL REQUIRED)
target_link_libraries(integrals_quad GSL::gsl GSL::gslcblas)

# Specify the path to the bestlime library
set(BESTLIME_LIB_DIR "/usr/local/lib")

# Find the bestlime library
find_library(BESTLIME_LIBRARY
    NAMES bestlime
    PATHS ${BESTLIME_LIB_DIR}
    NO_DEFAULT_PATH
)

# Add include directory
include_directories(/usr/local/include)

# Link your program with bestlime
target_link_libraries(integrals_levin PRIVATE ${BESTLIME_LIBRARY})