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
add_library(integrals SHARED
    pyTalbot/src/integrals.cpp
)

# Links the compiled files to GSL 
find_package(GSL REQUIRED)
target_link_libraries(integrals GSL::gsl GSL::gslcblas)

