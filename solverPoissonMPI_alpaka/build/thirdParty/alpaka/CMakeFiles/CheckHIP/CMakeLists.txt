cmake_minimum_required(VERSION 3.27.7)
project(CheckHIP HIP)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_HIP_COMPILER \"${CMAKE_HIP_COMPILER}\")\n"
  "\n"
  )