# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /cfs/klemming/pdc/software/dardel/eb/software/cmake/3.27.7/bin/cmake

# The command to remove a file.
RM = /cfs/klemming/pdc/software/dardel/eb/software/cmake/3.27.7/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build

# Utility rule file for alpakaIde.

# Include any custom commands dependencies for this target.
include thirdParty/alpaka/CMakeFiles/alpakaIde.dir/compiler_depend.make

# Include the progress variables for this target.
include thirdParty/alpaka/CMakeFiles/alpakaIde.dir/progress.make

alpakaIde: thirdParty/alpaka/CMakeFiles/alpakaIde.dir/build.make
.PHONY : alpakaIde

# Rule to build all files generated by this target.
thirdParty/alpaka/CMakeFiles/alpakaIde.dir/build: alpakaIde
.PHONY : thirdParty/alpaka/CMakeFiles/alpakaIde.dir/build

thirdParty/alpaka/CMakeFiles/alpakaIde.dir/clean:
	cd /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/thirdParty/alpaka && $(CMAKE_COMMAND) -P CMakeFiles/alpakaIde.dir/cmake_clean.cmake
.PHONY : thirdParty/alpaka/CMakeFiles/alpakaIde.dir/clean

thirdParty/alpaka/CMakeFiles/alpakaIde.dir/depend:
	cd /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/thirdParty/alpaka /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/thirdParty/alpaka /cfs/klemming/home/p/pennati/solverPoisson/solverPoissonMPI_alpaka/build/thirdParty/alpaka/CMakeFiles/alpakaIde.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : thirdParty/alpaka/CMakeFiles/alpakaIde.dir/depend

