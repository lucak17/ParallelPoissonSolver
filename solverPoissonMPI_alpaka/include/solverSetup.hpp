#ifndef SOLVERSETUP_HPP
#define SOLVERSETUP_HPP

#pragma once
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iomanip> 
#include <ctime>

inline constexpr int DIM=3; // domain dimensions (1, 2 or 3)

using T_data=double; // data precision of the main solver - double or float (keep it to double)
using T_data_base=double; // must be equal to T_data - it must be explicited here due to alpaka kernels instantiation
using T_data_chebyshev=double; //data precision for the chebyshev preconditioner iteration


// accelerator Alpaka
using Dim = alpaka::DimInt<3>; // alpaka dimension is set to 3 - keep to 3 even if DIM=1 or DIM=2
using Idx = std::size_t; // alpaka index data type 
//using Acc = alpaka::AccGpuHipRt<Dim,Idx>; // alapaka backend --> only alapaka setup to change according to the hardware architecture - must be coherent with CMakeLists.txt
using Acc = alpaka::AccCpuOmp2Blocks<Dim,Idx>; // blocks parallel and threads sequentials --> chose this for openMP!
//using Acc = alpaka::AccCpuOmp2Threads<Dim,Idx>; // blocks sequential and threads parallel
/*
#ifdef alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE
        using Acc = alpaka::AccCpuOmp2Threads<Dim,Idx>;
#endif
#ifdef alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE
        using Acc = alpaka::AccCpuOmp2Blocks<Dim,Idx>;
#endif
#ifdef alpaka_ACC_GPU_HIP_ENABLE
        using Acc = alpaka::AccGpuHipRt<Dim,Idx>;
#endif
#ifdef alpaka_ACC_GPU_CUDA_ENABLE
        using Acc = alpaka::AccGpuCudaRt<Dim,Idx>;
#endif
*/

using T_Host = alpaka::Dev<alpaka::PlatformCpu>;
using T_Dev = alpaka::Dev<alpaka::Platform<Acc>>;

inline constexpr std::array<int,3> guards={1,1,1}; // halo-points in each direction x,y,z - for a second order centered FD scheme must be {1,1,1}
inline constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,1,1};
//inline constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,8,8};
//inline constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,1,32};
inline constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {guards[2],guards[1],guards[0]};
//inline constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {0,guards[1],guards[0]};
//inline constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {0,0,guards[0]};
inline constexpr Idx sMemSizeFixed = (blockExtentFixed[0]+2*haloSizeFixed[0])*(blockExtentFixed[1]+2*haloSizeFixed[1])*(blockExtentFixed[2]+2*haloSizeFixed[2]);

inline constexpr T_data PI = 3.141592653589793;

inline constexpr T_data tollScalingFactor = 1e-10;

// Order Neuman BCs scheme (only 2nd available)
inline constexpr int orderNeumanBcs=2;

// Main Solver BiCGSTAB parameters
inline constexpr int tollMainSolver=1;  // Main solver tollerance --> effective toll = tollMainSolver*tollScalingFactor --> needed since only int can be passed as template arguments
inline constexpr int iterMaxMainSolver=1000; // maximum iterations main solver
inline constexpr bool trackErrorFromIterationHistory=1; // set to 1 to save the residual history (to save residual history in output file set writeResidual = true in inputParam.hpp)

// Iterative BiCGSTAB preconditioner parameters - both for Global-BiCGS and BJ-BiCGS preconditioners
inline constexpr int tollPreconditionerSolver=tollMainSolver*1e8;
inline constexpr int iterMaxPreconditioner=500;


// chebyshev preconditioner paramters - if Chebyshev iteration is used as main solver these paramters must be changed accordingly
inline constexpr T_data epsilon=0; // keep to 0
inline constexpr T_data rescaleEigMin= 100; // rescale factor for minimum eigenvalue (if the chebyshev iteration is set to be local in inputParameter.hpp eigenvalues are not rescaled disregarding these values)
inline constexpr T_data rescaleEigMax= 1 - 1e-4; // rescale factor for maximum eigenvalue
inline constexpr int chebyshevMax=24; // number of chebyshev iterations
inline constexpr int jumpCheb = 0; // change kernel2 in the chebyshev iteration !keep to 0! - 0 -> no block shared memory, 1 -> shared memory and set how many grid points to jump to avoid memory bank conflicts (it works but not sure is effective), 2 -> basic shared memory
// if jumpCheb = 1 set the number of grid points to jump
inline constexpr Idx jumpI=1u;
inline constexpr Idx jumpJ=1u;
inline constexpr Idx jumpK=1u;



// function to set MPI_Datatype according to T_data in MPI calls
template <typename T_data>
inline MPI_Datatype getMPIType();
template <>
inline MPI_Datatype getMPIType<float>() {
    return MPI_FLOAT;
}
template <>
inline MPI_Datatype getMPIType<double>() {
    return MPI_DOUBLE;
}



// set up test problem 
template<int DIM,typename T_data>
class ExactSolutionAndBCs
{   
    public:
        ALPAKA_FN_HOST_ACC inline T_data setFieldB(const T_data x, const T_data y, const T_data z) const
        {
            if constexpr(DIM==3)
            {
                return -sin(x) - cos(y) - 3*sin(z) + 2*y*z + 2; 
            }
            else if constexpr (DIM==2)
            {
                return -sin(x) - cos(y)  + 2*y + 2;
            } 
            else
            {
                return -sin(x) + 4;
            }
        }
        ALPAKA_FN_HOST_ACC inline T_data trueSolutionFxyz(const T_data x, const T_data y, const T_data z) const
        {
            if constexpr(DIM==3)
            {
                return sin(x) + cos(y) + 3*sin(z) + x*x*y*z + x*x + 10; 
            }
            else if constexpr (DIM==2)
            {
                return sin(x) + cos(y) + x*x*y + x*x + 10;
            } 
            else
            {
                return sin(x) + 2*x*x + 10;
            }
        }
        ALPAKA_FN_HOST_ACC inline T_data trueSolutionDdir(const T_data x, const T_data y, const T_data z, const int dir) const
        {
            if constexpr (DIM == 1)
            {
                return trueSolutionDdirX(x,y,z);
            }
            else if constexpr(DIM == 2 || DIM == 3)
            {
                if(dir == 0)
                {
                    return trueSolutionDdirX(x,y,z);
                }
                else if(dir == 1)
                {
                    return trueSolutionDdirY(x,y,z);
                }
                else if(dir == 2)
                {
                    return trueSolutionDdirZ(x,y,z);
                }
                else 
                {
                    return -100;
                }
            }
        }
        
    private:
        ALPAKA_FN_HOST_ACC inline T_data trueSolutionDdirX(const T_data x, const T_data y, const T_data z) const 
        {
            if constexpr(DIM==3)
            {
                return cos(x) + 2*x*y*z + 2*x; 
            }
            else if constexpr (DIM==2)
            {
                return cos(x) + 2*x*y + 2*x;
            } 
            else
            {
                return cos(x) + 4*x;
            }
        }
        ALPAKA_FN_HOST_ACC inline T_data trueSolutionDdirY(const T_data x, const T_data y, const T_data z) const
        {
            if constexpr(DIM==3)
            {
                return -sin(y) + x*x*z;
            }
            else if constexpr (DIM==2)
            {
                return -sin(y) + x*x;
            } 
            else
            {
                return 0;
            }   
        }
        ALPAKA_FN_HOST_ACC inline T_data trueSolutionDdirZ(const T_data x, const T_data y, const T_data z) const
        {
            if constexpr(DIM==3)
            {
                return 3*cos(z) + x*x*y;
            }
            else if constexpr (DIM==2)
            {
                return 0;
            } 
            else
            {
                return 0;
            }
        }
};



// time counters class
template<int DIM,typename T_data>
class TimeCounter
{
    public:
    TimeCounter():
    timeTotPreconditioner1(std::chrono::duration<double>(0)),
    timeTotPreconditioner2(std::chrono::duration<double>(0)),
    timeTotCommunication1(std::chrono::duration<double>(0)),
    timeTotCommunication2(std::chrono::duration<double>(0)),
    timeTotKernel1(std::chrono::duration<double>(0)),
    timeTotKernel2(std::chrono::duration<double>(0)),
    timeTotKernel3(std::chrono::duration<double>(0)),
    timeTotKernel4(std::chrono::duration<double>(0)),
    timeTotKernel5(std::chrono::duration<double>(0)),
    timeTotKernel6(std::chrono::duration<double>(0)),
    timeTotResetNeumanBCs(std::chrono::duration<double>(0)),
    timeTotAllReduction1(std::chrono::duration<double>(0)),
    timeTotAllReduction2(std::chrono::duration<double>(0))
    {}

    std::chrono::duration<double> getTimeTotal()
    {
        timeTotal = timeTotPreconditioner1
                    + timeTotPreconditioner2
                    + timeTotCommunication1
                    + timeTotCommunication2
                    + timeTotKernel1
                    + timeTotKernel2
                    + timeTotKernel3
                    + timeTotKernel4
                    + timeTotKernel5
                    + timeTotKernel6
                    + timeTotResetNeumanBCs
                    + timeTotAllReduction1
                    + timeTotAllReduction2;
        return timeTotal;
    }



    void printAverageTime(const int numCycles)
    {
        std::cout << "Time in milliseconds " << std::endl;
        std::cout << "timeTotPreconditioner1 " << timeTotPreconditioner1.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotPreconditioner2 " << timeTotPreconditioner2.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotCommunication1 " << timeTotCommunication1.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotCommunication2 " << timeTotCommunication2.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotAllReduction1 " << timeTotAllReduction1.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotAllReduction2 " << timeTotAllReduction2.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel1 " << timeTotKernel1.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel2 " << timeTotKernel2.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel3 " << timeTotKernel3.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel4 " << timeTotKernel4.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel5 " << timeTotKernel5.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotKernel6 " << timeTotKernel6.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotPreconditionerTot " << (timeTotPreconditioner1.count() + timeTotPreconditioner2.count()) *1000 / numCycles << std::endl;
        std::cout << "timeTotCommunicationTot " << (timeTotCommunication1.count() + timeTotCommunication2.count()) *1000 / numCycles << std::endl;
        std::cout << "timeTotAllReductionTot " << (timeTotAllReduction1.count() + timeTotAllReduction2.count()) *1000 / numCycles << std::endl;
        std::cout << "timeTotKernelsBBiCGstabTot " << (timeTotKernel1.count() + timeTotKernel2.count() + timeTotKernel3.count() + 
                                                        timeTotKernel4.count() + timeTotKernel5.count() + timeTotKernel6.count() ) *1000 / numCycles << std::endl;
        std::cout << "timeTotResetNeumanBCs " << timeTotResetNeumanBCs.count() *1000 / numCycles << std::endl;
        std::cout << "timeTotal " << this->getTimeTotal().count() *1000 / numCycles << std::endl;
    }

    //private:
    std::chrono::duration<double> timeTotPreconditioner1;
    std::chrono::duration<double> timeTotPreconditioner2;
    std::chrono::duration<double> timeTotCommunication1;
    std::chrono::duration<double> timeTotCommunication2;
    std::chrono::duration<double> timeTotKernel1;
    std::chrono::duration<double> timeTotKernel2;
    std::chrono::duration<double> timeTotKernel3;
    std::chrono::duration<double> timeTotKernel4;
    std::chrono::duration<double> timeTotKernel5;
    std::chrono::duration<double> timeTotKernel6;
    std::chrono::duration<double> timeTotResetNeumanBCs;
    std::chrono::duration<double> timeTotAllReduction1;
    std::chrono::duration<double> timeTotAllReduction2;
    std::chrono::duration<double> timeTotal;
};
#endif
