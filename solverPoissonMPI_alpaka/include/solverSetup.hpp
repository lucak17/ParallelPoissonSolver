#ifndef SOLVERSETUP_HPP
#define SOLVERSETUP_HPP

#pragma once
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iomanip> 
#include <ctime>

constexpr int DIM=3;

using T_data_base=double; // needed to be explicit here due to alpaka kernels instantiation
using T_data=double;
using T_data_chebyshev=double;


// accelerator Alpaka
using Dim = alpaka::DimInt<3>;
using Idx = std::size_t;
using Acc = alpaka::AccGpuHipRt<Dim,Idx>;
//using Acc = alpaka::AccCpuOmp2Threads<Dim,Idx>;
//using Acc = alpaka::AccCpuOmp2Blocks<Dim,Idx>;
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

constexpr std::array<int,3> guards={1,1,1};
constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,1,1};
//constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,8,8};
//constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {1,1,32};
constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {guards[2],guards[1],guards[0]};
//constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {0,guards[1],guards[0]};
//constexpr alpaka::Vec<Dim, Idx> haloSizeFixed = {0,0,guards[0]};
constexpr Idx sMemSizeFixed = (blockExtentFixed[0]+2*haloSizeFixed[0])*(blockExtentFixed[1]+2*haloSizeFixed[1])*(blockExtentFixed[2]+2*haloSizeFixed[2]);

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


constexpr T_data PI = 3.141592653589793;

constexpr T_data tollScalingFactor = 1e-10;

// Order Neuman BCs scheme (only 2nd available)
constexpr int orderNeumanBcs=2;

constexpr int tollMainSolver=1;  //effective toll = tollMainSolver*tollScalingFactor
constexpr int iterMaxMainSolver=1000;
constexpr bool trackErrorFromIterationHistory=1;

// preconditioner
constexpr int tollPreconditionerSolver=tollMainSolver*1e8;
constexpr int iterMaxPreconditioner=500;


// chebyshev preconditioner
constexpr T_data epsilon=0;
constexpr T_data rescaleEigMin= 10;
constexpr T_data rescaleEigMax= 1 - 1e-4;
constexpr int chebyshevMax=24;
constexpr int jumpCheb = 0;
constexpr Idx jumpI=1u;
constexpr Idx jumpJ=1u;
constexpr Idx jumpK=1u;



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


#endif


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