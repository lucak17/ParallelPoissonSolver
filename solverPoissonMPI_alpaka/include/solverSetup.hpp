#ifndef SOLVERSETUP_HPP
#define SOLVERSETUP_HPP

#pragma once
#include <alpaka/alpaka.hpp>

constexpr int DIM=3;

using T_data=double;


// accelerator Alpaka
using Dim = alpaka::DimInt<3>;
using Idx = std::size_t;
using Acc = alpaka::AccGpuHipRt<Dim,Idx>;
using T_Host = alpaka::Dev<alpaka::PlatformCpu>;
using T_Dev = alpaka::Dev<alpaka::Platform<Acc>>;

constexpr std::array<int,3> guards={1,1,1};
constexpr alpaka::Vec<Dim, Idx> blockExtentFixed = {8,16,8};
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

constexpr T_data tollScalingFactor = 1e-13;

// Order Neuman BCs scheme
constexpr int orderNeumanBcs=2;

constexpr int tollMainSolver=1; //effective toll = tollMainSolver*tollScalingFactor
constexpr int iterMaxMainSolver=1500;
constexpr bool trackErrorFromIterationHistory=1;

// preconditioner
constexpr int tollPreconditionerSolver=1e4;
constexpr int iterMaxPreconditioner=150;


// chebyshev preconditioner
constexpr T_data epsilon=1e-4;
constexpr T_data rescaleEigMin= 600;
constexpr T_data rescaleEigMax= 1 - 5e-4;
constexpr int chebyshevMax=40;



template<int DIM,typename T_data>
class ExactSolutionAndBCs
{   
    public:
        inline T_data setFieldB(const T_data x, const T_data y, const T_data z) const
        {
            return -sin(x) - cos(y) - 3*sin(z) + 2*y*z + 2; 
            //return -sin(x) - cos(y) - 3*sin(z) + 2 ;
            //return -sin(x);
            //return -sin(x) - cos(y) + 2;
            //return -sin(x) - cos(y); 
        }
        inline T_data trueSolutionFxyz(const T_data x, const T_data y, const T_data z) const
        {
            //return sin(x) + x*x +y*y + z*z;
            return sin(x) + cos(y) + 3*sin(z) + x*x*y*z + x*x + 10;
            //return sin(x) + cos(y) + 3*sin(z) + x*x + y + z + 10;
            //return sin(x) + cos(y) + x*x + y;
            //return sin(x)  + cos(y) + x + 2*y + 10;
            //return sin(x) + 2*x;
            //return x*x*x - 2*x*x + 10;
            //return sin(x) + cos(y) +  10;
        }
        inline T_data trueSolutionDdir(const T_data x, const T_data y, const T_data z, const int dir) const
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
        inline T_data trueSolutionDdirX(const T_data x, const T_data y, const T_data z) const 
        {
            return cos(x) + 2*x*y*z + 2*x;
            //return cos(x) + 2*x;
            //return cos(x) + 1;
        }
        inline T_data trueSolutionDdirY(const T_data x, const T_data y, const T_data z) const
        {
            return -sin(y) + x*x*z;
            //return -sin(y) + 1;
        }
        inline T_data trueSolutionDdirZ(const T_data x, const T_data y, const T_data z) const
        {
            return 3*cos(z) + x*x*y;
            //return 3*cos(z) + 1;
        }
};


#endif

