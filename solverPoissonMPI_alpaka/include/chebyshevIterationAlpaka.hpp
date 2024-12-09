#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once
#include <alpaka/alpaka.hpp>
#include "iterativeSolverBaseAlpaka.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpakaChebyshev.hpp"
//#include "operationGrid.hpp"

template <int DIM, typename T_data, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class ChebyshevIterationAlpaka : public IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>{
    public:

    ChebyshevIterationAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                        CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        preconditioner(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        theta_( (this->eigenValuesGlobal_[0]*rescaleEigMin + this->eigenValuesGlobal_[1]*rescaleEigMax) * 0.5 * (1.0 + epsilon) ),
        delta_( (this->eigenValuesGlobal_[0]*rescaleEigMin - this->eigenValuesGlobal_[1]*rescaleEigMax) * 0.5 ),
        //theta_( (this->eigenValuesLocal_[0]*rescaleEigMin + this->eigenValuesLocal_[1]*rescaleEigMax) * 0.5 * (1.0 + epsilon) ),
        //delta_( (this->eigenValuesLocal_[0]*rescaleEigMin - this->eigenValuesLocal_[1]*rescaleEigMax) * 0.5 ),
        sigma_( theta_/delta_ ),
        bufY(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufW(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufZ(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        Abuf(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        workDivExtentKernel1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev1Kernel, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_) ),
        workDivExtentKernel1Shared(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev1KernelShared, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_) ),
        workDivExtentKernel2Shared(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev2KernelShared, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentKernel1Solver(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev1KernelSharedMemSolver, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, theta_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentKernel2Fixed(alpaka::WorkDivMembers<Dim,Idx>(this->alpakaHelper_.numBlocksGrid_,blockExtentFixed,this->alpakaHelper_.elemPerThread_)),
        workDivExtentKernel2Solver(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev2KernelSharedMemSolver, 
                                                           alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentKernel2SharedMemeLoop1D(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev2KernelSharedMemLoop1D, 
                                                           alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentAssign1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufZ), alpaka::experimental::getMdSpan(this->bufY), static_cast<T_data>(1.0)/theta_, 
                            this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.haloSize_))
    {
        
        constexpr std::uint8_t value0{0}; 
        alpaka::memset(this->queueSolverNonBlocking1_, bufY, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufW, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufZ, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, Abuf, value0);


        if(this->my_rank_==0)
        {
            std::cout << "Chebyshev preconditioner" <<
                        "\n workDivExtentKernel1Solver:\t" << workDivExtentKernel1Solver <<
                        "\n workDivExtentKernel2Solver:\t" << workDivExtentKernel2Solver <<
                        "\n workDivExtentKernel2SharedMemeLoop1D:\t" << workDivExtentKernel2SharedMemeLoop1D <<
                        "\n workDivExtentKernel2Shared:\t" << workDivExtentKernel2Shared <<
                        "\n workDivExtentKernel2Fixed:\t" << workDivExtentKernel2Fixed  << std::endl;
            std::cout<< "Min eigenvalue "<< this->eigenValuesGlobal_[0]*rescaleEigMin << "\nMax eigenvalue "<< this->eigenValuesGlobal_[1]*rescaleEigMax << std::endl;
        }

        alpaka::wait(this->queueSolverNonBlocking1_);
    }

    ~ChebyshevIterationAlpaka()
    {
    }

    void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB)
    {   
        T_data rhoOld=1/sigma_;
        T_data rhoCurr=1/(2*sigma_ - rhoOld);


        auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);
        auto bufYMdSpan = alpaka::experimental::getMdSpan(this->bufY);
        auto bufWMdSpan = alpaka::experimental::getMdSpan(this->bufW);
        auto bufZMdSpan = alpaka::experimental::getMdSpan(this->bufZ);


        this-> template resetNeumanBCsAlpaka<isMainLoop,false>(bufB);

        auto startSolver = std::chrono::high_resolution_clock::now();
        
        // apply s0 and s1
        /*
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldZ[indx] = fieldB[indx]/theta_;
                    fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                }
            }
        }
        */
        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<true>(getPtrNative(bufB));
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, chebyshev1Kernel, bufBMdSpan, bufYMdSpan, bufZMdSpan, 
                       delta_, theta_, rhoCurr, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);



        // apply si
        /*
        for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
        {   
            rhoOld=rhoCurr;
            rhoCurr=1/(2*sigma_ - rhoOld);
            if constexpr (communicationON)
            {
                this->communicatorMPI_(fieldY);
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                    }
                }
            }
            std::swap(fieldZ,fieldY);
            std::swap(fieldW,fieldY);
        }
        */
        
        for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
        {   
            rhoOld=rhoCurr;
            rhoCurr=1/(2*sigma_ - rhoOld);
            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<true>(getPtrNative(bufY));
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            
            this-> template resetNeumanBCsAlpaka<isMainLoop,false>(bufY);
            
            if constexpr (jumpCheb == 1)
            {
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2SharedMemeLoop1D, chebyshev2KernelSharedMemLoop1D, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                            bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                            this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_);
            }
            else if constexpr (jumpCheb == 2)
            {
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Shared, chebyshev2KernelShared, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                        bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                        this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            }
            else
            {
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Solver, chebyshev2Kernel, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
            }
            
            std::swap(this->bufZ,this->bufY);
            std::swap(this->bufW,this->bufY);
        }

        /*
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldX[indx] = (-1)*fieldW[indx];
                }
            }
        }
        */

        alpaka::exec<Acc>(this->queueSolver_, workDivExtentAssign1, assignFieldWith1FieldKernel, bufXMdSpan, bufWMdSpan, static_cast<T_data>(-1.0),
                         this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.haloSize_);

        auto endSolver = std::chrono::high_resolution_clock::now();
        /*
        if constexpr (isMainLoop)
        {
            std::swap(fieldBtmp,fieldB);
            this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,fieldBtmp,operatorA);
            this->errorFromIteration_ = this->errorComputeOperator_;
            this->numIterationFinal_ = maxIteration;
            this->durationSolver_ = endSolver - startSolver;
        }
        */
    }
    
    
    private:

    

    T_Preconditioner preconditioner;
    
    const T_data theta_;
    const T_data delta_;
    const T_data sigma_;
    
    T_data toll_;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufY;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufW;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufZ;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> Abuf;

    Chebyshev1Kernel<DIM,T_data> chebyshev1Kernel;
    Chebyshev2Kernel<DIM,T_data> chebyshev2Kernel;
    Chebyshev1KernelSharedMem<DIM,T_data> chebyshev1KernelShared;
    Chebyshev2KernelSharedMem<DIM,T_data> chebyshev2KernelShared;
    Chebyshev1KernelSharedMemSolver<DIM, T_data> chebyshev1KernelSharedMemSolver;
    Chebyshev2KernelSharedMemSolver<DIM, T_data> chebyshev2KernelSharedMemSolver;
    Chebyshev2KernelSharedMemLoop1D<DIM, T_data> chebyshev2KernelSharedMemLoop1D;
    AssignFieldWith1FieldKernel<DIM, T_data> assignFieldWith1FieldKernel;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1Shared;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Shared;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1Solver;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Fixed;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Solver;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2SharedMemeLoop1D;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentAssign1;
};
