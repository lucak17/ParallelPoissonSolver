#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once
#include <alpaka/alpaka.hpp>
#include "iterativeSolverBase.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpakaChebyshev.hpp"
//#include "operationGrid.hpp"

template <int DIM, typename T_data, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class ChebyshevIterationAlpaka : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    ChebyshevIterationAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                        CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI),
        alpakaHelper_(alpakaHelper),
        theta_( (this->eigenValuesGlobal_[0]*rescaleEigMin + this->eigenValuesGlobal_[1]*rescaleEigMax) * 0.5 * (1.0 + epsilon) ),
        delta_( (this->eigenValuesGlobal_[0]*rescaleEigMin - this->eigenValuesGlobal_[1]*rescaleEigMax) * 0.5 ),
        //theta_( (this->eigenValuesLocal_[0]*rescaleEigMin + this->eigenValuesLocal_[1]*rescaleEigMax) * 0.5 * (1.0 + epsilon) ),
        //delta_( (this->eigenValuesLocal_[0]*rescaleEigMin - this->eigenValuesLocal_[1]*rescaleEigMax) * 0.5 ),
        sigma_( theta_/delta_ ),
        bufY(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufW(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufZ(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        Abuf(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        cfgExtent_({this->alpakaHelper_.extent_, this->alpakaHelper_.elemPerThread_}),
        cfgExtentNoHalo_({this->alpakaHelper_.extentNoHalo_, this->alpakaHelper_.elemPerThread_}),
        workDivExtentKernel1(alpaka::getValidWorkDiv(cfgExtentNoHalo_, this->alpakaHelper_.devAcc_, chebyshev1Kernel, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_) ),
        workDivExtentKernel2(alpaka::getValidWorkDiv(cfgExtentNoHalo_, this->alpakaHelper_.devAcc_, chebyshev2Kernel, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentKernel2Fixed(alpaka::WorkDivMembers<Dim,Idx>(this->alpakaHelper_.numBlocksGrid_,blockExtentFixed,this->alpakaHelper_.elemPerThread_)),
        workDivExtentAssign1(alpaka::getValidWorkDiv(cfgExtent_, this->alpakaHelper_.devAcc_, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufZ), alpaka::experimental::getMdSpan(this->bufY), static_cast<T_data>(1.0)/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_))
     
        
    {
        tmp = new T_data[this->ntotlocal_guards_];
        std::fill(tmp, tmp + this->ntotlocal_guards_, 0.0);
        setToZeroBuffers();
    }

    ~ChebyshevIterationAlpaka()
    {
        delete[] tmp;
    }

    void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx> bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx> bufB)
    {   

        alpaka::Queue<Acc, alpaka::Blocking> queue{this->alpakaHelper_.devAcc_};

        //auto fieldTmpView = createView(this->alpakaHelper_.devHost_, this->tmp, this->alpakaHelper_.extent_);
        //memcpy(queue, fieldTmpView, bufB, this->alpakaHelper_.extent_);


        //this->printFieldWithGuards(this->tmp);

        int i,j,k,indx;
        const int imin=this->indexLimitsSolver_[0];
        const int imax=this->indexLimitsSolver_[1];
        const int jmin=this->indexLimitsSolver_[2];
        const int jmax=this->indexLimitsSolver_[3];
        const int kmin=this->indexLimitsSolver_[4];
        const int kmax=this->indexLimitsSolver_[5];

        T_data rhoOld=1/sigma_;
        T_data rhoCurr=1/(2*sigma_ - rhoOld);


        auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);


        //alpaka::KernelCfg<Acc> const cfgExtent = {this->alpakaHelper_.extent_, this->alpakaHelper_.elemPerThread_};
        //alpaka::KernelCfg<Acc> const cfgExtentNoHalo = {this->alpakaHelper_.extentNoHalo_, this->alpakaHelper_.elemPerThread_};
        /*
        StencilKernel<DIM, T_data> stencilKernel;
        auto workDivExtentStencil = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, stencilKernel, alpaka::experimental::getMdSpan(this->Abuf), bufBMdSpan, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_);
        AssignFieldWith1FieldKernel<DIM, T_data> assignFieldWith1FieldKernel;
        auto workDivExtentAssign1 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufZ), bufBMdSpan, static_cast<T_data>(1.0)/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        AssignFieldWith2FieldsKernel<DIM, T_data> assignFieldWith2FieldsKernel;
        auto workDivExtentAssign2 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, assignFieldWith2FieldsKernel, alpaka::experimental::getMdSpan(this->bufY), bufBMdSpan, alpaka::experimental::getMdSpan(this->Abuf),
                                                            static_cast<T_data>(4)*rhoCurr/delta_, static_cast<T_data>(2)*rhoCurr/delta_/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        AssignFieldWith4FieldsKernel<DIM, T_data> assignFieldWith4FieldsKernel;
        auto workDivExtentAssign4 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, assignFieldWith4FieldsKernel, 
                                                            alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufY), bufBMdSpan, alpaka::experimental::getMdSpan(this->Abuf), alpaka::experimental::getMdSpan(this->bufZ),
                                                            static_cast<T_data>(2)*rhoCurr*sigma_, static_cast<T_data>(2)*rhoCurr/delta_, static_cast<T_data>(2)*rhoCurr/delta_, -rhoCurr*rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        */
        //Chebyshev1Kernel<DIM,T_data> chebyshev1Kernel;
        
        /*
        Chebyshev1KernelSharedMem<DIM,T_data> chebyshev1Kernel;
        auto workDivExtentKernel1 = alpaka::getValidWorkDiv(cfgExtentNoHalo, this->alpakaHelper_.devAcc_, chebyshev1Kernel, 
                                                            bufBMdSpan, alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, rhoCurr,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
        Chebyshev2KernelSharedMem<DIM,T_data> chebyshev2Kernel;
        //Chebyshev2Kernel<DIM,T_data> chebyshev2Kernel;
        auto workDivExtentKernel2 = alpaka::getValidWorkDiv(cfgExtentNoHalo, this->alpakaHelper_.devAcc_, chebyshev2Kernel, 
                                                            bufBMdSpan, alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, rhoCurr, rhoOld,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
        auto workDivExtentKernel2Fixed = alpaka::WorkDivMembers<Dim,Idx>(this->alpakaHelper_.numBlocksGrid_,blockExtentFixed,this->alpakaHelper_.elemPerThread_);
        AssignFieldWith1FieldKernel<DIM, T_data> assignFieldWith1FieldKernel;
        auto workDivExtentAssign1 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufZ), bufBMdSpan, static_cast<T_data>(1.0)/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        */

        //this-> template resetNeumanBCs<isMainLoop,false>(fieldB);

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
       
       // alpaka::exec<Acc>(queue, workDivExtentAssign1, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufZ), bufBMdSpan, static_cast<T_data>(1)/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_ );
       // alpaka::exec<Acc>(queue, workDivExtentStencil, stencilKernel, alpaka::experimental::getMdSpan(this->Abuf), bufBMdSpan, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_ );
       // alpaka::exec<Acc>(queue, workDivExtentAssign2, assignFieldWith2FieldsKernel, alpaka::experimental::getMdSpan(this->bufY), bufBMdSpan, alpaka::experimental::getMdSpan(this->Abuf), 
        //                    static_cast<T_data>(4)*rhoCurr/delta_, static_cast<T_data>(2)*rhoCurr/delta_/theta_, this->alpakaHelper_.indexLimitsSolverAlpaka_);

        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<true>(getPtrNative(bufB));
            this->communicatorMPI_.waitAllandCheckRcv();
        }
       
        alpaka::exec<Acc>(queue, workDivExtentKernel1, chebyshev1Kernel, bufBMdSpan, alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), 
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
            alpaka::exec<Acc>(queue, workDivExtentKernel2Fixed, chebyshev2Kernel, bufBMdSpan, alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), 
                                        alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
           // alpaka::exec<Acc>(queue, workDivExtentKernel2, chebyshev2Kernel, bufBMdSpan, alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), 
            //                            alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            
            /*
            alpaka::exec<Acc>(queue, workDivExtentStencil, stencilKernel, alpaka::experimental::getMdSpan(this->Abuf), alpaka::experimental::getMdSpan(this->bufY), this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_);
            alpaka::exec<Acc>(queue, workDivExtentAssign4, assignFieldWith4FieldsKernel, alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufY), bufBMdSpan, 
                                alpaka::experimental::getMdSpan(this->Abuf), alpaka::experimental::getMdSpan(this->bufZ),
                                static_cast<T_data>(2)*rhoCurr*sigma_, static_cast<T_data>(2)*rhoCurr/delta_, static_cast<T_data>(2)*rhoCurr/delta_, -rhoCurr*rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            */
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
       alpaka::exec<Acc>(queue, workDivExtentAssign1, assignFieldWith1FieldKernel, bufXMdSpan, alpaka::experimental::getMdSpan(this->bufW), static_cast<T_data>(-1.0), this->alpakaHelper_.indexLimitsSolverAlpaka_ );

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

    void setToZeroBuffers()
    {
        alpaka::Queue<Acc, alpaka::Blocking> queue{this->alpakaHelper_.devAcc_};
        auto tmpView = createView(this->alpakaHelper_.devHost_, this->tmp, this->alpakaHelper_.extent_);

        memcpy(queue, bufY, tmpView, this->alpakaHelper_.extent_);
        memcpy(queue, bufW, tmpView, this->alpakaHelper_.extent_);
        memcpy(queue, bufZ, tmpView, this->alpakaHelper_.extent_);
        memcpy(queue, Abuf, tmpView, this->alpakaHelper_.extent_);
    }

    const AlpakaHelper<DIM,T_data>& alpakaHelper_;
    //T_Preconditioner preconditioner;
    
    const T_data theta_;
    const T_data delta_;
    const T_data sigma_;
    
    T_data toll_;
    T_data* tmp;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufY;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufW;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufZ;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> Abuf;

    alpaka::KernelCfg<Acc> const cfgExtent_;
    alpaka::KernelCfg<Acc> const cfgExtentNoHalo_;
    Chebyshev1KernelSharedMem<DIM,T_data> chebyshev1Kernel;
    Chebyshev2KernelSharedMem<DIM,T_data> chebyshev2Kernel;
    AssignFieldWith1FieldKernel<DIM, T_data> assignFieldWith1FieldKernel;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Fixed;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentAssign1;
};
