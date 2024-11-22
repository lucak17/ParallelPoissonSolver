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
#include "kernelsAlpaka.hpp"


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
        sigma_( theta_/delta_ )
    {
    }

    ~ChebyshevIterationAlpaka()
    {
    }

    void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB)
    {
        int i,j,k,indx;
        const int imin=this->indexLimitsSolver_[0];
        const int imax=this->indexLimitsSolver_[1];
        const int jmin=this->indexLimitsSolver_[2];
        const int jmax=this->indexLimitsSolver_[3];
        const int kmin=this->indexLimitsSolver_[4];
        const int kmax=this->indexLimitsSolver_[5];

        T_data rhoOld=1/sigma_;
        T_data rhoCurr=1/(2*sigma_ - rhoOld);

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

        // apply si
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
    const AlpakaHelper<DIM,T_data>& alpakaHelper_;
    //T_Preconditioner preconditioner;
    
    const T_data theta_;
    const T_data delta_;
    const T_data sigma_;
    
    T_data toll_;

    T_data* fieldY;
    T_data* fieldZ;
    T_data* fieldW;   
    T_data* fieldBtmp;
};