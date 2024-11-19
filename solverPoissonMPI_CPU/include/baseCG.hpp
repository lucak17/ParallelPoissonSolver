#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once

#include "iterativeSolverBase.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"


template <int DIM, typename T_data, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class BaseCG : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    BaseCG(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, CommunicatorMPI<DIM,T_data>& communicatorMPI):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI),
        preconditioner(blockGrid,exactSolutionAndBCs,communicatorMPI)
    {
        toll_ = static_cast<T_data>(tolerance) * tollScalingFactor;
        pk = new T_data[this->ntotlocal_guards_];
        rk = new T_data[this->ntotlocal_guards_];
        Apk = new T_data[this->ntotlocal_guards_];
        zk = new T_data[this->ntotlocal_guards_];

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(Apk, Apk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);
    }

    ~BaseCG()
    {
        delete[] pk;
        delete[] rk;
        delete[] Apk;
        delete[] zk;        
    }


    
    void operator()(T_data fieldX[], T_data fieldB[], MatrixFreeOperatorA<DIM,T_data>& operatorA)
    {
        int i,j,k,indx;
        int iter=0;

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(Apk, Apk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);

        const int imin=this->indexLimitsSolver_[0];
        const int imax=this->indexLimitsSolver_[1];
        const int jmin=this->indexLimitsSolver_[2];
        const int jmax=this->indexLimitsSolver_[3];
        const int kmin=this->indexLimitsSolver_[4];
        const int kmax=this->indexLimitsSolver_[5];
            
        T_data alphak=1;
        T_data betak=1;
        T_data pSum1=0.0;
        T_data pSum2=0.0;
        T_data totSum1=0.0;
        T_data totSum2=0.0;
        T_data pSumRk=1;
        T_data totSumRk=1;


        if (isMainLoop && communicationON)
        {
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        if constexpr(isMainLoop)
        {
            this-> template resetNeumanBCs<isMainLoop,true>(fieldX);
        }
        else 
        {
            std::fill(fieldX, fieldX + this->ntotlocal_guards_, 0);
        }
        this-> template normalizeProblemToFieldBNorm<isMainLoop,communicationON>(fieldX,fieldB);

        if constexpr(isMainLoop)
        {
            if(this->my_rank_==0)
            {    std::cout<<"Debug in baseCG START " << " main loop "<< isMainLoop << " globalLocation " << this->globalLocation_[0] << " "<< this->globalLocation_[1] <<" "<< this->globalLocation_[2] <<
                " indexLimitsData "<< this->indexLimitsData_[0]<< " "<<this->indexLimitsData_[1] <<" "<< this->indexLimitsData_[2] << " " <<this->indexLimitsData_[3] << " "<< this->indexLimitsData_[4] << " "<< this->indexLimitsData_[5] <<
                " indexLimitsSolver "<< this->indexLimitsSolver_[0]<< " "<<this->indexLimitsSolver_[1] <<" "<< this->indexLimitsSolver_[2] << " " <<this->indexLimitsSolver_[3] << " "<< this->indexLimitsSolver_[4] << " "<< this->indexLimitsSolver_[5] << 
                " norm fieldB " <<this->normFieldB_ <<std::endl;
            }
        }

        // r0= b - A(x0)
        this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,rk,operatorA);
        if constexpr (isMainLoop && trackErrorFromIterationHistory)
        {
            this->errorFromIterationHistory_[0]=this->errorComputeOperator_;
        }
        if(this->errorComputeOperator_ < toll_)
        {   
            this->numIterationFinal_ = 0;
            return;
        }

        preconditioner(zk,rk,operatorA);
        //p0=r0
        std::memcpy(pk, zk, this->ntotlocal_guards_* sizeof(T_data));
        
        auto startSolver = std::chrono::high_resolution_clock::now();
        
        while(iter<maxIteration)
        {
            
            if constexpr(communicationON)
            {
                this->communicatorMPI_(pk);
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            if constexpr (orderNeumanBcs==1)
                this-> template resetNeumanBCs<isMainLoop,false>(pk);

            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        Apk[indx] = operatorA(i,j,k,pk);
                        pSum1 += rk[indx] * zk[indx];
                        pSum2 += pk[indx] * Apk[indx];
                    }
                }
            }
            if constexpr(communicationON)
            {
                MPI_Allreduce(&pSum1, &totSum1, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                alphak=totSum1/totSum2;
            }
            else
            {
                alphak=pSum1/pSum2;
            }
            

            // sk = rk - alphak Apk
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        fieldX[indx] = fieldX[indx] + alphak * pk[indx];
                        rk[indx] = rk[indx] - alphak * Apk[indx];
                    }
                }
            }

            //std::fill(zk, zk + this->ntotlocal_guards_, 0);
            preconditioner(zk,rk,operatorA);
            pSum2=0.0;
            pSumRk=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        pSum2 += rk[indx] * zk[indx];
                        pSumRk += rk[indx] * rk[indx];
                    }
                }
            }
            if constexpr(communicationON)
            {
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSumRk, &totSumRk, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                betak=totSum2/totSum1;
                this->errorFromIteration_ = std::sqrt(totSumRk);
            }
            else
            {
                betak=pSum2/pSum1;
                this->errorFromIteration_ = std::sqrt(pSumRk);
            }

            // pk+1 = rk+1 + beta*(pk - omeagak A(pk) )
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        pk[indx] = zk[indx] + betak * pk[indx];
                    }
                }
            }


            iter++;
            if constexpr (isMainLoop)
            {
                if constexpr (trackErrorFromIterationHistory)
                {
                    this->errorFromIterationHistory_[iter]=this->errorFromIteration_;
                }
                
                if(this->my_rank_==0 && iter%10==0)
                {
                    std::cout<<" Debug in base CG iter "<< iter << " alpha "<< alphak << " beta " << betak << " error "<< this->errorFromIteration_  <<std::endl;
                }
            }

            if(this->errorFromIteration_ < toll_)
            {
                break;
            }
        }

        this->numIterationFinal_ = iter;
        if constexpr(communicationON)
        {   
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        if constexpr(orderNeumanBcs==1)
            this-> template resetNeumanBCs<isMainLoop,true>(fieldX);

        auto endSolver = std::chrono::high_resolution_clock::now();
        this->durationSolver_ = endSolver - startSolver;

        if constexpr (isMainLoop)
        {
            this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,rk,operatorA);
        }

        for(i=0; i < this->ntotlocal_guards_; i++)
        {
            fieldX[i] *= this->normFieldB_;
            fieldB[i] *= this->normFieldB_;
        }
        this->normFieldB_=1;

        if constexpr(communicationON)
        {   
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
    }

    private:
    T_Preconditioner preconditioner;
    T_data toll_;

    T_data* pk;
    T_data* rk;
    T_data* Apk;
    T_data* zk;
};