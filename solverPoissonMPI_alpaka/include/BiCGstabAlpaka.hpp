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


template <int DIM, typename T_data, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class BiCGstabAlpaka : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    BiCGstabAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, CommunicatorMPI<DIM,T_data>& communicatorMPI):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI)
    {
        toll_ = static_cast<T_data>(tolerance) * tollScalingFactor;
        pk = new T_data[this->ntotlocal_guards_];
        rk = new T_data[this->ntotlocal_guards_];
        r0 = new T_data[this->ntotlocal_guards_];
        Mpk = new T_data[this->ntotlocal_guards_];
        AMpk = new T_data[this->ntotlocal_guards_];
        zk = new T_data[this->ntotlocal_guards_];
        Azk = new T_data[this->ntotlocal_guards_];

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(r0, r0 + this->ntotlocal_guards_, 0);
        std::fill(Mpk, Mpk + this->ntotlocal_guards_, 0);
        std::fill(AMpk, AMpk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);
        std::fill(Azk, Azk + this->ntotlocal_guards_, 0);
        
    }

    ~BiCGstabAlpaka()
    {
        delete[] pk;
        delete[] rk;
        delete[] r0;
        delete[] Mpk;
        delete[] AMpk;
        delete[] zk;
        delete[] Azk;
        
    }


    
    void operator()(T_data fieldX[], T_data fieldB[], MatrixFreeOperatorA<DIM,T_data>& operatorA)
    {
        int i,j,k,indx;
        int iter=0;

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(r0, r0 + this->ntotlocal_guards_, 0);
        std::fill(Mpk, Mpk + this->ntotlocal_guards_, 0);
        std::fill(AMpk, AMpk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);
        std::fill(Azk, Azk + this->ntotlocal_guards_, 0);

        const int imin=this->indexLimitsSolver_[0];
        const int imax=this->indexLimitsSolver_[1];
        const int jmin=this->indexLimitsSolver_[2];
        const int jmax=this->indexLimitsSolver_[3];
        const int kmin=this->indexLimitsSolver_[4];
        const int kmax=this->indexLimitsSolver_[5];
            
        T_data alphak=1;
        T_data betak=1;
        T_data omegak=1;
        T_data pSum1=0.0;
        T_data pSum2=0.0;
        T_data totSum1=0.0;
        T_data totSum2=0.0;
        T_data rho0=1;
        T_data rho1=1;


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
            {    std::cout<<"Debug in BiCGSTAB START " << " main loop "<< isMainLoop << " globalLocation " << this->globalLocation_[0] << " "<< this->globalLocation_[1] <<" "<< this->globalLocation_[2] <<
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

        //p0=r0
        std::memcpy(pk, rk, this->ntotlocal_guards_* sizeof(T_data));
        std::memcpy(r0, rk, this->ntotlocal_guards_* sizeof(T_data));
        rho0=1;
        rho1=rho0;
        auto startSolver = std::chrono::high_resolution_clock::now();
        
        while(iter<maxIteration)
        {
            preconditioner(Mpk,pk,operatorA);
            
            if constexpr(communicationON)
            {
                this->communicatorMPI_(Mpk);
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            this-> template resetNeumanBCs<isMainLoop,false>(Mpk);

            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        AMpk[indx] = operatorA(i,j,k,Mpk);
                        pSum2 += r0[indx] * AMpk[indx];
                    }
                }
            }
            if constexpr(communicationON)
            {
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                alphak=rho0/totSum2;
            }
            else
            {
                alphak=rho0/pSum2;
            }
            

            // sk = rk - alphak Apk
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        rk[indx] = rk[indx] - alphak * AMpk[indx];
                    }
                }
            }

            //std::fill(zk, zk + this->ntotlocal_guards_, 0);
            preconditioner(zk,rk,operatorA);
            if constexpr (communicationON)
            {
                this->communicatorMPI_(zk);
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            this-> template resetNeumanBCs<isMainLoop,false>(zk);
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        Azk[indx] = operatorA(i,j,k,zk);
                    }
                }
            }

            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        pSum1 +=  rk[indx] * Azk[indx];
                        pSum2 += Azk[indx] * Azk[indx];
                    }
                }
            }

            if  constexpr (communicationON)
            {   
                MPI_Allreduce(&pSum1, &totSum1, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                omegak=totSum1/totSum2;  
            }
            else
            {
                omegak=pSum1/pSum2;
            }

            for(i=0; i< this->ntotlocal_guards_; i++)
            {
                fieldX[i] = fieldX[i] + alphak*Mpk[i] + omegak*zk[i];
            }

            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin;j<jmax; j++)
                {
                    for(i=imin;i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        rk[indx] = rk[indx] - omegak *  Azk[indx];
                        pSum1 += r0[indx] * rk[indx];
                        pSum2 += rk[indx] * rk[indx];
                    }
                }
            }
            if constexpr (communicationON)
            {
                MPI_Allreduce(&pSum1, &rho1, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                this->errorFromIteration_ = std::sqrt(totSum2);
            }
            else
            {
                this->errorFromIteration_ = std::sqrt(pSum2);
                rho1=pSum1;
            }
            betak = rho1/rho0 * alphak / omegak;
            rho0=rho1;

            // pk+1 = rk+1 + beta*(pk - omeagak A(pk) )
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        pk[indx] = rk[indx] + betak * (pk[indx] - omegak * AMpk[indx] );
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
                    std::cout<<" Debug in BiCGSTAB iter "<< iter << " alpha "<< alphak << " omega " << omegak << " rho0 "<< rho0 <<" error "<< this->errorFromIteration_  <<std::endl;
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
    //T_Preconditioner preconditioner;
    T_data toll_;

    T_data* pk;
    T_data* rk;
    T_data* r0;
    T_data* Mpk;
    T_data* AMpk;
    T_data* zk;
    T_data* Azk;
};