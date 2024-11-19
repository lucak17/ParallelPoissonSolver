#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once

#include "solverSetup.hpp"


template <int DIM,typename T_data>
class BlockGrid{
    public:

    BlockGrid(const std::array<int, 3>& nranks, const int my_rank, const std::array<int,3>& npglobal, const std::array<T_data,3>& ds, const std::array<T_data,3>& origin, const std::array<int,3>& guards, const std::array<int,6>& bcsType,const std::array<T_data,6>& bcsValue) : 
            nranks_(nranks),
            my_rank_(my_rank),
            globalLocation_(calculateGlobalLocation(nranks,my_rank)),
            npglobal_(npglobal),
            origin_(origin),
            ds_(ds),
            guards_(guards),
            nlocal_noguards_(calculateNlocal_noguards(nranks,npglobal)),
            nlocal_guards_(calculateNlocal_guards(calculateNlocal_noguards(nranks,npglobal),guards)),
            ntotlocal_guards_(calculateNtotlocal(calculateNlocal_guards(calculateNlocal_noguards(nranks,npglobal),guards))),
            ntotlocal_noguards_(calculateNtotlocal(calculateNlocal_noguards(nranks,npglobal))),
            indexLimitsData_(calculateIndexLimitsData(calculateNlocal_noguards(nranks,npglobal),guards)),
            hasCommunication_(setHasCommunication(nranks,calculateGlobalLocation(nranks,my_rank))),
            hasBoundary_(setHasBoundary(nranks,calculateGlobalLocation(nranks,my_rank))),
            bcsType_(bcsType),
            bcsValue_(bcsValue),
            flagBCsSet_(false)
    {
        calculateIndexLimitsSolver();
        computeEigenvalues();
    }

    ~BlockGrid() = default;

    const std::array<int, 3> getNranks() const
    {
        return nranks_;
    }
    const std::array<int,3> getNpglobal() const
    {
        return npglobal_;
    }
    const std::array<T_data,3> getDs() const
    {
        return ds_;
    }
    const std::array<T_data,3> getOrigin() const
    {
        return origin_;
    }
    const std::array<int,3> getGuards() const
    {
        return guards_;
    }
    const int getMyrank() const
    {
        return my_rank_;
    }
    const std::array<int,3> getGlobalLocation() const
    {
        return globalLocation_;
    }
    const std::array<int,3> getNlocalNoGuards() const
    {
        return nlocal_noguards_;
    }
    const std::array<int,3> getNlocalGuards() const
    {
        return nlocal_guards_;
    }
    const std::array<int,6> getIndexLimitsData() const
    {
        return indexLimitsData_;
    }
    const std::array<int,6> getIndexLimitsSolver() const
    {
        return indexLimitsSolver_;
    }
    const std::array<int,6> getIndexLimitsComm() const
    {
        return indexLimitsComm_;
    }
    const int getNumCommunication() const
    {
        return numCommunication_;
    }

    const std::array<int,3> getNumElementsComm() const
    {
        return numElementsComm_;
    }

    const int getNtotLocalGuards() const
    {
        return ntotlocal_guards_;
    }
    const int getNtotLocalNoGuards() const
    {
        return ntotlocal_noguards_;
    }
    const std::array<bool,6> getHasBoundary() const
    {
        return hasBoundary_;
    }
    const std::array<bool,6> getHasCommunication() const
    {
        return hasCommunication_;
    }
    const std::array<int,6> getBcsType() const
    {
        return bcsType_;
    }
    const std::array<T_data,6> getBcsValue() const
    {
        return bcsValue_;
    }

    const bool checkBCsSet() const
    {
        return flagBCsSet_;
    }

    const std::array<T_data,2> getEigenValuesLocal() const
    {
        return eigenValuesLocal_;
    }
    const std::array<T_data,2> getEigenValuesGlobal() const
    {
        return eigenValuesGlobal_;
    }
    const int getNtotNpglobal() const
    {
        int tmp=1;

        for(int i=0; i<DIM; i++)
        {
            tmp=tmp * npglobal_[i];
        }   
        return tmp;
    }


    
    private:

    std::array<int,3> calculateGlobalLocation(const std::array<int, 3>& nranks, const int my_rank) const
    {
        std::array<int,3> tmp;
        tmp[0]=my_rank % nranks[0];
        tmp[1]=static_cast<int>(my_rank / nranks[0]) % nranks[1];
        tmp[2]=static_cast<int>(my_rank / (nranks[0]*nranks[1]));
        return tmp;
    }

    std::array<int, 3> calculateNlocal_noguards(const std::array<int,3>& nranks, const std::array<int,3>& npglobal) const
    {
        std::array<int, 3> tmp;
        for(int i=0; i < 3; i++)
        {
            tmp[i]=npglobal[i]/nranks[i];
            if(i>=DIM)
                tmp[i]=1;
        }
        return tmp;
    }

    std::array<int, 3> calculateNlocal_guards(const std::array<int,3>& nlocal_noguards, const std::array<int,3>& guards) const
    {
        std::array<int, 3> tmp;
        for(int i=0; i< 3; i++)
        {
            tmp[i]=nlocal_noguards[i] + 2*guards[i];
            if(i>=DIM)
                tmp[i]=1;
        }
        return tmp;
    }
    
    std::array<int,6> calculateIndexLimitsData(const std::array<int,3>& nlocal_noguards, const std::array<int,3>& guards) const
    {
        std::array<int,6> indexLimits;
        indexLimits[0]=guards[0];
        indexLimits[1]=nlocal_noguards[0] + guards[0];
        indexLimits[2]=guards[1];
        indexLimits[3]=nlocal_noguards[1] + guards[1];
        indexLimits[4]=guards[2];
        indexLimits[5]=nlocal_noguards[2] + guards[2];
        if(DIM==1)
        {
            indexLimits[2]=0;
            indexLimits[3]=1;
            indexLimits[4]=0;
            indexLimits[5]=1;
        }
        if(DIM==2)
        {
            indexLimits[4]=0;
            indexLimits[5]=1;
        }  
        return indexLimits; 
    }

    void calculateIndexLimitsSolver()
    {
        this->indexLimitsSolver_=this->indexLimitsData_;
        for(int i=0; i< DIM; i++)
        {
            if(bcsType_[2*i]==0 && this->hasBoundary_[2*i])
            {
                this->indexLimitsSolver_[2*i]+=1;
            }
            if(bcsType_[2*i+1]==0 && this->hasBoundary_[2*i+1])
            {
                this->indexLimitsSolver_[2*i+1]-=1;
            }
        }
    }

    int calculateNtotlocal(const std::array<int,3>& nlocal) const
    {
        int tmp=1;
        for(int i=0; i<DIM; i++)
        {
            tmp*=nlocal[i];
        }
        return tmp;
    }

    std::array<bool,6> setHasBoundary(const std::array<int,3>& nranks,const std::array<int,3> globalLocation) const 
    {
        std::array<bool,6> tmp={0,0,0,0,0,0};
        for(int i=0; i<DIM; i++)
        {
            if(globalLocation[i]==0 && nranks[i]==1)
            {
                tmp[2*i]=1;
                tmp[2*i+1]=1;
            }
            else if(globalLocation[i]==0 && nranks[i]>1)
            {
                tmp[2*i]=1;
            }
            else if(globalLocation[i]==nranks[i]-1 && nranks[i]>1)
            {
                tmp[2*i+1]=1;
            }
        }
        return tmp;
    }

    std::array<bool,6> setHasCommunication(const std::array<int,3>& nranks,const std::array<int,3>& globalLocation)
    {
        std::array<bool,6> tmp={0,0,0,0,0,0};
        this->indexLimitsComm_=this->indexLimitsData_;

        for(int dir=0; dir<DIM; dir++)
        {
            if(globalLocation[dir]==0 && nranks[dir]>1)
            {
                tmp[2*dir+1]=1;
                this->indexLimitsComm_[2*dir] = this->indexLimitsData_[2*dir+1] - this->guards_[dir];
            }
            else if(globalLocation[dir]==nranks[dir]-1 && nranks[dir]>1)
            {
                tmp[2*dir]=1;
                this->indexLimitsComm_[2*dir+1]=this->indexLimitsData_[2*dir] + this->guards_[dir];
            }
            else if(nranks[dir]>1)
            {
                tmp[2*dir]=1;
                tmp[2*dir+1]=1;
                this->indexLimitsComm_[2*dir] = this->indexLimitsData_[2*dir+1] - this->guards_[dir];
                this->indexLimitsComm_[2*dir+1]=this->indexLimitsData_[2*dir] + this->guards_[dir];
            }
        }
        numCommunication_=0;
        for(int i=0; i<2*DIM; i++)
        {
            numCommunication_+=tmp[i];
        }

        this->numElementsComm_={0,0,0};
        for(int dir =0; dir < DIM ; dir ++)
        {   
            this->numElementsComm_[dir]=1;
            for(int i =0; i<DIM; i++)
            {
                if(i!=dir)
                    this->numElementsComm_[dir]*=this->nlocal_noguards_[i];
            }
        }

        return tmp;
    }

    void computeEigenvalues()
    {
        std::array<T_data,6> eigenValues={0,0,0,0,0,0};
        eigenValuesLocal_={0,0};
        eigenValuesGlobal_={0,0};

        T_data eigMax=0;
        T_data eigMin=0;
        for(int i=0;i<DIM;i++)
        {
            eigenValues[2*i]=4*sin(1*PI/2/(indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i]+1))*sin(1*PI/2/(indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i]+1))/(ds_[i]*ds_[i]);
            eigenValues[2*i+1]=4*sin( (indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i])*PI/2/(indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i]+1))*
                                    sin((indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i])*PI/2/(indexLimitsSolver_[2*i+1]-indexLimitsSolver_[2*i]+1))/(ds_[i]*ds_[i]);
            eigMin+=eigenValues[2*i];
            eigMax+=eigenValues[2*i+1];
            //std::cout<< " Eigen Min " << eigenValues[2*i] << " dir " << i << " " << eigMin <<std::endl;
            //std::cout<< " Eigen Max " << eigenValues[2*i+1] << " dir " << i << " " << eigMax<<std::endl;
        }
        eigenValuesLocal_[0]=eigMin;
        eigenValuesLocal_[1]=eigMax;

        int ni;
        eigMax=0;
        eigMin=0;
        for(int i=0;i<DIM;i++)
        {
            ni=npglobal_[i];
            if(bcsType_[2*i]==0)
                ni-=1;
            if(bcsType_[2*i+1]==0)
                ni-=1;

            eigenValues[2*i]=4*sin(1*PI/2/(ni+1))*sin(1*PI/2/(ni+1))/(ds_[i]*ds_[i]);
            eigenValues[2*i+1]=4*sin( ni*PI/2/(ni+1))*sin(ni*PI/2/(ni+1))/(ds_[i]*ds_[i]);
            eigMin+=eigenValues[2*i];
            eigMax+=eigenValues[2*i+1];
        }
        eigenValuesGlobal_[0]=eigMin;
        eigenValuesGlobal_[1]=eigMax;
    }


    const std::array<int, 3> nranks_;
    const int my_rank_;
    const std::array<int,3> globalLocation_;
    const std::array<int,3> npglobal_;
    const std::array<T_data,3> origin_;
    const std::array<T_data,3> ds_;
    const std::array<int,3> guards_;
    const std::array<int,3> nlocal_noguards_;
    const std::array<int,3> nlocal_guards_;
    const int ntotlocal_guards_;
    const int ntotlocal_noguards_;
    const std::array<int,6> indexLimitsData_;
    const std::array<bool,6> hasCommunication_;
    const std::array<bool,6> hasBoundary_;
    const std::array<int,6> bcsType_; //0 Dirichlet, 1 Neumann, -1 No dim; order x-x+ y-y+ z-z+
    const std::array<T_data,6> bcsValue_;

    std::array<int,6> indexLimitsSolver_;
    std::array<int,6> indexLimitsComm_;
    int numCommunication_;
    std::array<int,3> numElementsComm_;
    std::array<T_data,2> eigenValuesLocal_;
    std::array<T_data,2> eigenValuesGlobal_;
    bool flagBCsSet_;
};