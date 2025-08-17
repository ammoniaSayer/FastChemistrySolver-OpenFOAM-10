void inline OptReaction::JF11
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];

    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;
    if(this->isIrreversible[i]==0)
    {
        const double Kp = ExpNegGbyRT[sr0]/ExpNegGbyRT[sl0];
        const double Kc = Kp > 1.4901171103413047e-8?Kp:1.4901171103413047e-8;
        const double sumVdBdT = (dBdT[sr0] ) - (dBdT[sl0] );
        const double dKcdTByKc = sumVdBdT;
        invKc = 1.0/Kc;        
        Kr = Kf*invKc;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));

        const double dCrdC0 = 1;
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    }
    const double CF = C[sl0];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();        
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);
        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;
            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);
            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            unsigned int j1 = this->nSpecies-2+1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;   
            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
        }      
    }
}   
void inline OptReaction::JF11
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    const double dCrdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0); 

    const double CF = C[sl0];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);
        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            unsigned int j1 = this->nSpecies-2+1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;   
            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
        }      
    }
}          
void inline OptReaction::JF12
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];

    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;
    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]);
        double Kc = Kp*this->Pow_pByRT_SumVki[3];  
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;
        invKc = 1.0/Kc;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1]) - (dBdT[sl0] );
        const double dKcdTByKc = sumVdBdT - 1*invT;
        Kr = Kf*invKc;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));

        const double dCrdC0 = C[sr1];
        const double dCrdC1 = C[sr0];     
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);    
    }


    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;   

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = 1; 
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];

        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
      
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

 
            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
     
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;   
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;  
        }      
    }
}             
void inline OptReaction::JF12
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];

    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    const double dCrdC0 = C[sr1];
    const double dCrdC1 = C[sr0];     
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);  

    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1];
    const double dCfdC0 = 1;

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];

        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
      
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

 
            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
     
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;   
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;  
        }      
    }
}             

void inline OptReaction::JF13
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]);
        double Kc = Kp*this->Pow_pByRT_SumVki[4];  
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;
        invKc = 1.0/Kc;
        Kr = Kf*invKc;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1] + dBdT[sr2]) - (dBdT[sl0] );
        const double dKcdTByKc = sumVdBdT - 2*invT;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));    

        const double dCrdC0 = C[sr1]*C[sr2];
        const double dCrdC1 = C[sr0]*C[sr2];
        const double dCrdC2 = C[sr0]*C[sr1];
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2);
            
    }
    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1]*C[sr2];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;


        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0]; 

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;   

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;   

        }      

    }
}            

void inline OptReaction::JF13
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];
    
    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    const double dCrdC0 = C[sr1]*C[sr2];
    const double dCrdC1 = C[sr0]*C[sr2];
    const double dCrdC2 = C[sr0]*C[sr1];
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2);   
    
    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1]*C[sr2];
 
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[Itbr[4]+k]);

        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0]; 

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;   

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;   

        }      

    }
}            
void inline OptReaction::JF21
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];


    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp*this->Pow_pByRT_SumVki[1];  
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;
        const double sumVdBdT = (dBdT[sr0] ) - (dBdT[sl0] + dBdT[sl1]);
        const double dKcdTByKc = sumVdBdT +1*invT;
        invKc = 1.0/Kc;  
        Kr = Kf*invKc;        
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));  
        
        const double dCrdC0 = 1;
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);     
    }

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    
    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;


        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;


        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);



        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];    
            
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;            
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;              
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;   
            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;          
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;   
            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;            
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;          
        }
    }    
}     

void inline OptReaction::JF21
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];

    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];
    const double dCrdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);    

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;


        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[Itbr[4]+k]);

        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);



        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];    
            
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;            
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;              
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;   
            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;          
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;   
            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;            
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;          
        }
    }    
}     

void inline OptReaction::JF22
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];


    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp;  
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1]) - (dBdT[sl0] + dBdT[sl1]);
        const double dKcdTByKc = sumVdBdT;    
        invKc = 1.0/Kc;    
        Kr = Kf*invKc;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));     

        const double dCrdC0 = C[sr1];
        const double dCrdC1 = C[sr0];
        ddNdtByVdcTp[sl0*(this->n_)+ sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+ sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+ sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl1*(this->n_)+ sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+ sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+ sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+ sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+ sr1] += (-Kr*dCrdC1);        
    }

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+ this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+ this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+ this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+ this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+ sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+ sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+ sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+ sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+ sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+ sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+ sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+ sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);


            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0]; 

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;            
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;      
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;  
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;  


            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;  
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;   

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;            
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;   
        }
    }   
}   


void inline OptReaction::JF22
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];


    double Kr = 0;
    double dKrdT = 0;
    double dCrdC0 = 1;
    double dCrdC1 = 1; 
    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    dCrdC0 = C[sr1];
    dCrdC1 = C[sr0]; 
    ddNdtByVdcTp[sl0*(this->n_)+ sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+ sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+ sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl1*(this->n_)+ sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+ sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+ sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+ sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+ sr1] += (-Kr*dCrdC1);        

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1];
    
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+ this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+ this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+ this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+ this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+ sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+ sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+ sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+ sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+ sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+ sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+ sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+ sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = this->dKfdC_[Itbr[4]+k];
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);
        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);


            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0]; 

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;            
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;      
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;  
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;  


            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;            
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;  
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;            
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;    
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;   

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;            
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;    
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;   
        }
    }   
} 

void inline OptReaction::JF23
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{


    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp*this->Pow_pByRT_SumVki[3];  
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;
        invKc = 1.0/Kc;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1] + dBdT[sr2]) - (dBdT[sl0] + dBdT[sl1] );
        const double dKcdTByKc = sumVdBdT - 1*invT;
        Kr = Kf*invKc;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));  
        
        const double dCrdC0 = C[sr1]*C[sr2];
        const double dCrdC1 = C[sr0]*C[sr2];
        const double dCrdC2 = C[sr0]*C[sr1];
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sl1*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2);   
    }

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1]*C[sr2];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q;
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr2*(this->n_)+sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0]; 


            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1]; 

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 

        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;  
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;  

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;  
        }      

    }  
}    

void inline OptReaction::JF23
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];
    const double dCrdC0 = C[sr1]*C[sr2];
    const double dCrdC1 = C[sr0]*C[sr2];
    const double dCrdC2 = C[sr0]*C[sr1];
    
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sl1*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2);  
    
    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1]*C[sr2];
    
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1];
    const double dCfdC1 = C[sl0];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr2*(this->n_)+sl1] += Kf*dCfdC1;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0]; 


            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1]; 

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 

        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;

            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;  
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;  

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;  
        }      

    }  
}


void inline OptReaction::JF31
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{

    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];


    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp*this->Pow_pByRT_SumVki[0];
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;  
        invKc = 1.0/Kc;
        Kr = Kf*invKc;
        const double sumVdBdT = (dBdT[sr0]) - (dBdT[sl0] + dBdT[sl1] + dBdT[sl2]);
        const double dKcdTByKc = sumVdBdT + 2*invT;
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));  
        
        const double dCrdC0 = 1;
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);   
    }

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        if(this->isIrreversible[i]==2)
        {dKrdC = (this->dKfdC_[Itbr[4]+k]);}
        else
        {dKrdC = (this->dKfdC_[k]*invKc);}
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
       
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;


            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
    
        }
        else if(remain ==3)
        {
            unsigned int j = this->nSpecies-3;
            double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;

            j = j+1;
            WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;   
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;          
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;   

            j = j+1;
            WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;  
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;          
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;    

        }      
    
    }  
}             

void inline OptReaction::JF31
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];

    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];
    const double dCrdC0 = 1;
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);  
    
    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
       
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;


            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
    
        }
        else if(remain ==3)
        {
            unsigned int j = this->nSpecies-3;
            double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;

            j = j+1;
            WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;   
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;          
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;   

            j = j+1;
            WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;  
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;          
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;    

        }      
    
    }  
}             

void inline OptReaction::JF32
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{


    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];


    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;
    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp*this->Pow_pByRT_SumVki[1];
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;  
        invKc = 1.0/Kc;
        Kr = Kf*invKc;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1] ) - (dBdT[sl0] + dBdT[sl1] + dBdT[sl2]);
        const double dKcdTByKc = sumVdBdT + 1*invT;
        dKrdT = (dKfdT/Kc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));  
        
        const double dCrdC0 = C[sr1];
        const double dCrdC1 = C[sr0];
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl2*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);  
    }
    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1];

    
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    
    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr1*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;

        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);
            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];

        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
       
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1  = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1  = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2  = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
        }      

    }  
}    

void inline OptReaction::JF32
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];

    double Kr = 0;
    double dKrdT = 0;
    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    const double dCrdC0 = C[sr1];
    const double dCrdC1 = C[sr0];
    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl2*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);  

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1];
    
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;  

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr1*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);
            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];

            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];

        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
       
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1  = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1  = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2  = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
        }      
    }  
}  

void inline OptReaction::JF33
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{

    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    double dKrdT = 0;
    double invKc = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp;
        Kc = Kc > 1.4901171103413047e-8?Kc:1.4901171103413047e-8;  
        invKc = 1.0/Kc;
        Kr = Kf*invKc;
        const double sumVdBdT = (dBdT[sr0] + dBdT[sr1] + dBdT[sr2]) - (dBdT[sl0] + dBdT[sl1] + dBdT[sl2]);
        const double dKcdTByKc = sumVdBdT;    
        dKrdT = (dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0));  
        
        const double dCrdC0 = C[sr1]*C[sr2];
        const double dCrdC1 = C[sr0]*C[sr2];
        const double dCrdC2 = C[sr0]*C[sr1];  
        ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
        ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
        ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
        ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sl2*(this->n_)+sr1] -= (-Kr*dCrdC1);
        ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
        ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sl1*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sl2*(this->n_)+sr2] -= (-Kr*dCrdC2);
        ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
        ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2); 
    }

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1]*C[sr2];
    
    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;
    
    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr2*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr1*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr2*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && 
        i <this->nReactions-this->n_PlogReaction)
    {
      double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[k]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);
            __m256d WdMdC = _mm256_mul_pd(dKrdC_,CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0];
            
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;  
        }      
    }  
}            

void inline OptReaction::JF33
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{

    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    double dKrdT = 0;

    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    const double dCrdC0 = C[sr1]*C[sr2];
    const double dCrdC1 = C[sr0]*C[sr2];
    const double dCrdC2 = C[sr0]*C[sr1];  

    ddNdtByVdcTp[sl0*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl1*(this->n_)+sr0] -= (-Kr*dCrdC0);
    ddNdtByVdcTp[sl2*(this->n_)+sr0] -= (-Kr*dCrdC0);    
    ddNdtByVdcTp[sr0*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr1*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sr2*(this->n_)+sr0] += (-Kr*dCrdC0);
    ddNdtByVdcTp[sl0*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl1*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sl2*(this->n_)+sr1] -= (-Kr*dCrdC1);
    ddNdtByVdcTp[sr0*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr1*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sr2*(this->n_)+sr1] += (-Kr*dCrdC1);
    ddNdtByVdcTp[sl0*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sl1*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sl2*(this->n_)+sr2] -= (-Kr*dCrdC2);
    ddNdtByVdcTp[sr0*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr1*(this->n_)+sr2] += (-Kr*dCrdC2);
    ddNdtByVdcTp[sr2*(this->n_)+sr2] += (-Kr*dCrdC2); 

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1]*C[sr2];

    const double q = (Kf*CF) - (Kr*CR);
    dNdtByV[sl0] = dNdtByV[sl0] - q; 
    dNdtByV[sl1] = dNdtByV[sl1] - q;
    dNdtByV[sl2] = dNdtByV[sl2] - q;
    dNdtByV[sr0] = dNdtByV[sr0] + q;
    dNdtByV[sr1] = dNdtByV[sr1] + q;
    dNdtByV[sr2] = dNdtByV[sr2] + q;

    const double dqdT = (dKfdT*CF)-(dKrdT*CR);
    ddNdtByVdcTp[sl0*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl1*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sl2*(this->n_)+this->nSpecies] -= dqdT;
    ddNdtByVdcTp[sr0*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr1*(this->n_)+this->nSpecies] += dqdT;
    ddNdtByVdcTp[sr2*(this->n_)+this->nSpecies] += dqdT;

    const double dCfdC0 = C[sl1]*C[sl2];
    const double dCfdC1 = C[sl0]*C[sl2];
    const double dCfdC2 = C[sl0]*C[sl1];
    ddNdtByVdcTp[sl0*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl1*(this->n_)+sl0] -= Kf*dCfdC0;
    ddNdtByVdcTp[sl2*(this->n_)+sl0] -= Kf*dCfdC0;    
    ddNdtByVdcTp[sr0*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr1*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sr2*(this->n_)+sl0] += Kf*dCfdC0;
    ddNdtByVdcTp[sl0*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl1*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sl2*(this->n_)+sl1] -= Kf*dCfdC1;
    ddNdtByVdcTp[sr0*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr1*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sr2*(this->n_)+sl1] += Kf*dCfdC1;
    ddNdtByVdcTp[sl0*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl1*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sl2*(this->n_)+sl2] -= Kf*dCfdC2;
    ddNdtByVdcTp[sr0*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr1*(this->n_)+sl2] += Kf*dCfdC2;
    ddNdtByVdcTp[sr2*(this->n_)+sl2] += Kf*dCfdC2;

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && 
        i <this->nReactions-this->n_PlogReaction)
    {
      double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);
            __m256d WdMdC = _mm256_mul_pd(dKrdC_,CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            ddNdtByVdcTp[sl0*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= ArrRes[0][0];
            ddNdtByVdcTp[sr0*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr1*(this->n_)+j0] += ArrRes[0][0];
            ddNdtByVdcTp[sr2*(this->n_)+j0] += ArrRes[0][0];
            
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= ArrRes[0][1];
            ddNdtByVdcTp[sr0*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr1*(this->n_)+j1] += ArrRes[0][1];
            ddNdtByVdcTp[sr2*(this->n_)+j1] += ArrRes[0][1];

            ddNdtByVdcTp[sl0*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= ArrRes[0][2];
            ddNdtByVdcTp[sr0*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr1*(this->n_)+j2] += ArrRes[0][2];
            ddNdtByVdcTp[sr2*(this->n_)+j2] += ArrRes[0][2];

            ddNdtByVdcTp[sl0*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl1*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sl2*(this->n_)+j3] -= ArrRes[0][3];
            ddNdtByVdcTp[sr0*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr1*(this->n_)+j3] += ArrRes[0][3];
            ddNdtByVdcTp[sr2*(this->n_)+j3] += ArrRes[0][3];
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl1*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sl2*(this->n_)+j] -= WdMdC;
            ddNdtByVdcTp[sr0*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr1*(this->n_)+j] += WdMdC;
            ddNdtByVdcTp[sr2*(this->n_)+j] += WdMdC;        
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-1;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl1*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sl2*(this->n_)+j0] -= WdMdC0;
            ddNdtByVdcTp[sr0*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr1*(this->n_)+j0] += WdMdC0;
            ddNdtByVdcTp[sr2*(this->n_)+j0] += WdMdC0;  

            unsigned int j1 = this->nSpecies-2;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl1*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sl2*(this->n_)+j1] -= WdMdC1;
            ddNdtByVdcTp[sr0*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr1*(this->n_)+j1] += WdMdC1;
            ddNdtByVdcTp[sr2*(this->n_)+j1] += WdMdC1;      

            unsigned int j2 = this->nSpecies-1;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;
            ddNdtByVdcTp[sl0*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl1*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sl2*(this->n_)+j2] -= WdMdC2;
            ddNdtByVdcTp[sr0*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr1*(this->n_)+j2] += WdMdC2;
            ddNdtByVdcTp[sr2*(this->n_)+j2] += WdMdC2;  
        }      
    }  
}            

void inline OptReaction::JFG
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ ExpNegGbyRT,
    const double* __restrict__ dBdT
)const noexcept
{

    double Kr = 0;
    double dKrdT = 0;
    double CF = 1.0;
    double CR = 1.0;
    double invKc = 1;
    if(this->isIrreversible[i]==0)
    {
        int sumVki = 0;
        double Kp = 1.0;
        double sumVdBdT = 0.0;
        for(unsigned int j = 0; j < this->lhsIndex[i].size();j++)
        {
            const unsigned int si = this->lhsIndex[i][j];
            Kp = Kp / this->tmp_Exp[si];
            sumVki = sumVki - 1;
            sumVdBdT = sumVdBdT - dBdT[si];
        }
    
        for(unsigned int j = 0; j < this->rhsIndex[i].size();j++)
        {
            const unsigned int si = this->rhsIndex[i][j];
            Kp = Kp * this->tmp_Exp[si];
            sumVki = sumVki + 1;
            sumVdBdT = sumVdBdT + dBdT[si];
            CR = CR * C[si];
        }
        this->update_Pow_pByRT_SumVki(T);
        const double Kc = std::max(Kp*this->Pow_pByRT_SumVki_[sumVki],1.4901171103413047e-8);
        invKc = 1.0/Kc;            

        const double dKcdTByKc = sumVdBdT - sumVki*invT;
        Kr = Kf*invKc;
        dKrdT = dKfdT*invKc - (Kc > 1.4901171103413047e-8 ? Kr*dKcdTByKc : 0);
    }

    for(unsigned int j = 0; j < lhsIndex[i].size();j++)
    {
        const unsigned int index = lhsIndex[i][j];
        CF = CF * C[index];
    }

    const double q = Kf*CF - Kr*CR;
    const double dqdT = (dKfdT*CF-dKrdT*CR);

    for(unsigned int j = 0; j < this->lhsIndex[i].size();j++)
    {
        const unsigned int si = this->lhsIndex[i][j];
        ddNdtByVdcTp[si*(n_)+this->nSpecies] -= dqdT;
        dNdtByV[si] = dNdtByV[si] - q;                        
    }

    for(unsigned int j = 0; j < this->rhsIndex[i].size();j++)
    {
        const unsigned int si = this->rhsIndex[i][j];
        ddNdtByVdcTp[si*(n_)+this->nSpecies] += dqdT;
        dNdtByV[si] = dNdtByV[si] + q;                        
    }            
                

    for(unsigned int j = 0; j < lhsIndex[i].size();j++)
    {
        const unsigned int sj = lhsIndex[i][j];

        double dCfdCj = 1.0;
        for(unsigned int k = 0; k < lhsIndex[i].size();k++)
        {
            const unsigned int sk = lhsIndex[i][k];

            dCfdCj = dCfdCj*
            (
                (j==k)? 1:C[sk]
            );
        }

        const double KfdCfdCj = Kf*dCfdCj;

        for(unsigned int k = 0; k < lhsIndex[i].size();k++)
        {
            const unsigned int sl = lhsstoichCoeff[i][k];
            const unsigned int sk = lhsIndex[i][k];
            ddNdtByVdcTp[sk*(n_)+sj] -= sl*KfdCfdCj;
        }

        for(unsigned int k = 0; k < rhsIndex[i].size();k++)
        {
            const unsigned int sr = rhsstoichCoeff[i][k];
            const unsigned int sk = rhsIndex[i][k];
            ddNdtByVdcTp[sk*(n_)+sj] += sr*KfdCfdCj;
        }

    }  
    if(this->isIrreversible[i]==0)
    {
        for(unsigned int j = 0; j < rhsIndex[i].size();j++)
        {
            const unsigned int sj = rhsIndex[i][j];                        
    
            double dCrdCj = 1.0;
            for(unsigned int k = 0; k < rhsIndex[i].size();k++)
            {
                const unsigned int sk = rhsIndex[i][k];
    
                dCrdCj = dCrdCj*
                (
                    (j==k)? 1:C[sk]
                );
            }
    
            const double negKrdCrdCj = -Kr*dCrdCj;

            for(unsigned int k = 0; k < lhsIndex[i].size();k++)
            {
                const unsigned int sl = lhsstoichCoeff[i][k];
                const unsigned int sk = lhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+sj] -= sl*negKrdCrdCj;
            }

            for(unsigned int k = 0; k < rhsIndex[i].size();k++)
            {
                const unsigned int sr = rhsstoichCoeff[i][k];
                const unsigned int sk = rhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+sj] += sr*negKrdCrdCj;                        
            }
        }          
    }




    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        double* const* __restrict__ ArrRes = this->ArrPtr.data();
        const unsigned int kk = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[kk]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[kk]*invKc);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[kk*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            for(unsigned int k = 0; k < lhsIndex[i].size();k++)
            {
                const unsigned int sl = lhsstoichCoeff[i][k];
                const unsigned int sk = lhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*ArrRes[0][0];
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*ArrRes[0][1];
                ddNdtByVdcTp[sk*(n_)+j2] -= sl*ArrRes[0][2];
                ddNdtByVdcTp[sk*(n_)+j3] -= sl*ArrRes[0][3];
            }

            for(unsigned int k = 0; k < rhsIndex[i].size();k++)
            {
                const unsigned int sr = rhsstoichCoeff[i][k];
                const unsigned int sk = rhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*ArrRes[0][0];
                ddNdtByVdcTp[sk*(n_)+j1] += sr*ArrRes[0][1];    
                ddNdtByVdcTp[sk*(n_)+j2] += sr*ArrRes[0][2];
                ddNdtByVdcTp[sk*(n_)+j3] += sr*ArrRes[0][3];                                       
            }
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[kk]*this->dMdC(kk,j)*CF - dKrdC*this->dMdC(kk,j)*CR;
            for(unsigned int k = 0; k < lhsIndex[i].size();k++)
            {
                const unsigned int sl = lhsstoichCoeff[i][k];
                const unsigned int sk = lhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j] -= sl*WdMdC;
            }
    
            for(unsigned int k = 0; k < rhsIndex[i].size();k++)
            {
                const unsigned int sr = rhsstoichCoeff[i][k];
                const unsigned int sk = rhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j] += sr*WdMdC;                                     
            }                
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            unsigned int j1 = this->nSpecies-1;

            double WdMdC0 = this->dKfdC_[kk]*this->dMdC(kk,j0)*CF - dKrdC*this->dMdC(kk,j0)*CR;
            double WdMdC1 = this->dKfdC_[kk]*this->dMdC(kk,j1)*CF - dKrdC*this->dMdC(kk,j1)*CR;

            for(unsigned int k = 0; k < lhsIndex[i].size();k++)
            {
                const unsigned int sl = lhsstoichCoeff[i][k];
                const unsigned int sk = lhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*WdMdC0;
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*WdMdC1;                
            }
    
            for(unsigned int k = 0; k < rhsIndex[i].size();k++)
            {
                const unsigned int sr = rhsstoichCoeff[i][k];
                const unsigned int sk = rhsIndex[i][k];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*WdMdC0;  
                ddNdtByVdcTp[sk*(n_)+j1] += sr*WdMdC1;                                   
            }         
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            unsigned int j1 = this->nSpecies-2;
            unsigned int j2 = this->nSpecies-1;
            double WdMdC0 = this->dKfdC_[kk]*this->dMdC(kk,j0)*CF - dKrdC*this->dMdC(kk,j0)*CR;
            double WdMdC1 = this->dKfdC_[kk]*this->dMdC(kk,j1)*CF - dKrdC*this->dMdC(kk,j1)*CR;
            double WdMdC2 = this->dKfdC_[kk]*this->dMdC(kk,j2)*CF - dKrdC*this->dMdC(kk,j2)*CR;

            for(unsigned int l = 0; l < lhsIndex[i].size();l++)
            {
                const unsigned int sl = lhsstoichCoeff[i][l];
                const unsigned int sk = lhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*WdMdC0;
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*WdMdC1; 
                ddNdtByVdcTp[sk*(n_)+j2] -= sl*WdMdC2;               
            }

            for(unsigned int l = 0; l < rhsIndex[i].size();l++)
            {
                const unsigned int sr = rhsstoichCoeff[i][l];
                const unsigned int sk = rhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*WdMdC0;  
                ddNdtByVdcTp[sk*(n_)+j1] += sr*WdMdC1; 
                ddNdtByVdcTp[sk*(n_)+j2] += sr*WdMdC2;                                  
            }                     
        }
    }   
}


void inline OptReaction::JFG
(
    unsigned int i,
    double Kf,
    double dKfdT,
    const double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ddNdtByVdcTp
)const noexcept
{
    double* const* __restrict__ ArrRes = this->ArrPtr.data();
    double Kr = 0;
    double dKrdT = 0;
    double CF = 1.0;
    double CR = 1.0;

 
    Kr = this->Kf_[i - Ikf[1] + Ikf[9]];
    dKrdT = this->dKfdT_[i - Ikf[1] + Ikf[9]];

    for(unsigned int j = 0; j < lhsIndex[i].size();j++)
    {
        const unsigned int index = lhsIndex[i][j];
        CF = CF * C[index];
    }

    const double q = Kf*CF - Kr*CR;
    const double dqdT = (dKfdT*CF-dKrdT*CR);

    for(unsigned int j = 0; j < this->lhsIndex[i].size();j++)
    {
        const unsigned int si = this->lhsIndex[i][j];
        ddNdtByVdcTp[si*(n_)+this->nSpecies] -= dqdT;
        dNdtByV[si] = dNdtByV[si] - q;                        
    }

    for(unsigned int j = 0; j < this->rhsIndex[i].size();j++)
    {
        const unsigned int si = this->rhsIndex[i][j];
        ddNdtByVdcTp[si*(n_)+this->nSpecies] += dqdT;
        dNdtByV[si] = dNdtByV[si] + q;                        
    }            
                

    for(unsigned int j = 0; j < lhsIndex[i].size();j++)
    {
        const unsigned int sj = lhsIndex[i][j];

        double dCfdCj = 1.0;
        for(unsigned int k = 0; k < lhsIndex[i].size();k++)
        {
            const unsigned int sk = lhsIndex[i][k];

            dCfdCj = dCfdCj*
            (
                (j==k)? 1:C[sk]
            );
        }

        const double KfdCfdCj = Kf*dCfdCj;

        for(unsigned int k = 0; k < lhsIndex[i].size();k++)
        {
            const unsigned int sl = lhsstoichCoeff[i][k];
            const unsigned int sk = lhsIndex[i][k];
            ddNdtByVdcTp[sk*(n_)+sj] -= sl*KfdCfdCj;
        }

        for(unsigned int k = 0; k < rhsIndex[i].size();k++)
        {
            const unsigned int sr = rhsstoichCoeff[i][k];
            const unsigned int sk = rhsIndex[i][k];
            ddNdtByVdcTp[sk*(n_)+sj] += sr*KfdCfdCj;
        }

    }  

    if(i>=this->n_Arrhenius + this->n_NonEquilibriumReversibleArrhenius && i <this->nReactions-this->n_PlogReaction)
    {
        const unsigned int k = i - this->n_Arrhenius - this->n_NonEquilibriumReversibleArrhenius;
        unsigned int remain = this->nSpecies%4;

        __m256d dKfdC = _mm256_set1_pd(this->dKfdC_[k]);
        __m256d CF_ = _mm256_set1_pd(CF);            
        __m256d CR_ = _mm256_set1_pd(CR);  
        double dKrdC = 0;
        dKrdC = (this->dKfdC_[Itbr[4]+k]);
        __m256d dKrdC_ = _mm256_set1_pd(dKrdC);

        for(unsigned int j = 0; j < this->nSpecies-remain;j=j+4)
        {
            unsigned int j0 = j+0;
            unsigned int j1 = j+1;
            unsigned int j2 = j+2;
            unsigned int j3 = j+3;

            __m256d dMdC = _mm256_loadu_pd(&this->ThirdBodyFactor1D[k*nSpecies+j0]);

            __m256d WdMdC = _mm256_mul_pd(_mm256_mul_pd(dKrdC_,dMdC),CR_);
            WdMdC = _mm256_fmsub_pd(_mm256_mul_pd(dKfdC,dMdC),CF_,WdMdC);
            _mm256_storeu_pd(&ArrRes[0][0],WdMdC);

            for(unsigned int l = 0; l < lhsIndex[i].size();l++)
            {
                const unsigned int sl = lhsstoichCoeff[i][l];
                const unsigned int sk = lhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*ArrRes[0][0];
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*ArrRes[0][1];
                ddNdtByVdcTp[sk*(n_)+j2] -= sl*ArrRes[0][2];
                ddNdtByVdcTp[sk*(n_)+j3] -= sl*ArrRes[0][3];
            }

            for(unsigned int l = 0; l < rhsIndex[i].size();l++)
            {
                const unsigned int sr = rhsstoichCoeff[i][l];
                const unsigned int sk = rhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*ArrRes[0][0];
                ddNdtByVdcTp[sk*(n_)+j1] += sr*ArrRes[0][1];    
                ddNdtByVdcTp[sk*(n_)+j2] += sr*ArrRes[0][2];
                ddNdtByVdcTp[sk*(n_)+j3] += sr*ArrRes[0][3];                                       
            }
        } 
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double WdMdC = this->dKfdC_[k]*this->dMdC(k,j)*CF - dKrdC*this->dMdC(k,j)*CR;

            for(unsigned int l = 0; l < lhsIndex[i].size();l++)
            {
                const unsigned int sl = lhsstoichCoeff[i][l];
                const unsigned int sk = lhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j] -= sl*WdMdC;
            }

            for(unsigned int l = 0; l < rhsIndex[i].size();l++)
            {
                const unsigned int sr = rhsstoichCoeff[i][l];
                const unsigned int sk = rhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j] += sr*WdMdC;                                     
            }                
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            unsigned int j1 = this->nSpecies-1;

            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;

            for(unsigned int l = 0; l < lhsIndex[i].size();l++)
            {
                const unsigned int sl = lhsstoichCoeff[i][l];
                const unsigned int sk = lhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*WdMdC0;
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*WdMdC1;                
            }

            for(unsigned int l = 0; l < rhsIndex[i].size();l++)
            {
                const unsigned int sr = rhsstoichCoeff[i][l];
                const unsigned int sk = rhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*WdMdC0;  
                ddNdtByVdcTp[sk*(n_)+j1] += sr*WdMdC1;                                   
            }         
        }
        else if(remain ==3)
        {
            unsigned int j0 = this->nSpecies-3;
            unsigned int j1 = this->nSpecies-2;
            unsigned int j2 = this->nSpecies-1;
            double WdMdC0 = this->dKfdC_[k]*this->dMdC(k,j0)*CF - dKrdC*this->dMdC(k,j0)*CR;
            double WdMdC1 = this->dKfdC_[k]*this->dMdC(k,j1)*CF - dKrdC*this->dMdC(k,j1)*CR;
            double WdMdC2 = this->dKfdC_[k]*this->dMdC(k,j2)*CF - dKrdC*this->dMdC(k,j2)*CR;

            for(unsigned int l = 0; l < lhsIndex[i].size();l++)
            {
                const unsigned int sl = lhsstoichCoeff[i][l];
                const unsigned int sk = lhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] -= sl*WdMdC0;
                ddNdtByVdcTp[sk*(n_)+j1] -= sl*WdMdC1; 
                ddNdtByVdcTp[sk*(n_)+j2] -= sl*WdMdC2;               
            }

            for(unsigned int l = 0; l < rhsIndex[i].size();l++)
            {
                const unsigned int sr = rhsstoichCoeff[i][l];
                const unsigned int sk = rhsIndex[i][l];
                ddNdtByVdcTp[sk*(n_)+j0] += sr*WdMdC0;  
                ddNdtByVdcTp[sk*(n_)+j1] += sr*WdMdC1; 
                ddNdtByVdcTp[sk*(n_)+j2] += sr*WdMdC2;                                  
            }                     
        }
    }   
}
