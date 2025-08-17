#include "OptReaction.H"
#include <immintrin.h>  


void 
OptReaction::ddNdtByVdcTp
(
    double p,
    double Temperature,
    double* __restrict__ YTp,
    double* __restrict__ C,
    double* __restrict__ dNdtByV,
    double* __restrict__ ExpNegGbyRT,
    double* __restrict__ dBdT,
    double* __restrict__ dCpdT,
    double* __restrict__ Cp,
    double* __restrict__ Ha,
    double* __restrict__ rhoMvj,
    double* __restrict__ WiByrhoM,
    double* __restrict__ ddNdtByVdcTp,
    bool massFractionBased  
) const noexcept
{
    Temperature = Temperature<TlowMin?TlowMin:Temperature;
    Temperature = Temperature>ThighMax?ThighMax:Temperature;
    this->logT = std::log(Temperature);
    this->T = Temperature;
    this->invT = 1/Temperature;
    this->sqrT = Temperature*Temperature;
    this->setPtrCoeffs(Temperature);

    {
        this->JacobianThermo
        (
            p,
            Temperature,
            YTp,
            C,
            dNdtByV,
            &this->tmp_Exp[0],
            dBdT,
            dCpdT,
            Cp,
            Ha,
            rhoMvj,
            WiByrhoM,
            ddNdtByVdcTp
        );
    }

    this->update_Pow_pByRT_SumVki2(Temperature);

    for(size_t i = 0; i <this->Troe.size();i++)
    {
        size_t j0 = i + this->nSpecies;
        size_t j1 = i + this->nSpecies + this->Troe.size();
        size_t j2 = i + this->nSpecies + this->Troe.size()*2;         
        this->tmp_Exp[j0] = -Temperature*this->invTsss_[i];            
        this->tmp_Exp[j1] = -this->Tss_[i]*invT; 
        this->tmp_Exp[j2] = -Temperature*this->invTs_[i];            
    }
    
    for(size_t i = 0; i <this->SRI.size();i++)
    {
        size_t j0 = i + this->nSpecies + this->Troe.size()*3;
        size_t j1 = i + this->nSpecies + this->Troe.size()*3 + this->SRI.size();
        this->tmp_Exp[j0] = -this->b_[i]*invT;
        this->tmp_Exp[j1] = -Temperature*this->invc_[i];            
    }   



    {
        size_t remain = this->tmp_Exp.size()%4;
        for(size_t i = 0; i < this->tmp_Exp.size()-remain;i=i+4)
        {
            __m256d tmp = _mm256_loadu_pd(&this->tmp_Exp[i]);
            tmp = vec256_expd(tmp);
            _mm256_storeu_pd(&this->tmp_Exp[i],tmp);
        }
        if(remain==1)
        {
            size_t i = this->tmp_Exp.size()-1;
            this->tmp_Exp[i] = std::exp(this->tmp_Exp[i]);
        }
        else if(remain==2)
        {
            size_t i0 = this->tmp_Exp.size()-2;
            size_t i1 = this->tmp_Exp.size()-1;
            __m256d tmp = _mm256_setr_pd(tmp_Exp[i0],tmp_Exp[i1],0,0);
            tmp = vec256_expd(tmp);
            this->tmp_Exp[i0] = get_elem0(tmp);
            this->tmp_Exp[i1] = get_elem1(tmp);
        }
        else if(remain==3)
        {
            size_t i0 = this->tmp_Exp.size()-3;
            size_t i1 = this->tmp_Exp.size()-2;
            size_t i2 = this->tmp_Exp.size()-1;
            __m256d tmp = _mm256_setr_pd(tmp_Exp[i0],tmp_Exp[i1],tmp_Exp[i2],0);
            tmp = vec256_expd(tmp);
            this->tmp_Exp[i0] = get_elem0(tmp);
            this->tmp_Exp[i1] = get_elem1(tmp);
            this->tmp_Exp[i2] = get_elem2(tmp);
        }
    }

    unsigned int remain = (this->Ikf[11]-n_Temperature_Independent_Reaction)%4;
    unsigned int times = (this->Ikf[11]-n_Temperature_Independent_Reaction)/4;
    
    for(unsigned int z = 0; z <times;z=z+1)
    {
        unsigned int i = z*4 + this->n_Temperature_Independent_Reaction;
        __m256d betav = _mm256_loadu_pd(&this->beta[i]);
        __m256d Tav = _mm256_loadu_pd(&this->Ta[i]);
        __m256d LogTv = _mm256_set1_pd(logT);
        __m256d InvTv = _mm256_set1_pd(invT);
        __m256d Kfv = _mm256_mul_pd(Tav,-InvTv);
        Kfv = _mm256_fmadd_pd(betav,LogTv,Kfv);
        __m256d A_ = _mm256_loadu_pd(&this->A[i]);
        Kfv = vec256_expd(Kfv);
        Kfv = _mm256_mul_pd(A_,Kfv);
        _mm256_storeu_pd(&this->Kf_[i],Kfv);
        __m256d dKfdT = _mm256_mul_pd(_mm256_fmadd_pd(Tav,InvTv,betav),InvTv);
        dKfdT = _mm256_mul_pd(dKfdT,Kfv); 
        _mm256_storeu_pd(&this->dKfdT_[i+0],dKfdT);           
    }
    if(remain==1)
    {
        unsigned int i = this->Ikf[11]-1;
        this->Kf_[i] = this->A[i]*std::exp(this->beta[i+0]*logT-this->Ta[i+0]*invT);   
        this->dKfdT_[i+0] = this->Kf_[i+0]*(this->beta[i+0]+this->Ta[i+0]*invT)*invT;  
    }
    else if(remain==2)
    {
        unsigned int i0 = this->Ikf[11]-2;
        unsigned int i1 = this->Ikf[11]-1;    
        __m256d logTv = _mm256_set1_pd(logT);
        __m256d invTv = _mm256_set1_pd(invT);
        __m256d betav = _mm256_setr_pd(beta[i0],beta[i1],0,0);
        __m256d Av = _mm256_setr_pd(A[i0],A[i1],0,0);
        __m256d Tav = _mm256_setr_pd(Ta[i0],Ta[i1],0,0);
        __m256d tmp = _mm256_fmsub_pd(betav,logTv,_mm256_mul_pd(Tav,invTv));
        tmp = vec256_expd(tmp);
        __m256d Kfv = _mm256_mul_pd(Av,tmp);
        this->Kf_[i0] = get_elem0(Kfv);
        this->Kf_[i1] = get_elem1(Kfv);
        tmp = _mm256_fmadd_pd(Tav,invTv,betav);
        __m256d dKfdTv = _mm256_mul_pd(Kfv,_mm256_mul_pd(tmp,invTv));
        this->dKfdT_[i0] = get_elem0(dKfdTv);
        this->dKfdT_[i1] = get_elem1(dKfdTv);
    }
    else if(remain==3)
    {
        unsigned int i0 = this->Ikf[11]-3;
        unsigned int i1 = this->Ikf[11]-2;
        unsigned int i2 = this->Ikf[11]-1;    
        __m256d logTv = _mm256_set1_pd(logT);
        __m256d invTv = _mm256_set1_pd(invT);
        __m256d betav = _mm256_setr_pd(beta[i0],beta[i1],beta[i2],0);
        __m256d Av = _mm256_setr_pd(A[i0],A[i1],A[i2],0);
        __m256d Tav = _mm256_setr_pd(Ta[i0],Ta[i1],Ta[i2],0);
        __m256d tmp = _mm256_fmsub_pd(betav,logTv,_mm256_mul_pd(Tav,invTv));
        tmp = vec256_expd(tmp);
        __m256d Kfv = _mm256_mul_pd(Av,tmp);
        this->Kf_[i0] = get_elem0(Kfv);
        this->Kf_[i1] = get_elem1(Kfv);
        this->Kf_[i2] = get_elem2(Kfv);
        tmp = _mm256_fmadd_pd(Tav,invTv,betav);
        __m256d dKfdTv = _mm256_mul_pd(Kfv,_mm256_mul_pd(tmp,invTv));
        this->dKfdT_[i0] = get_elem0(dKfdTv);  
        this->dKfdT_[i1] = get_elem1(dKfdTv); 
        this->dKfdT_[i2] = get_elem2(dKfdTv); 
    }



    {
        unsigned int Tremain = (Itbr[5])%4;
        unsigned int remain_Tbr = this->nSpecies%4;
        for(unsigned int i = 0; i < Itbr[5]-Tremain; i=i+4)
        {
            double M0 = 0;
            double M1 = 0;           
            double M2 = 0; 
            double M3 = 0;  

            __m256d arrM_0 = _mm256_setzero_pd();
            __m256d arrM_1 = _mm256_setzero_pd();
            __m256d arrM_2 = _mm256_setzero_pd();
            __m256d arrM_3 = _mm256_setzero_pd();
            double* __restrict__ TBF1DRowi0 = &ThirdBodyFactor1D[(i+0)*nSpecies];
            double* __restrict__ TBF1DRowi1 = &ThirdBodyFactor1D[(i+1)*nSpecies];
            double* __restrict__ TBF1DRowi2 = &ThirdBodyFactor1D[(i+2)*nSpecies];
            double* __restrict__ TBF1DRowi3 = &ThirdBodyFactor1D[(i+3)*nSpecies];
            for(unsigned int j  = 0;j<this->nSpecies-remain_Tbr;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d Factor1 = _mm256_loadu_pd(&TBF1DRowi1[j+0]);
                __m256d Factor2 = _mm256_loadu_pd(&TBF1DRowi2[j+0]);
                __m256d Factor3 = _mm256_loadu_pd(&TBF1DRowi3[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
                arrM_1 = _mm256_fmadd_pd(Factor1,C_,arrM_1);
                arrM_2 = _mm256_fmadd_pd(Factor2,C_,arrM_2);
                arrM_3 = _mm256_fmadd_pd(Factor3,C_,arrM_3);
            }

            M0 = M0 + hsum4(arrM_0);
            M1 = M1 + hsum4(arrM_1);
            M2 = M2 + hsum4(arrM_2);
            M3 = M3 + hsum4(arrM_3);

            if(remain_Tbr==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M3 = M3 + TBF1DRowi3[j+0]*C[j+0];
            }
            else if(remain_Tbr==2)
            {

                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M3 = M3 + TBF1DRowi3[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M2 = M2 + TBF1DRowi2[j+1]*C[j+1];
                M3 = M3 + TBF1DRowi3[j+1]*C[j+1];
            }
            else if(remain_Tbr==3)
            {            
                unsigned int j = this->nSpecies-3;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M3 = M3 + TBF1DRowi3[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M2 = M2 + TBF1DRowi2[j+1]*C[j+1];
                M3 = M3 + TBF1DRowi3[j+1]*C[j+1];   
                M0 = M0 + TBF1DRowi0[j+2]*C[j+2];
                M1 = M1 + TBF1DRowi1[j+2]*C[j+2];
                M2 = M2 + TBF1DRowi2[j+2]*C[j+2];
                M3 = M3 + TBF1DRowi3[j+2]*C[j+2];   
            }
            this->tmp_M[i+0] = M0;
            this->tmp_M[i+1] = M1;
            this->tmp_M[i+2] = M2;
            this->tmp_M[i+3] = M3;
        }
        if(Tremain==3)
        {
            unsigned int i =(Itbr[5]) -3;
            double* __restrict__ TBF1DRowi0 = &ThirdBodyFactor1D[(i+0)*nSpecies];
            double* __restrict__ TBF1DRowi1 = &ThirdBodyFactor1D[(i+1)*nSpecies];
            double* __restrict__ TBF1DRowi2 = &ThirdBodyFactor1D[(i+2)*nSpecies];
            double M0 = 0;
            double M1 = 0;           
            double M2 = 0; 
            __m256d arrM_0 = _mm256_setzero_pd();
            __m256d arrM_1 = _mm256_setzero_pd();
            __m256d arrM_2 = _mm256_setzero_pd();
            for(unsigned int j  = 0;j<this->nSpecies-remain_Tbr;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d Factor1 = _mm256_loadu_pd(&TBF1DRowi1[j+0]);
                __m256d Factor2 = _mm256_loadu_pd(&TBF1DRowi2[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
                arrM_1 = _mm256_fmadd_pd(Factor1,C_,arrM_1);
                arrM_2 = _mm256_fmadd_pd(Factor2,C_,arrM_2);
            }

            M0 = M0 + hsum4(arrM_0);
            M1 = M1 + hsum4(arrM_1);
            M2 = M2 + hsum4(arrM_2);

            if(remain_Tbr==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
            }
            else if(remain_Tbr==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M2 = M2 + TBF1DRowi2[j+1]*C[j+1];
            }
            else if(remain_Tbr==3)
            {
                unsigned int j = this->nSpecies-3;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M2 = M2 + TBF1DRowi2[j+1]*C[j+1];
                M0 = M0 + TBF1DRowi0[j+2]*C[j+2];
                M1 = M1 + TBF1DRowi1[j+2]*C[j+2];
                M2 = M2 + TBF1DRowi2[j+2]*C[j+2];
            }
            this->tmp_M[i+0] = M0;
            this->tmp_M[i+1] = M1;
            this->tmp_M[i+2] = M2;
        }
        else if(Tremain==2)
        {
            unsigned int i =(Itbr[5]) -2;
            double* __restrict__ TBF1DRowi0 = &ThirdBodyFactor1D[(i+0)*nSpecies];
            double* __restrict__ TBF1DRowi1 = &ThirdBodyFactor1D[(i+1)*nSpecies];
            double M0 = 0;
            double M1 = 0;           
            __m256d arrM_0 = _mm256_setzero_pd();
            __m256d arrM_1 = _mm256_setzero_pd();
            for(unsigned int j  = 0;j<this->nSpecies-remain_Tbr;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d Factor1 = _mm256_loadu_pd(&TBF1DRowi1[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
                arrM_1 = _mm256_fmadd_pd(Factor1,C_,arrM_1);
            }

            M0 = M0 + hsum4(arrM_0);
            M1 = M1 + hsum4(arrM_1);

            if(remain_Tbr==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
            }
            else if(remain_Tbr==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
            }
            else if(remain_Tbr==3)
            {
                unsigned int j = this->nSpecies-3;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M0 = M0 + TBF1DRowi0[j+2]*C[j+2];
                M1 = M1 + TBF1DRowi1[j+2]*C[j+2];
            }
            this->tmp_M[i+0] = M0;
            this->tmp_M[i+1] = M1;
        }
        else if(Tremain==1)
        {
            unsigned int i =(Itbr[5]) -1;
            double* __restrict__ TBF1DRowi0 = &ThirdBodyFactor1D[(i+0)*nSpecies];            
            double M0 = 0;
            __m256d arrM_0 = _mm256_setzero_pd();
            for(unsigned int j  = 0;j<this->nSpecies-remain_Tbr;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
            }
            M0 = M0 + hsum4(arrM_0);
            if(remain_Tbr==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
            }
            else if(remain_Tbr==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
            }
            else if(remain_Tbr==3)
            {
                unsigned int j = this->nSpecies-3;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M0 = M0 + TBF1DRowi0[j+2]*C[j+2];
            }
            this->tmp_M[i+0] = M0;
        }
    }

    for(unsigned int i = 0; i < this->n_ThirdBodyReaction; i++)
    {
        double M = this->tmp_M[i+this->Itbr[1]];
        const unsigned int j = i + Ikf[3];
        this->dKfdC_[i+this->Itbr[1]] = this->Kf_[j];   
        this->Kf_[j] = this->Kf_[j]*M;
        this->dKfdT_[j] = this->dKfdT_[j]*M;
    }
    


    for(unsigned int i = 0; i < this->n_NonEquilibriumThirdBodyReaction; i++)
    {
        double Mfwd = this->tmp_M[i];
        double Mrev = this->tmp_M[this->Itbr[4]+i];
        this->dKfdC_[i] = this->Kf_[Ikf[2]+i];
        this->dKfdC_[this->Itbr[4]+i] = this->Kf_[this->Ikf[10]+i];
        this->Kf_[Ikf[2]+i] = this->Kf_[Ikf[2]+i]*Mfwd;
        this->dKfdT_[Ikf[2]+i] = this->dKfdT_[Ikf[2]+i]*Mfwd;
        this->Kf_[Ikf[10]+i] = this->Kf_[Ikf[10]+i]*Mrev;
        this->dKfdT_[Ikf[10]+i] = this->dKfdT_[Ikf[10]+i]*Mrev;
    } 


    {
        size_t remain_Lindemann = (Lindemann.size())%4;
        for (size_t i = 0;i<Lindemann.size()-remain_Lindemann;i=i+4)
        {
            const unsigned int j0 = this->Lindemann[i+0];
            const unsigned int j1 = this->Lindemann[i+1];
            const unsigned int j2 = this->Lindemann[i+2];
            const unsigned int j3 = this->Lindemann[i+3];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            const unsigned int k2 = j2 - Ikf[4];
            const unsigned int k3 = j3 - Ikf[4];
            __m256d Kinf = _mm256_loadu_pd(&this->Kf_[j0 + offset_kinf]);    
            __m256d one = _mm256_set1_pd(1.0);       
            __m256d invKinf = _mm256_div_pd(_mm256_set1_pd(1.0),Kinf);
            __m256d dKinfdT = _mm256_loadu_pd(&this->dKfdT_[j0 + offset_kinf]);
            __m256d K0 = _mm256_loadu_pd(&this->Kf_[j0]);        
            __m256d dK0dT = _mm256_loadu_pd(&this->dKfdT_[j0]);     
            __m256d M = _mm256_loadu_pd(&tmp_M[m0]);
            __m256d Pr = _mm256_mul_pd(_mm256_mul_pd(K0,M),invKinf);
            __m256d dPrdT = _mm256_fmsub_pd(M,dK0dT,_mm256_mul_pd(Pr,dKinfdT));
            dPrdT = _mm256_mul_pd(dPrdT,invKinf);
            __m256d k = _mm256_setr_pd(k0,k1,k2,k3);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d dKdT = _mm256_blendv_pd(dK0dT,_mm256_mul_pd(Pr,dKinfdT),cmp);
            __m256d K = _mm256_blendv_pd(K0,Kinf,cmp);
            __m256d KK = _mm256_blendv_pd(_mm256_mul_pd(K0,invKinf),one,cmp);
            __m256d tmp = _mm256_div_pd(one,_mm256_add_pd(one,Pr));
            __m256d N1 = _mm256_blendv_pd(-tmp,tmp,cmp);
            __m256d N = _mm256_mul_pd(tmp,K0);
            __m256d dKfdT = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tmp,tmp),dPrdT),K);
            dKfdT = _mm256_fmadd_pd(tmp,dKdT,dKfdT);
            _mm256_storeu_pd(&this->dKfdT_[j0],dKfdT);
            __m256d dKfdC = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(K0,tmp),KK),N1);
            _mm256_storeu_pd(&this->dKfdC_[m0],dKfdC);
            __m256d KF = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);
            _mm256_storeu_pd(&this->Kf_[j0],KF);
        }
        if(remain_Lindemann==1)
        {
            size_t i = this->Lindemann.size()-1;
            const unsigned int j0 = this->Lindemann[i+0];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const double Kinf0 = this->Kf_[j0 + offset_kinf];
            const double dKinfdT0 = this->dKfdT_[j0 + offset_kinf];
            const double K00 = this->Kf_[j0];
            double M0 = tmp_M[m0];
            const double invKinf0 = 1.0/Kinf0;
            const double Pr0 = K00*M0*invKinf0; 
            const double dK0dT0 =  this->dKfdT_[j0];
            const double dPrdT0 = (M0*dK0dT0-Pr0*dKinfdT0)*invKinf0;
            const double invOnePlusPr0 = 1/(1+Pr0);
            const double dKdT0   = j0 - Ikf[4]<this->n_Fall_Off_Reaction?Pr0*dKinfdT0:dK0dT0;
            const double K0      = j0 - Ikf[4]<this->n_Fall_Off_Reaction?Kinf0      :K00;
            const double KK0     = j0 - Ikf[4]<this->n_Fall_Off_Reaction?1         :K00*invKinf0;
            const double N10     = j0 - Ikf[4]<this->n_Fall_Off_Reaction?invOnePlusPr0  :-invOnePlusPr0;
            const double N0  = invOnePlusPr0*1*K00;
            this->dKfdT_[j0] = invOnePlusPr0*dKdT0 + invOnePlusPr0*invOnePlusPr0*dPrdT0*K0;
            this->dKfdC_[m0] =  K00*KK0*(N10)*invOnePlusPr0; 
            this->Kf_[j0] = k0<this->n_Fall_Off_Reaction ? M0*N0 : N0;    
        }
        else if (remain_Lindemann==2)
        {
            size_t i = this->Lindemann.size()-2;

            const unsigned int j0 = this->Lindemann[i+0];
            const unsigned int j1 = this->Lindemann[i+1];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const unsigned int m1 = j1 - Ikf[4] + Itbr[2];            
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            __m256d Kinf = _mm256_setr_pd(Kf_[j0 + offset_kinf],Kf_[j1 + offset_kinf],1,1);    
            __m256d one = _mm256_set1_pd(1.0);       
            __m256d invKinf = _mm256_div_pd(_mm256_set1_pd(1.0),Kinf);
            __m256d dKinfdT = _mm256_setr_pd(dKfdT_[j0 + offset_kinf],dKfdT_[j1 + offset_kinf],1,1);
            __m256d K0 = _mm256_setr_pd(Kf_[j0],Kf_[j1],1,1);        
            __m256d dK0dT = _mm256_setr_pd(dKfdT_[j0],dKfdT_[j1],1,1);     
            __m256d M = _mm256_setr_pd(tmp_M[m0],tmp_M[m1],1,1);
            __m256d Pr = _mm256_mul_pd(_mm256_mul_pd(K0,M),invKinf);
            __m256d dPrdT = _mm256_fmsub_pd(M,dK0dT,_mm256_mul_pd(Pr,dKinfdT));
            dPrdT = _mm256_mul_pd(dPrdT,invKinf);
            __m256d k = _mm256_setr_pd(k0,k1,1,1);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d dKdT = _mm256_blendv_pd(dK0dT,_mm256_mul_pd(Pr,dKinfdT),cmp);
            __m256d K = _mm256_blendv_pd(K0,Kinf,cmp);
            __m256d KK = _mm256_blendv_pd(_mm256_mul_pd(K0,invKinf),one,cmp);
            __m256d tmp = _mm256_div_pd(one,_mm256_add_pd(one,Pr));
            __m256d N1 = _mm256_blendv_pd(-tmp,tmp,cmp);
            __m256d N = _mm256_mul_pd(tmp,K0);
            __m256d dKfdT = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tmp,tmp),dPrdT),K);
            dKfdT = _mm256_fmadd_pd(tmp,dKdT,dKfdT);
            dKfdT_[j0] = get_elem0(dKfdT);
            dKfdT_[j1] = get_elem1(dKfdT);
            __m256d dKfdC = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(K0,tmp),KK),N1);
            dKfdC_[m0] = get_elem0(dKfdC);
            dKfdC_[m1] = get_elem1(dKfdC);
            __m256d KF = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);   
            Kf_[j0] = get_elem0(KF);
            Kf_[j1] = get_elem1(KF);
        }
        else if (remain_Lindemann==3)
        {
            size_t i = this->Lindemann.size()-3;
            const unsigned int j0 = this->Lindemann[i+0];
            const unsigned int j1 = this->Lindemann[i+1];
            const unsigned int j2 = this->Lindemann[i+2];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            const unsigned int k2 = j2 - Ikf[4];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const unsigned int m1 = j1 - Ikf[4] + Itbr[2];
            const unsigned int m2 = j2 - Ikf[4] + Itbr[2];
            __m256d Kinf = _mm256_setr_pd(Kf_[j0+offset_kinf],Kf_[j1+offset_kinf],Kf_[j2+offset_kinf],1);    
            __m256d one = _mm256_set1_pd(1.0);       
            __m256d invKinf = _mm256_div_pd(_mm256_set1_pd(1.0),Kinf);
            __m256d dKinfdT = _mm256_setr_pd(dKfdT_[j0+offset_kinf],dKfdT_[j1+offset_kinf],dKfdT_[j2+offset_kinf],1);
            __m256d K0 = _mm256_setr_pd(Kf_[j0],Kf_[j1],Kf_[j2],1);        
            __m256d dK0dT = _mm256_setr_pd(dKfdT_[j0],dKfdT_[j1],dKfdT_[j2],1);     
            __m256d M = _mm256_setr_pd(tmp_M[m0],tmp_M[m1],tmp_M[m2],1);
            __m256d Pr = _mm256_mul_pd(_mm256_mul_pd(K0,M),invKinf);
            __m256d dPrdT = _mm256_fmsub_pd(M,dK0dT,_mm256_mul_pd(Pr,dKinfdT));
            dPrdT = _mm256_mul_pd(dPrdT,invKinf);
            __m256d k = _mm256_setr_pd(k0,k1,k2,1);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d dKdT = _mm256_blendv_pd(dK0dT,_mm256_mul_pd(Pr,dKinfdT),cmp);
            __m256d K = _mm256_blendv_pd(K0,Kinf,cmp);
            __m256d KK = _mm256_blendv_pd(_mm256_mul_pd(K0,invKinf),one,cmp);
            __m256d tmp = _mm256_div_pd(one,_mm256_add_pd(one,Pr));
            __m256d N1 = _mm256_blendv_pd(-tmp,tmp,cmp);
            __m256d N = _mm256_mul_pd(tmp,K0);
            __m256d dKfdT = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(tmp,tmp),dPrdT),K);
            dKfdT = _mm256_fmadd_pd(tmp,dKdT,dKfdT);
            this->dKfdT_[j0] = get_elem0(dKfdT);
            this->dKfdT_[j1] = get_elem1(dKfdT);
            this->dKfdT_[j2] = get_elem2(dKfdT);
            __m256d dKfdC = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(K0,tmp),KK),N1);
            this->dKfdC_[m0] = get_elem0(dKfdC);
            this->dKfdC_[m1] = get_elem1(dKfdC);
            this->dKfdC_[m2] = get_elem2(dKfdC);
            __m256d KF = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);
            this->Kf_[j0] = get_elem0(KF);
            this->Kf_[j1] = get_elem1(KF);
            this->Kf_[j2] = get_elem2(KF);
        }   
    }


    {
        size_t remain_Troe = (Troe.size())%4;
        for (size_t i = 0;i<Troe.size()-remain_Troe;i=i+4)
        {
            const unsigned int j0 = this->Troe[i+0];
            const unsigned int j1 = this->Troe[i+1];
            const unsigned int j2 = this->Troe[i+2];
            const unsigned int j3 = this->Troe[i+3];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            const unsigned int k2 = j2 - Ikf[4];
            const unsigned int k3 = j3 - Ikf[4];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            __m256d Kinf = _mm256_loadu_pd(&this->Kf_[j0+offset_kinf]);
            __m256d invKinf = _mm256_div_pd(_mm256_set1_pd(1.0),Kinf);
            __m256d dKinfdT = _mm256_loadu_pd(&this->dKfdT_[j0+offset_kinf]);
            __m256d K0 = _mm256_loadu_pd(&this->Kf_[j0]);           
            __m256d M = _mm256_loadu_pd(&this->tmp_M[m0]);
            __m256d Pr = _mm256_mul_pd(_mm256_mul_pd(M,K0),invKinf);
            __m256d small = _mm256_set1_pd(2.2e-16);
            __m256d cmp_result_Pr = _mm256_cmp_pd(Pr,small,_CMP_GE_OQ);
            Pr = _mm256_add_pd(Pr,_mm256_set1_pd(1e-100));
            const double invLog10 = 1.0/std::log(10);
            __m256d logPr_ = _mm256_mul_pd(vec256_logd(_mm256_max_pd(small,Pr)),_mm256_set1_pd(invLog10));
            __m256d InvTsss = _mm256_loadu_pd(&this->invTsss_[i]);
            __m256d InvTs = _mm256_loadu_pd(&this->invTs_[i]);
            __m256d Tss = _mm256_loadu_pd(&this->Tss_[i]);
            __m256d expTTsss = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies]);
            __m256d expTTss = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies+this->Troe.size()]);
            __m256d expTTs = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies+this->Troe.size()*2]);
            __m256d one = _mm256_set1_pd(1.0);
            __m256d alpha = _mm256_loadu_pd(&this->alpha_[i]);
            __m256d Fcent  = _mm256_mul_pd(_mm256_sub_pd(one,alpha),expTTsss);
            Fcent = _mm256_fmadd_pd(alpha,expTTs,Fcent);
            Fcent = _mm256_add_pd(expTTss,Fcent);
            __m256d logFcent = _mm256_mul_pd(vec256_logd(_mm256_max_pd(Fcent,small)),_mm256_set1_pd(invLog10));
            __m256d c = _mm256_fmadd_pd(logFcent,_mm256_set1_pd(0.67),_mm256_set1_pd(0.4));
            __m256d n = _mm256_fmadd_pd(logFcent,_mm256_set1_pd(-1.27),_mm256_set1_pd(0.75));
            __m256d x1 = _mm256_fmadd_pd(_mm256_sub_pd(c,logPr_),_mm256_set1_pd(0.14),n);
            __m256d invx1 = _mm256_div_pd(one,x1);
            __m256d x2 = _mm256_mul_pd(_mm256_sub_pd(logPr_,c),invx1);
            __m256d x3 = _mm256_fmadd_pd(x2,x2,one);
            __m256d invx3 = _mm256_div_pd(one,x3);
            __m256d x4 = _mm256_mul_pd(logFcent,invx3);
            __m256d  F = vec256_powd(_mm256_set1_pd(10),x4);
            __m256d logTen = _mm256_set1_pd(std::log(10));
            __m256d dFcentdT = _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(alpha,one),InvTsss),expTTsss);
            dFcentdT = _mm256_sub_pd(dFcentdT,_mm256_mul_pd(_mm256_mul_pd(alpha,InvTs),expTTs));
            __m256d invT2 = _mm256_set1_pd(invT*invT);
            dFcentdT = _mm256_fmadd_pd(expTTss,_mm256_mul_pd(Tss,invT2),dFcentdT);
            __m256d cmp2 = _mm256_cmp_pd(Fcent,small,_CMP_GE_OQ);
            __m256d dlogFcentdT = _mm256_div_pd(_mm256_div_pd(dFcentdT,_mm256_max_pd(Fcent,small)),logTen);
            dlogFcentdT = _mm256_blendv_pd(_mm256_setzero_pd(), dlogFcentdT, cmp2);
            __m256d dcdT = _mm256_mul_pd(dlogFcentdT,_mm256_set1_pd(-0.67));
            __m256d dndT = _mm256_mul_pd(dlogFcentdT,_mm256_set1_pd(-1.27));
            __m256d dx1dT = _mm256_fmadd_pd(dcdT,_mm256_set1_pd(-0.14),dndT);
            __m256d dx2dT = _mm256_mul_pd(_mm256_sub_pd(dcdT,_mm256_mul_pd(x2,dx1dT)),invx1);
            __m256d dx3dT = _mm256_mul_pd(_mm256_mul_pd(x2,dx2dT),_mm256_set1_pd(2.0));
            __m256d dx4dT = _mm256_mul_pd(_mm256_sub_pd(dlogFcentdT,_mm256_mul_pd(x4,dx3dT)),invx3);
            __m256d dFdT = _mm256_mul_pd(logTen,_mm256_mul_pd(F,dx4dT));
            __m256d dlogPrdPr = _mm256_div_pd(_mm256_set1_pd(1.0),_mm256_mul_pd(Pr,logTen));
            dlogPrdPr = _mm256_blendv_pd(_mm256_setzero_pd(), dlogPrdPr, cmp_result_Pr);
            __m256d dx1dPr = _mm256_mul_pd(dlogPrdPr,_mm256_set1_pd(-0.14));
            __m256d dx2dPr = _mm256_mul_pd(_mm256_sub_pd(dlogPrdPr,_mm256_mul_pd(x2,dx1dPr)),invx1);
            __m256d dx3dPr = _mm256_mul_pd(_mm256_mul_pd(x2,dx2dPr),_mm256_set1_pd(2.0));
            __m256d dx4dPr = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0),x4),dx3dPr),invx3);
            __m256d dFdPr  = _mm256_mul_pd(_mm256_mul_pd(logTen,F),dx4dPr);    
            __m256d dK0dT = _mm256_loadu_pd(&this->dKfdT_[j0]);            
            __m256d dPrdT = _mm256_mul_pd(_mm256_fmsub_pd(M,dK0dT,_mm256_mul_pd(Pr,dKinfdT)),invKinf);
            dFdT = _mm256_fmadd_pd(dFdPr,dPrdT,dFdT);
            __m256d k = _mm256_setr_pd(k0,k1,k2,k3);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d dKdT = _mm256_blendv_pd(dK0dT, _mm256_mul_pd(Pr,dKinfdT), cmp);
            __m256d K = _mm256_blendv_pd(K0, Kinf, cmp);
            __m256d MM = _mm256_blendv_pd(_mm256_set1_pd(1), M, cmp);
            __m256d KK = _mm256_blendv_pd(_mm256_mul_pd(K0,invKinf), _mm256_set1_pd(1), cmp);
            __m256d invOnePlusPr = _mm256_div_pd(_mm256_set1_pd(1.0),_mm256_add_pd(_mm256_set1_pd(1.0),Pr));
            __m256d N1 = _mm256_mul_pd(F,invOnePlusPr);
            N1 = _mm256_blendv_pd(_mm256_sub_pd(_mm256_setzero_pd(),N1),N1,cmp);            
            __m256d N2 = _mm256_blendv_pd(dFdPr,_mm256_mul_pd(Pr,dFdPr),cmp);
            __m256d N = _mm256_mul_pd(_mm256_mul_pd(F,K0),invOnePlusPr);
            __m256d dKfdT = _mm256_mul_pd(_mm256_mul_pd(F,invOnePlusPr),dKdT);
            dKfdT = _mm256_fmadd_pd(K,_mm256_mul_pd(_mm256_mul_pd(F,_mm256_mul_pd(invOnePlusPr,invOnePlusPr)),dPrdT),dKfdT);
            dKfdT = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_mul_pd(K0,invOnePlusPr),dFdT),MM,dKfdT); 
            _mm256_storeu_pd(&this->dKfdT_[j0],dKfdT);           
            __m256d dKfdC = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(K0,invOnePlusPr),KK),_mm256_add_pd(N1,N2));
            _mm256_storeu_pd(&this->dKfdC_[m0],dKfdC);     
            __m256d KF = _mm256_blendv_pd(N,_mm256_mul_pd(N,M),cmp);
            _mm256_storeu_pd(&this->Kf_[j0],KF);
        }
        if(remain_Troe==1)
        {
            this->Troe_Jac_1();
        }
        else if(remain_Troe==2)
        {

            this->Troe_Jac_2();
        }
        else if(remain_Troe==3)
        {
            this->Troe_Jac_3();
        }
    }        


    {
        for (unsigned int i = 0;i<SRI.size();i++)
        {
            const unsigned int j = this->SRI[i];
            const unsigned int k = j - Ikf[4];
            const unsigned int m = j - Ikf[4] + Itbr[2];
            const double Kinf = this->Kf_[j+offset_kinf];
            const double invKinf = 1.0/Kinf;
            const double K0 = this->Kf_[j];
            const double dKinfdT = this->dKfdT_[j+offset_kinf];
            double F ;
            double dFdT;
            double dFdPr;
            double M = tmp_M[m];
            const double Pr = K0*M*invKinf; 
            this->SRI_F_dFdT_dFdPr(Temperature,Pr,i,F,dFdT,dFdPr);
            const double dK0dT =  this->dKfdT_[j]; 
            const double invOnePlusPr = 1.0/(1.0+Pr);
            const double dPrdT = (M*dK0dT-Pr*dKinfdT)*invKinf;
            const double dKdT   = k<this->n_Fall_Off_Reaction?Pr*dKinfdT:dK0dT;
            const double K      = k<this->n_Fall_Off_Reaction?Kinf      :K0;
            const double MM     = k<this->n_Fall_Off_Reaction?M         :1;
            const double KK     = k<this->n_Fall_Off_Reaction?1         :K0*invKinf;
            const double N1     = k<this->n_Fall_Off_Reaction?F*invOnePlusPr  :-F*invOnePlusPr;
            const double N2     = k<this->n_Fall_Off_Reaction?Pr*dFdPr  :dFdPr;
            const double N  = invOnePlusPr*F*K0;
            this->dKfdT_[j] = F*invOnePlusPr*dKdT 
            + F*invOnePlusPr*invOnePlusPr*dPrdT*K 
            + K0*invOnePlusPr*dFdT*MM;
            this->dKfdC_[m] =  K0*invOnePlusPr*KK*(N1 + N2); 
            this->Kf_[j] = k<this->n_Fall_Off_Reaction ? M*N : N;   
        }
    }

    for(unsigned int z = 0; z < Ikf[1];z++)
    {
        const unsigned int i = z ;

        const size_t j = this->lhsIndex[i].size();
        const size_t k = this->rhsIndex[i].size();

        if(j==2)
        {
            if(k==2)        {this->JF22(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF21(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF23(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
        }
        else if(j==1)
        {
            if(k==2)        {this->JF12(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF11(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF13(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
        }
        else if(j==3)
        {
            if(k==2)        {this->JF32(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF31(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF33(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}                        
        }
        if(j>3 || k>3){this->JFG(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
    }

    for(unsigned int z = Ikf[1]; z < Ikf[3];z++)
    {
        const unsigned int i = z ;

        const size_t j = this->lhsIndex[i].size();
        const size_t k = this->rhsIndex[i].size();

        if(j==2)
        {
            if(k==2)        {this->JF22(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==1)   {this->JF21(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==3)   {this->JF23(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
        }
        else if(j==1)
        {
            if(k==2)        {this->JF12(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==1)   {this->JF11(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==3)   {this->JF13(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
        }
        else if(j==3)
        {
            if(k==2)        {this->JF32(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==1)   {this->JF31(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
            else if(k==3)   {this->JF33(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}                        
        }
        if(j>3 || k>3){this->JFG(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp);}
    }

    for(unsigned int z = Ikf[3]; z < Ikf[7];z++)
    {
        const unsigned int i = z ;

        const size_t j = this->lhsIndex[i].size();
        const size_t k = this->rhsIndex[i].size();
        if(j==2)
        {
            if(k==2)        {this->JF22(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF21(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF23(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
        }
        else if(j==1)
        {
            if(k==2)        {this->JF12(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF11(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF13(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
        }
        else if(j==3)
        {
            if(k==2)        {this->JF32(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==1)   {this->JF31(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            else if(k==3)   {this->JF33(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}                        
        }
        if(j>3 || k>3){this->JFG(i,this->Kf_[z],this->dKfdT_[z],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
    }


    if(n_PlogReaction>0)
    {
        const double logP = std::log(p);

        if(uniformPressureRange==false)
        {
            for(unsigned int i = 0; i< this->n_PlogReaction; i ++)
            {
                const size_t length = this->Prange[i].size();
                if(p<=this->Prange[i][0])
                {

                    this->Kf_Plog[i] =  this->logAPlog[i][0] + 
                                        this->betaPlog[i][0]*this->logT - 
                                        this->TaPlog[i][0]*invT;
                    this->dKfdT_Plog[i] = (this->betaPlog[i][0]+this->TaPlog[i][0]*invT)*invT;
                }
                else if(p>=this->Prange[i][length-1])
                {
                    this->Kf_Plog[i] =   this->logAPlog[i][length-1] + 
                                                this->betaPlog[i][length-1]*this->logT - 
                                                this->TaPlog[i][length-1]*invT;  
                    this->dKfdT_Plog[i] = (this->betaPlog[i][length-1]+this->TaPlog[i][length-1]*invT)*invT;
                }
                else
                {
                    size_t index = 0;
                    for(size_t j = 0; j < length-1;j++)
                    {
                        if(this->Prange[i][j]<=p && p<=this->Prange[i][j+1])
                        {
                            index = j;
                            break;
                        }
                    }


                    this->Kf_Plog[i] =  this->logAPlog[i][index] + 
                                        this->betaPlog[i][index]*this->logT - 
                                        this->TaPlog[i][index]*invT +
                                        (
                                            this->logAPlog[i][index+1] - this->logAPlog[i][index] +
                                            (this->betaPlog[i][index+1] - this->betaPlog[i][index])*this->logT -
                                            (this->TaPlog[i][index+1] - this->TaPlog[i][index])*invT
                                        )*
                                        (
                                            (logP - this->logPi[i][index])/
                                            (this->logPi[i][index+1] - this->logPi[i][index])
                                        ); 
                    this->dKfdT_Plog[i] = invT*
                    (
                        this->betaPlog[i][index]+this->TaPlog[i][index]*invT+
                        (
                            this->betaPlog[i][index+1] - this->betaPlog[i][index] +
                            (this->TaPlog[i][index+1] - this->TaPlog[i][index])*invT
                        )*
                        (
                            (logP - this->logPi[i][index])/
                            (this->logPi[i][index+1] - this->logPi[i][index])
                        )
                    );
                }

            }
        }
        else 
        {
            const size_t length = this->Prange[0].size();

            if(p<=this->Prange[0][0])
            {
                for(unsigned int i = 0; i< this->n_PlogReaction; i++)
                {
                    this->Kf_Plog[i] =  this->logAPlog[i][0] + 
                                        this->betaPlog[i][0]*this->logT - 
                                        this->TaPlog[i][0]*invT;
                    this->dKfdT_Plog[i] = (this->betaPlog[i][0]+this->TaPlog[i][0]*invT)*invT;
                }     
            }
            else if(p>=this->Prange[0][length-1])
            {
                for(unsigned int i = 0; i< this->n_PlogReaction; i++)
                {
                    this->Kf_Plog[i] =  this->logAPlog[i][length -1] + 
                                        this->betaPlog[i][length -1]*this->logT - 
                                        this->TaPlog[i][length -1]*invT;
                    this->dKfdT_Plog[i] = (this->betaPlog[i][length -1]+this->TaPlog[i][length -1]*invT)*invT;
                }             
            }
            else
            {
                size_t index = 0;
                for(size_t j = 0; j < length-1;j++)
                {
                    if(this->Prange[0][j]<=p && p<=this->Prange[0][j+1])
                    {
                        index = j;
                        break;
                    }
                }
                for(unsigned int j = 0; j < this->n_PlogReaction; j++)
                {
                    this->Kf_Plog[j] =  this->logAPlog[j][index] + 
                    this->betaPlog[j][index]*this->logT - 
                    this->TaPlog[j][index]*invT +
                    (
                        (this->logAPlog[j][index+1] - this->logAPlog[j][index]) +
                        (this->betaPlog[j][index+1] - this->betaPlog[j][index])*this->logT -
                        (this->TaPlog[j][index+1] - this->TaPlog[j][index])*invT
                    )*
                    (
                        (logP - this->logPi[j][index])/
                        (this->logPi[j][index+1] - this->logPi[j][index])
                    ); 
                    this->dKfdT_Plog[j] = invT*
                    (
                        this->betaPlog[j][index]+this->TaPlog[j][index]*invT+
                        (
                            this->betaPlog[j][index+1] - this->betaPlog[j][index] +
                            (this->TaPlog[j][index+1] - this->TaPlog[j][index])*invT
                        )*
                        (
                            (logP - this->logPi[j][index])/
                            (this->logPi[j][index+1] - this->logPi[j][index])
                        ) 
                    );
                }
            }
        }


        unsigned int remain_Plog = this->n_PlogReaction%4;
        for(unsigned int i = 0; i < this->n_PlogReaction-remain_Plog; i=i+4)
        {

            __m256d Kfv = _mm256_loadu_pd(&this->Kf_Plog[i]);
            Kfv = vec256_expd(Kfv);
            _mm256_storeu_pd(&this->Kf_Plog[i],Kfv);
            __m256d dKfdTv = _mm256_loadu_pd(&this->dKfdT_Plog[i]);
            dKfdTv = _mm256_mul_pd(dKfdTv,Kfv);
            _mm256_storeu_pd(&this->dKfdT_Plog[i],dKfdTv);

        }
        if(remain_Plog==1)
        {

            this->Kf_Plog[this->n_PlogReaction-1] = std::exp(this->Kf_Plog[this->n_PlogReaction-1]);
            this->dKfdT_Plog[this->n_PlogReaction-1] = 
            this->Kf_Plog[this->n_PlogReaction-1] * this->dKfdT_Plog[this->n_PlogReaction-1];
        }
        else if(remain_Plog==2)
        {

            __m256d A_ = _mm256_setr_pd(Kf_Plog[this->n_PlogReaction-2],Kf_Plog[this->n_PlogReaction-1],0,0);
            A_ = vec256_expd(A_);
            this->Kf_Plog[this->n_PlogReaction-2] = get_elem0(A_);
            this->Kf_Plog[this->n_PlogReaction-1] = get_elem1(A_);

            this->dKfdT_Plog[this->n_PlogReaction-2] = this->Kf_Plog[this->n_PlogReaction-2] * this->dKfdT_Plog[this->n_PlogReaction-2];
            this->dKfdT_Plog[this->n_PlogReaction-1] = this->Kf_Plog[this->n_PlogReaction-1] * this->dKfdT_Plog[this->n_PlogReaction-1];
        }
        else if(remain_Plog==3)
        {

            __m256d A_ = _mm256_setr_pd(Kf_Plog[this->n_PlogReaction-3],Kf_Plog[this->n_PlogReaction-2],Kf_Plog[this->n_PlogReaction-1],0);
            A_ = vec256_expd(A_);
            this->Kf_Plog[this->n_PlogReaction-3] = get_elem0(A_);
            this->Kf_Plog[this->n_PlogReaction-2] = get_elem1(A_);
            this->Kf_Plog[this->n_PlogReaction-1] = get_elem2(A_);
            this->dKfdT_Plog[this->n_PlogReaction-3] = this->Kf_Plog[this->n_PlogReaction-3] * this->dKfdT_Plog[this->n_PlogReaction-3];
            this->dKfdT_Plog[this->n_PlogReaction-2] = this->Kf_Plog[this->n_PlogReaction-2] * this->dKfdT_Plog[this->n_PlogReaction-2];
            this->dKfdT_Plog[this->n_PlogReaction-1] = this->Kf_Plog[this->n_PlogReaction-1] * this->dKfdT_Plog[this->n_PlogReaction-1];
        }

        for(unsigned int i = 0; i < this->n_PlogReaction; i ++)
        {
            unsigned int n = Ikf[7] + i;
            size_t j = this->lhsIndex[n].size();
            size_t k = this->rhsIndex[n].size(); 

            if(j==2)
            {
                if(k==2)        {this->JF22(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==1)   {this->JF21(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==3)   {this->JF23(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            }
            else if(j==1)      
            {
                if(k==2)        {this->JF12(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==1)   {this->JF11(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==3)   {this->JF13(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
            }
            else if(j==3)
            {
                if(k==2)        {this->JF32(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==1)   {this->JF31(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
                else if(k==3)   {this->JF33(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}                        
            }
            if(j>3 || k>3){this->JFG(n,this->Kf_Plog[i],this->dKfdT_Plog[i],C,dNdtByV,ddNdtByVdcTp,&this->tmp_Exp[0],dBdT);}
        }
    }


}




void 
OptReaction::ddYdtdY_Vec
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* const* __restrict__ ArrRes = this->ArrPtr.data();
    unsigned int remain = this->nSpecies%4;
    for(unsigned int j = 0; j < this->nSpecies - remain; j = j + 4)
    {
        const double rhoMvj0_ = rhoMvj[j+0];     
        const double rhoMvj1_ = rhoMvj[j+1]; 
        const double rhoMvj2_ = rhoMvj[j+2]; 
        const double rhoMvj3_ = rhoMvj[j+3]; 
        __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies-remain; i=i+4)
        {
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*this->invW[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*this->invW[i+3]*YTp[i+3];
            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvj_));
        }
        if(remain==1)
        {
            unsigned int i = this->nSpecies-1;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;
        }
        else if(remain==2)
        {
            unsigned int i = this->nSpecies-2;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;   
            ArrRes[i+1][0] = rhoMByWiYTPi1*rhoMvj0_;
            ArrRes[i+1][1] = rhoMByWiYTPi1*rhoMvj1_;
            ArrRes[i+1][2] = rhoMByWiYTPi1*rhoMvj2_;
            ArrRes[i+1][3] = rhoMByWiYTPi1*rhoMvj3_;          
        }
        else if(remain==3)
        {
            unsigned int i = this->nSpecies-3;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*this->invW[i+2]*YTp[i+2];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;   
            ArrRes[i+1][0] = rhoMByWiYTPi1*rhoMvj0_;
            ArrRes[i+1][1] = rhoMByWiYTPi1*rhoMvj1_;
            ArrRes[i+1][2] = rhoMByWiYTPi1*rhoMvj2_;
            ArrRes[i+1][3] = rhoMByWiYTPi1*rhoMvj3_;  
            ArrRes[i+2][0] = rhoMByWiYTPi2*rhoMvj0_;
            ArrRes[i+2][1] = rhoMByWiYTPi2*rhoMvj1_;
            ArrRes[i+2][2] = rhoMByWiYTPi2*rhoMvj2_;
            ArrRes[i+2][3] = rhoMByWiYTPi2*rhoMvj3_;  
        }
        ArrRes[j+0][0] += rhoM*this->invW[j+0];
        ArrRes[j+1][1] += rhoM*this->invW[j+1];
        ArrRes[j+2][2] += rhoM*this->invW[j+2];
        ArrRes[j+3][3] += rhoM*this->invW[j+3];  
        for(unsigned int i=0; i<this->nSpecies - remain; i=i+4)
        {
            const double Wi0ByrhoM_ = WiByrhoM[i+0];
            const double Wi1ByrhoM_ = WiByrhoM[i+1];
            const double Wi2ByrhoM_ = WiByrhoM[i+2];
            const double Wi3ByrhoM_ = WiByrhoM[i+3];

            const double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            const double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            const double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            const double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;

            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = ddNdtByVdcTp[(i+0)*(n_)+k];
                const double ddNi1dtByVdck = ddNdtByVdcTp[(i+1)*(n_)+k];
                const double ddNi2dtByVdck = ddNdtByVdcTp[(i+2)*(n_)+k];
                const double ddNi3dtByVdck = ddNdtByVdcTp[(i+3)*(n_)+k];
                __m256d dCkdYj = _mm256_loadu_pd(&ArrRes[k][0]);
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck),dCkdYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck),dCkdYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck),dCkdYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck),dCkdYj,ddNi3dtByVdYj);
            }
            __m256d ddYi0dtdYj = _mm256_loadu_pd(&J[(i+0)*(n_)+ j+0]);
            __m256d ddYi1dtdYj = _mm256_loadu_pd(&J[(i+1)*(n_)+ j+0]);
            __m256d ddYi2dtdYj = _mm256_loadu_pd(&J[(i+2)*(n_)+ j+0]);
            __m256d ddYi3dtdYj = _mm256_loadu_pd(&J[(i+3)*(n_)+ j+0]);
            __m256d WiByrhoM_ = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidt = _mm256_set1_pd(dYi0dt);
            ddYi0dtdYj = _mm256_add_pd(_mm256_mul_pd(WiByrhoM_,ddNi0dtByVdYj),_mm256_mul_pd(rhoMvj_,dYidt));
            WiByrhoM_ = _mm256_set1_pd(Wi1ByrhoM_);
            dYidt = _mm256_set1_pd(dYi1dt);
            ddYi1dtdYj = _mm256_add_pd(_mm256_mul_pd(WiByrhoM_,ddNi1dtByVdYj),_mm256_mul_pd(rhoMvj_,dYidt));
            WiByrhoM_ = _mm256_set1_pd(Wi2ByrhoM_);
            dYidt = _mm256_set1_pd(dYi2dt);
            ddYi2dtdYj = _mm256_add_pd(_mm256_mul_pd(WiByrhoM_,ddNi2dtByVdYj),_mm256_mul_pd(rhoMvj_,dYidt));
            WiByrhoM_ = _mm256_set1_pd(Wi3ByrhoM_);
            dYidt = _mm256_set1_pd(dYi3dt);
            ddYi3dtdYj = _mm256_add_pd(_mm256_mul_pd(WiByrhoM_,ddNi3dtByVdYj),_mm256_mul_pd(rhoMvj_,dYidt)); 
            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
        }     
        
        if(remain==1)
        {
            unsigned int i = this->nSpecies-1;
            const double WiByrhoM_ = WiByrhoM[i];
    
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;
    
            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;
            double ddNidtByVdYj2 = 0;
            double ddNidtByVdYj3 = 0;
    
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];
                ddNidtByVdYj2 += ddNidtByVdck*ArrRes[k][2];
                ddNidtByVdYj3 += ddNidtByVdck*ArrRes[k][3];
            }
    
            double& ddYidtdYj0 = J[i*(n_) + j+0];
            double& ddYidtdYj1 = J[i*(n_) + j+1];
            double& ddYidtdYj2 = J[i*(n_) + j+2];
            double& ddYidtdYj3 = J[i*(n_) + j+3];
    
            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;     
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;         
            ddYidtdYj2 = WiByrhoM_*ddNidtByVdYj2 + rhoMvj2_*dYidt;
            ddYidtdYj3 = WiByrhoM_*ddNidtByVdYj3 + rhoMvj3_*dYidt;  
        }
        else if(remain==2)
        {
            unsigned int i0 = this->nSpecies-2;
            unsigned int i1 = this->nSpecies-1;

            const double Wi0ByrhoM_ = WiByrhoM[i0];
            const double Wi1ByrhoM_ = WiByrhoM[i1];

            double dYi0dt = dYTpdt[i0];
            double dYi1dt = dYTpdt[i1];

            dYi0dt *= Wi0ByrhoM_;
            dYi1dt *= Wi1ByrhoM_;

            double ddNi0dtByVdYj0 = 0;
            double ddNi0dtByVdYj1 = 0;
            double ddNi0dtByVdYj2 = 0;
            double ddNi0dtByVdYj3 = 0;

            double ddNi1dtByVdYj0 = 0;
            double ddNi1dtByVdYj1 = 0;
            double ddNi1dtByVdYj2 = 0;
            double ddNi1dtByVdYj3 = 0;  

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = ddNdtByVdcTp[i0*(n_) + k];
                const double ddNi1dtByVdck = ddNdtByVdcTp[i1*(n_) + k];
                ddNi0dtByVdYj0 += ddNi0dtByVdck*ArrRes[k][0]; 
                ddNi0dtByVdYj1 += ddNi0dtByVdck*ArrRes[k][1]; 
                ddNi0dtByVdYj2 += ddNi0dtByVdck*ArrRes[k][2]; 
                ddNi0dtByVdYj3 += ddNi0dtByVdck*ArrRes[k][3];  
                ddNi1dtByVdYj0 += ddNi1dtByVdck*ArrRes[k][0]; 
                ddNi1dtByVdYj1 += ddNi1dtByVdck*ArrRes[k][1]; 
                ddNi1dtByVdYj2 += ddNi1dtByVdck*ArrRes[k][2]; 
                ddNi1dtByVdYj3 += ddNi1dtByVdck*ArrRes[k][3];  
            }
    
            double& ddYi0dtdYj0 = J[i0*(n_) + j+0];
            double& ddYi0dtdYj1 = J[i0*(n_) + j+1];
            double& ddYi0dtdYj2 = J[i0*(n_) + j+2];
            double& ddYi0dtdYj3 = J[i0*(n_) + j+3];
    
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj0_*dYi0dt;     
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj1_*dYi0dt;         
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj2_*dYi0dt;
            ddYi0dtdYj3 = Wi0ByrhoM_*ddNi0dtByVdYj3 + rhoMvj3_*dYi0dt;  

            double& ddYi1dtdYj0 = J[i1*(n_) + j+0];
            double& ddYi1dtdYj1 = J[i1*(n_) + j+1];
            double& ddYi1dtdYj2 = J[i1*(n_) + j+2];
            double& ddYi1dtdYj3 = J[i1*(n_) + j+3];
    
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj0_*dYi1dt;     
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj1_*dYi1dt;         
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj2_*dYi1dt;
            ddYi1dtdYj3 = Wi1ByrhoM_*ddNi1dtByVdYj3 + rhoMvj3_*dYi1dt; 
        }
        else if(remain==3)
        {
            unsigned int i0 = this->nSpecies-3;            
            unsigned int i1 = this->nSpecies-2;
            unsigned int i2 = this->nSpecies-1;

            const double Wi0ByrhoM_ = WiByrhoM[i0];
            const double Wi1ByrhoM_ = WiByrhoM[i1];
            const double Wi2ByrhoM_ = WiByrhoM[i2];

            double dYi0dt = dYTpdt[i0];
            double dYi1dt = dYTpdt[i1];
            double dYi2dt = dYTpdt[i2];

            dYi0dt *= Wi0ByrhoM_;
            dYi1dt *= Wi1ByrhoM_;
            dYi2dt *= Wi2ByrhoM_;

            double ddNi0dtByVdYj0 = 0;
            double ddNi0dtByVdYj1 = 0;
            double ddNi0dtByVdYj2 = 0;
            double ddNi0dtByVdYj3 = 0;

            double ddNi1dtByVdYj0 = 0;
            double ddNi1dtByVdYj1 = 0;
            double ddNi1dtByVdYj2 = 0;
            double ddNi1dtByVdYj3 = 0;  

            double ddNi2dtByVdYj0 = 0;
            double ddNi2dtByVdYj1 = 0;
            double ddNi2dtByVdYj2 = 0;
            double ddNi2dtByVdYj3 = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = ddNdtByVdcTp[i0*(n_) + k];
                const double ddNi1dtByVdck = ddNdtByVdcTp[i1*(n_) + k];
                const double ddNi2dtByVdck = ddNdtByVdcTp[i2*(n_) + k];
                ddNi0dtByVdYj0 += ddNi0dtByVdck*ArrRes[k][0]; 
                ddNi0dtByVdYj1 += ddNi0dtByVdck*ArrRes[k][1]; 
                ddNi0dtByVdYj2 += ddNi0dtByVdck*ArrRes[k][2]; 
                ddNi0dtByVdYj3 += ddNi0dtByVdck*ArrRes[k][3]; 
                ddNi1dtByVdYj0 += ddNi1dtByVdck*ArrRes[k][0]; 
                ddNi1dtByVdYj1 += ddNi1dtByVdck*ArrRes[k][1]; 
                ddNi1dtByVdYj2 += ddNi1dtByVdck*ArrRes[k][2]; 
                ddNi1dtByVdYj3 += ddNi1dtByVdck*ArrRes[k][3];   
                ddNi2dtByVdYj0 += ddNi2dtByVdck*ArrRes[k][0]; 
                ddNi2dtByVdYj1 += ddNi2dtByVdck*ArrRes[k][1]; 
                ddNi2dtByVdYj2 += ddNi2dtByVdck*ArrRes[k][2]; 
                ddNi2dtByVdYj3 += ddNi2dtByVdck*ArrRes[k][3];                
            }
            double& ddYi0dtdYj0 = J[i0*(n_) + j+0];
            double& ddYi0dtdYj1 = J[i0*(n_) + j+1];
            double& ddYi0dtdYj2 = J[i0*(n_) + j+2];
            double& ddYi0dtdYj3 = J[i0*(n_) + j+3];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj0_*dYi0dt;     
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj1_*dYi0dt;         
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj2_*dYi0dt;
            ddYi0dtdYj3 = Wi0ByrhoM_*ddNi0dtByVdYj3 + rhoMvj3_*dYi0dt;  
            double& ddYi1dtdYj0 = J[i1*(n_) + j+0];
            double& ddYi1dtdYj1 = J[i1*(n_) + j+1];
            double& ddYi1dtdYj2 = J[i1*(n_) + j+2];
            double& ddYi1dtdYj3 = J[i1*(n_) + j+3];
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj0_*dYi1dt; 
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj1_*dYi1dt; 
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj2_*dYi1dt;
            ddYi1dtdYj3 = Wi1ByrhoM_*ddNi1dtByVdYj3 + rhoMvj3_*dYi1dt; 
            double& ddYi2dtdYj0 = J[i2*(n_) + j+0];
            double& ddYi2dtdYj1 = J[i2*(n_) + j+1];
            double& ddYi2dtdYj2 = J[i2*(n_) + j+2];
            double& ddYi2dtdYj3 = J[i2*(n_) + j+3];
            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj0_*dYi2dt; 
            ddYi2dtdYj1 = Wi2ByrhoM_*ddNi2dtByVdYj1 + rhoMvj1_*dYi2dt; 
            ddYi2dtdYj2 = Wi2ByrhoM_*ddNi2dtByVdYj2 + rhoMvj2_*dYi2dt;
            ddYi2dtdYj3 = Wi2ByrhoM_*ddNi2dtByVdYj3 + rhoMvj3_*dYi2dt;             
        }        
    }
    if(remain==1)
    {
        unsigned int j = this->nSpecies-1;
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj_ = rhoMvj[j];
            ArrRes[i][0] = rhoMByWi*((i == j) - rhoMvj_*YTp[i]);
        }

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj_ = rhoMvj[j];
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;
            double ddNidtByVdYj = 0;
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj += ddNidtByVdck*ArrRes[k][0];          
            }
            double& ddYidtdYj = J[i*(n_) + j];
            ddYidtdYj = WiByrhoM_*ddNidtByVdYj + rhoMvj_*dYidt;
        }
    }
    else if(remain==2)
    {
        unsigned int j0 = this->nSpecies-2;
        unsigned int j1 = this->nSpecies-1;

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1]; 
            ArrRes[i][0] = rhoMByWi*((i == j0) - rhoMvj0_*YTp[i]);
            ArrRes[i][1] = rhoMByWi*((i == j1) - rhoMvj1_*YTp[i]);          
        }

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;
            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];        
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];  
            }
            double& ddYidtdYj0 = J[i*(n_) + j0];
            double& ddYidtdYj1 = J[i*(n_) + j1];   
            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;            
        }
    }    
    else if(remain==3)
    {
        unsigned int j0 = this->nSpecies-3;
        unsigned int j1 = this->nSpecies-2;
        unsigned int j2 = this->nSpecies-1;

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];
            ArrRes[i][0] = rhoMByWi*((i == j0) - rhoMvj0_*YTp[i]);
            ArrRes[i][1] = rhoMByWi*((i == j1) - rhoMvj1_*YTp[i]);
            ArrRes[i][2] = rhoMByWi*((i == j2) - rhoMvj2_*YTp[i]);
        }
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];

            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;

            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;
            double ddNidtByVdYj2 = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];
                ddNidtByVdYj2 += ddNidtByVdck*ArrRes[k][2];
            }

            double& ddYidtdYj0 = J[i*(n_) + j0];
            double& ddYidtdYj1 = J[i*(n_) + j1];
            double& ddYidtdYj2 = J[i*(n_) + j2];

            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;
            ddYidtdYj2 = WiByrhoM_*ddNidtByVdYj2 + rhoMvj2_*dYidt;
        }
    }        
}

void 
OptReaction::ddYdtdY_Vec1
(
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* const* __restrict__ ArrRes = this->ArrPtr.data();

    unsigned int remain = this->nSpecies%4;

    for(unsigned int j = 0; j < this->nSpecies - remain; j = j + 4)
    {
        const double rhoMvj0_ = rhoMvj[j+0];     
        const double rhoMvj1_ = rhoMvj[j+1]; 
        const double rhoMvj2_ = rhoMvj[j+2]; 
        const double rhoMvj3_ = rhoMvj[j+3]; 
        __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies-remain; i=i+4)
        {

            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*this->invW[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*this->invW[i+3]*YTp[i+3];

            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvj_));
        }
        if(remain==1)
        {
            unsigned int i = this->nSpecies-1;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;
        }
        else if(remain==2)
        {
            unsigned int i = this->nSpecies-2;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;   

            ArrRes[i+1][0] = rhoMByWiYTPi1*rhoMvj0_;
            ArrRes[i+1][1] = rhoMByWiYTPi1*rhoMvj1_;
            ArrRes[i+1][2] = rhoMByWiYTPi1*rhoMvj2_;
            ArrRes[i+1][3] = rhoMByWiYTPi1*rhoMvj3_;          
        }
        else if(remain==3)
        {
            unsigned int i = this->nSpecies-3;
            const double rhoMByWiYTPi0 = -rhoM*this->invW[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*this->invW[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*this->invW[i+2]*YTp[i+2];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;   

            ArrRes[i+1][0] = rhoMByWiYTPi1*rhoMvj0_;
            ArrRes[i+1][1] = rhoMByWiYTPi1*rhoMvj1_;
            ArrRes[i+1][2] = rhoMByWiYTPi1*rhoMvj2_;
            ArrRes[i+1][3] = rhoMByWiYTPi1*rhoMvj3_;  

            ArrRes[i+2][0] = rhoMByWiYTPi2*rhoMvj0_;
            ArrRes[i+2][1] = rhoMByWiYTPi2*rhoMvj1_;
            ArrRes[i+2][2] = rhoMByWiYTPi2*rhoMvj2_;
            ArrRes[i+2][3] = rhoMByWiYTPi2*rhoMvj3_;  
        }
        ArrRes[j+0][0] += rhoM*this->invW[j+0];
        ArrRes[j+1][1] += rhoM*this->invW[j+1];
        ArrRes[j+2][2] += rhoM*this->invW[j+2];
        ArrRes[j+3][3] += rhoM*this->invW[j+3];  

        for(unsigned int i=0; i<this->nSpecies - remain; i=i+4)
        {
            const double Wi0ByrhoM_ = WiByrhoM[i+0];
            const double Wi1ByrhoM_ = WiByrhoM[i+1];
            const double Wi2ByrhoM_ = WiByrhoM[i+2];
            const double Wi3ByrhoM_ = WiByrhoM[i+3];

            const double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            const double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            const double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            const double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;

            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();

            for (unsigned int k=0; k<this->nSpecies-remain; k=k+4)
            {

                const double ddNi0dtByVdck0 = ddNdtByVdcTp[(i+0)*(n_)+k+0];
                const double ddNi1dtByVdck0 = ddNdtByVdcTp[(i+1)*(n_)+k+0];
                const double ddNi2dtByVdck0 = ddNdtByVdcTp[(i+2)*(n_)+k+0];
                const double ddNi3dtByVdck0 = ddNdtByVdcTp[(i+3)*(n_)+k+0];

                const double ddNi0dtByVdck1 = ddNdtByVdcTp[(i+0)*(n_)+k+1];
                const double ddNi1dtByVdck1 = ddNdtByVdcTp[(i+1)*(n_)+k+1];
                const double ddNi2dtByVdck1 = ddNdtByVdcTp[(i+2)*(n_)+k+1];
                const double ddNi3dtByVdck1 = ddNdtByVdcTp[(i+3)*(n_)+k+1];

                const double ddNi0dtByVdck2 = ddNdtByVdcTp[(i+0)*(n_)+k+2];
                const double ddNi1dtByVdck2 = ddNdtByVdcTp[(i+1)*(n_)+k+2];
                const double ddNi2dtByVdck2 = ddNdtByVdcTp[(i+2)*(n_)+k+2];
                const double ddNi3dtByVdck2 = ddNdtByVdcTp[(i+3)*(n_)+k+2];

                const double ddNi0dtByVdck3 = ddNdtByVdcTp[(i+0)*(n_)+k+3];
                const double ddNi1dtByVdck3 = ddNdtByVdcTp[(i+1)*(n_)+k+3];
                const double ddNi2dtByVdck3 = ddNdtByVdcTp[(i+2)*(n_)+k+3];
                const double ddNi3dtByVdck3 = ddNdtByVdcTp[(i+3)*(n_)+k+3];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);
                __m256d dCk3dYj = _mm256_loadu_pd(&ArrRes[k+3][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
                
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3),dCk3dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3),dCk3dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck3),dCk3dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck3),dCk3dYj,ddNi3dtByVdYj); 
            }
            if(remain==3)
            {
                unsigned int k = this->nSpecies-3;
                const double ddNi0dtByVdck0 = ddNdtByVdcTp[(i+0)*(n_)+k+0];
                const double ddNi1dtByVdck0 = ddNdtByVdcTp[(i+1)*(n_)+k+0];
                const double ddNi2dtByVdck0 = ddNdtByVdcTp[(i+2)*(n_)+k+0];
                const double ddNi3dtByVdck0 = ddNdtByVdcTp[(i+3)*(n_)+k+0];

                const double ddNi0dtByVdck1 = ddNdtByVdcTp[(i+0)*(n_)+k+1];
                const double ddNi1dtByVdck1 = ddNdtByVdcTp[(i+1)*(n_)+k+1];
                const double ddNi2dtByVdck1 = ddNdtByVdcTp[(i+2)*(n_)+k+1];
                const double ddNi3dtByVdck1 = ddNdtByVdcTp[(i+3)*(n_)+k+1];

                const double ddNi0dtByVdck2 = ddNdtByVdcTp[(i+0)*(n_)+k+2];
                const double ddNi1dtByVdck2 = ddNdtByVdcTp[(i+1)*(n_)+k+2];
                const double ddNi2dtByVdck2 = ddNdtByVdcTp[(i+2)*(n_)+k+2];
                const double ddNi3dtByVdck2 = ddNdtByVdcTp[(i+3)*(n_)+k+2];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
            }
            else if(remain==2)
            {
                unsigned int k = this->nSpecies-2;
                const double ddNi0dtByVdck0 = ddNdtByVdcTp[(i+0)*(n_)+k+0];
                const double ddNi1dtByVdck0 = ddNdtByVdcTp[(i+1)*(n_)+k+0];
                const double ddNi2dtByVdck0 = ddNdtByVdcTp[(i+2)*(n_)+k+0];
                const double ddNi3dtByVdck0 = ddNdtByVdcTp[(i+3)*(n_)+k+0];

                const double ddNi0dtByVdck1 = ddNdtByVdcTp[(i+0)*(n_)+k+1];
                const double ddNi1dtByVdck1 = ddNdtByVdcTp[(i+1)*(n_)+k+1];
                const double ddNi2dtByVdck1 = ddNdtByVdcTp[(i+2)*(n_)+k+1];
                const double ddNi3dtByVdck1 = ddNdtByVdcTp[(i+3)*(n_)+k+1];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj); 
            }
            else if(remain==1)
            {
                unsigned int k = this->nSpecies-1;
                const double ddNi0dtByVdck0 = ddNdtByVdcTp[(i+0)*(n_)+k+0];
                const double ddNi1dtByVdck0 = ddNdtByVdcTp[(i+1)*(n_)+k+0];
                const double ddNi2dtByVdck0 = ddNdtByVdcTp[(i+2)*(n_)+k+0];
                const double ddNi3dtByVdck0 = ddNdtByVdcTp[(i+3)*(n_)+k+0];
                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);
            }
            __m256d WiByrhoM_0 = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidtv0 = _mm256_set1_pd(dYi0dt);
            __m256d ddYi0dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv0);
            ddYi0dtdYj = _mm256_fmadd_pd(WiByrhoM_0,ddNi0dtByVdYj,ddYi0dtdYj);
            __m256d WiByrhoM_1 = _mm256_set1_pd(Wi1ByrhoM_);
            __m256d dYidtv1 = _mm256_set1_pd(dYi1dt);
            __m256d ddYi1dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv1);
            ddYi1dtdYj = _mm256_fmadd_pd(WiByrhoM_1,ddNi1dtByVdYj,ddYi1dtdYj);
            __m256d WiByrhoM_2 = _mm256_set1_pd(Wi2ByrhoM_);
            __m256d dYidtv2 = _mm256_set1_pd(dYi2dt);
            __m256d ddYi2dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv2);
            ddYi2dtdYj = _mm256_fmadd_pd(WiByrhoM_2,ddNi2dtByVdYj,ddYi2dtdYj);
            __m256d WiByrhoM_3 = _mm256_set1_pd(Wi3ByrhoM_);
            __m256d dYidtv3 = _mm256_set1_pd(dYi3dt);
            __m256d ddYi3dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv3);
            ddYi3dtdYj = _mm256_fmadd_pd(WiByrhoM_3,ddNi3dtByVdYj,ddYi3dtdYj);
            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
        }     
        
        if(remain==1)
        {
            unsigned int i = this->nSpecies-1;
            const double WiByrhoM_ = WiByrhoM[i];
    
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;
    
            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;
            double ddNidtByVdYj2 = 0;
            double ddNidtByVdYj3 = 0;
    
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];
                ddNidtByVdYj2 += ddNidtByVdck*ArrRes[k][2];
                ddNidtByVdYj3 += ddNidtByVdck*ArrRes[k][3];
            }
            double& ddYidtdYj0 = J[i*(n_) + j+0];
            double& ddYidtdYj1 = J[i*(n_) + j+1];
            double& ddYidtdYj2 = J[i*(n_) + j+2];
            double& ddYidtdYj3 = J[i*(n_) + j+3];
            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;     
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;         
            ddYidtdYj2 = WiByrhoM_*ddNidtByVdYj2 + rhoMvj2_*dYidt;
            ddYidtdYj3 = WiByrhoM_*ddNidtByVdYj3 + rhoMvj3_*dYidt;  
        }
        else if(remain==2)
        {
            unsigned int i0 = this->nSpecies-2;
            unsigned int i1 = this->nSpecies-1;

            const double Wi0ByrhoM_ = WiByrhoM[i0];
            const double Wi1ByrhoM_ = WiByrhoM[i1];

            double dYi0dt = dYTpdt[i0];
            double dYi1dt = dYTpdt[i1];

            dYi0dt *= Wi0ByrhoM_;
            dYi1dt *= Wi1ByrhoM_;

            double ddNi0dtByVdYj0 = 0;
            double ddNi0dtByVdYj1 = 0;
            double ddNi0dtByVdYj2 = 0;
            double ddNi0dtByVdYj3 = 0;

            double ddNi1dtByVdYj0 = 0;
            double ddNi1dtByVdYj1 = 0;
            double ddNi1dtByVdYj2 = 0;
            double ddNi1dtByVdYj3 = 0;  

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = ddNdtByVdcTp[i0*(n_) + k];
                const double ddNi1dtByVdck = ddNdtByVdcTp[i1*(n_) + k];
                ddNi0dtByVdYj0 += ddNi0dtByVdck*ArrRes[k][0]; 
                ddNi0dtByVdYj1 += ddNi0dtByVdck*ArrRes[k][1]; 
                ddNi0dtByVdYj2 += ddNi0dtByVdck*ArrRes[k][2]; 
                ddNi0dtByVdYj3 += ddNi0dtByVdck*ArrRes[k][3]; 
                ddNi1dtByVdYj0 += ddNi1dtByVdck*ArrRes[k][0]; 
                ddNi1dtByVdYj1 += ddNi1dtByVdck*ArrRes[k][1]; 
                ddNi1dtByVdYj2 += ddNi1dtByVdck*ArrRes[k][2]; 
                ddNi1dtByVdYj3 += ddNi1dtByVdck*ArrRes[k][3];  
            }
    
            double& ddYi0dtdYj0 = J[i0*(n_) + j+0];
            double& ddYi0dtdYj1 = J[i0*(n_) + j+1];
            double& ddYi0dtdYj2 = J[i0*(n_) + j+2];
            double& ddYi0dtdYj3 = J[i0*(n_) + j+3];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj0_*dYi0dt;     
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj1_*dYi0dt;         
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj2_*dYi0dt;
            ddYi0dtdYj3 = Wi0ByrhoM_*ddNi0dtByVdYj3 + rhoMvj3_*dYi0dt;  
            double& ddYi1dtdYj0 = J[i1*(n_) + j+0];
            double& ddYi1dtdYj1 = J[i1*(n_) + j+1];
            double& ddYi1dtdYj2 = J[i1*(n_) + j+2];
            double& ddYi1dtdYj3 = J[i1*(n_) + j+3];
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj0_*dYi1dt;     
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj1_*dYi1dt;         
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj2_*dYi1dt;
            ddYi1dtdYj3 = Wi1ByrhoM_*ddNi1dtByVdYj3 + rhoMvj3_*dYi1dt; 
        }
        else if(remain==3)
        {
            unsigned int i0 = this->nSpecies-3;            
            unsigned int i1 = this->nSpecies-2;
            unsigned int i2 = this->nSpecies-1;
            const double Wi0ByrhoM_ = WiByrhoM[i0];
            const double Wi1ByrhoM_ = WiByrhoM[i1];
            const double Wi2ByrhoM_ = WiByrhoM[i2];
            double dYi0dt = dYTpdt[i0];
            double dYi1dt = dYTpdt[i1];
            double dYi2dt = dYTpdt[i2];

            dYi0dt *= Wi0ByrhoM_;
            dYi1dt *= Wi1ByrhoM_;
            dYi2dt *= Wi2ByrhoM_;

            double ddNi0dtByVdYj0 = 0;
            double ddNi0dtByVdYj1 = 0;
            double ddNi0dtByVdYj2 = 0;
            double ddNi0dtByVdYj3 = 0;

            double ddNi1dtByVdYj0 = 0;
            double ddNi1dtByVdYj1 = 0;
            double ddNi1dtByVdYj2 = 0;
            double ddNi1dtByVdYj3 = 0;  

            double ddNi2dtByVdYj0 = 0;
            double ddNi2dtByVdYj1 = 0;
            double ddNi2dtByVdYj2 = 0;
            double ddNi2dtByVdYj3 = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = ddNdtByVdcTp[i0*(n_) + k];
                const double ddNi1dtByVdck = ddNdtByVdcTp[i1*(n_) + k];
                const double ddNi2dtByVdck = ddNdtByVdcTp[i2*(n_) + k];

                ddNi0dtByVdYj0 += ddNi0dtByVdck*ArrRes[k][0]; 
                ddNi0dtByVdYj1 += ddNi0dtByVdck*ArrRes[k][1]; 
                ddNi0dtByVdYj2 += ddNi0dtByVdck*ArrRes[k][2]; 
                ddNi0dtByVdYj3 += ddNi0dtByVdck*ArrRes[k][3]; 

                ddNi1dtByVdYj0 += ddNi1dtByVdck*ArrRes[k][0]; 
                ddNi1dtByVdYj1 += ddNi1dtByVdck*ArrRes[k][1]; 
                ddNi1dtByVdYj2 += ddNi1dtByVdck*ArrRes[k][2]; 
                ddNi1dtByVdYj3 += ddNi1dtByVdck*ArrRes[k][3];   
                
                ddNi2dtByVdYj0 += ddNi2dtByVdck*ArrRes[k][0]; 
                ddNi2dtByVdYj1 += ddNi2dtByVdck*ArrRes[k][1]; 
                ddNi2dtByVdYj2 += ddNi2dtByVdck*ArrRes[k][2]; 
                ddNi2dtByVdYj3 += ddNi2dtByVdck*ArrRes[k][3];                
            }
    
            double& ddYi0dtdYj0 = J[i0*(n_) + j+0];
            double& ddYi0dtdYj1 = J[i0*(n_) + j+1];
            double& ddYi0dtdYj2 = J[i0*(n_) + j+2];
            double& ddYi0dtdYj3 = J[i0*(n_) + j+3];
    
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj0_*dYi0dt;     
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj1_*dYi0dt;         
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj2_*dYi0dt;
            ddYi0dtdYj3 = Wi0ByrhoM_*ddNi0dtByVdYj3 + rhoMvj3_*dYi0dt;  

            double& ddYi1dtdYj0 = J[i1*(n_) + j+0];
            double& ddYi1dtdYj1 = J[i1*(n_) + j+1];
            double& ddYi1dtdYj2 = J[i1*(n_) + j+2];
            double& ddYi1dtdYj3 = J[i1*(n_) + j+3];
    
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj0_*dYi1dt; 
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj1_*dYi1dt; 
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj2_*dYi1dt;
            ddYi1dtdYj3 = Wi1ByrhoM_*ddNi1dtByVdYj3 + rhoMvj3_*dYi1dt; 

            double& ddYi2dtdYj0 = J[i2*(n_) + j+0];
            double& ddYi2dtdYj1 = J[i2*(n_) + j+1];
            double& ddYi2dtdYj2 = J[i2*(n_) + j+2];
            double& ddYi2dtdYj3 = J[i2*(n_) + j+3];
    
            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj0_*dYi2dt; 
            ddYi2dtdYj1 = Wi2ByrhoM_*ddNi2dtByVdYj1 + rhoMvj1_*dYi2dt; 
            ddYi2dtdYj2 = Wi2ByrhoM_*ddNi2dtByVdYj2 + rhoMvj2_*dYi2dt;
            ddYi2dtdYj3 = Wi2ByrhoM_*ddNi2dtByVdYj3 + rhoMvj3_*dYi2dt;             
        }        
    }
    if(remain==1)
    {
        unsigned int j = this->nSpecies-1;
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj_ = rhoMvj[j];
            ArrRes[i][0] = rhoMByWi*((i == j) - rhoMvj_*YTp[i]);
        }
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj_ = rhoMvj[j];
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;
            double ddNidtByVdYj = 0;
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj += ddNidtByVdck*ArrRes[k][0];          
            }
            double& ddYidtdYj = J[i*(n_) + j];
            ddYidtdYj = WiByrhoM_*ddNidtByVdYj + rhoMvj_*dYidt;
        }
    }
    else if(remain==2)
    {
        unsigned int j0 = this->nSpecies-2;
        unsigned int j1 = this->nSpecies-1;

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];  
            ArrRes[i][0] = rhoMByWi*((i == j0) - rhoMvj0_*YTp[i]);
            ArrRes[i][1] = rhoMByWi*((i == j1) - rhoMvj1_*YTp[i]);          
        }
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];

            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;

            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];        
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];  
            }

            double& ddYidtdYj0 = J[i*(n_) + j0];
            double& ddYidtdYj1 = J[i*(n_) + j1];   

            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;            
        }
    }    
    else if(remain==3)
    {
        unsigned int j0 = this->nSpecies-3;
        unsigned int j1 = this->nSpecies-2;
        unsigned int j2 = this->nSpecies-1;

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];
            ArrRes[i][0] = rhoMByWi*((i == j0) - rhoMvj0_*YTp[i]);
            ArrRes[i][1] = rhoMByWi*((i == j1) - rhoMvj1_*YTp[i]);
            ArrRes[i][2] = rhoMByWi*((i == j2) - rhoMvj2_*YTp[i]);
        }
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];

            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;

            double ddNidtByVdYj0 = 0;
            double ddNidtByVdYj1 = 0;
            double ddNidtByVdYj2 = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                ddNidtByVdYj0 += ddNidtByVdck*ArrRes[k][0];
                ddNidtByVdYj1 += ddNidtByVdck*ArrRes[k][1];
                ddNidtByVdYj2 += ddNidtByVdck*ArrRes[k][2];
            }

            double& ddYidtdYj0 = J[i*(n_) + j0];
            double& ddYidtdYj1 = J[i*(n_) + j1];
            double& ddYidtdYj2 = J[i*(n_) + j2];

            ddYidtdYj0 = WiByrhoM_*ddNidtByVdYj0 + rhoMvj0_*dYidt;
            ddYidtdYj1 = WiByrhoM_*ddNidtByVdYj1 + rhoMvj1_*dYidt;
            ddYidtdYj2 = WiByrhoM_*ddNidtByVdYj2 + rhoMvj2_*dYidt;
        }
    }        
}

void 
OptReaction::ddYdtdY_Vec1_0
(
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* __restrict__ invWPtr = &invW[0];
    double* const* __restrict__ ArrRes = this->ArrPtr.data();
    for(unsigned int j = 0; j < this->nSpecies; j = j + 4)
    {
        __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies; i=i+4)
        {
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*invWPtr[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*invWPtr[i+3]*YTp[i+3];
            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvj_));
        }
        ArrRes[j+0][0] += rhoM*invWPtr[j+0];
        ArrRes[j+1][1] += rhoM*invWPtr[j+1];
        ArrRes[j+2][2] += rhoM*invWPtr[j+2];
        ArrRes[j+3][3] += rhoM*invWPtr[j+3];  
        for(unsigned int i=0; i<this->nSpecies; i=i+4)
        {
            const double Wi0ByrhoM_ = WiByrhoM[i+0];
            const double Wi1ByrhoM_ = WiByrhoM[i+1];
            const double Wi2ByrhoM_ = WiByrhoM[i+2];
            const double Wi3ByrhoM_ = WiByrhoM[i+3];
            const double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            const double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            const double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            const double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;
            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();
            const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i+0)*n_];
            const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i+1)*n_];
            const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i+2)*n_];
            const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i+3)*n_];
            for (unsigned int k=0; k<this->nSpecies; k=k+4)
            {
                const double ddNi0dtByVdck0 = JcRowi0[k+0];
                const double ddNi1dtByVdck0 = JcRowi1[k+0];
                const double ddNi2dtByVdck0 = JcRowi2[k+0];
                const double ddNi3dtByVdck0 = JcRowi3[k+0];
                const double ddNi0dtByVdck1 = JcRowi0[k+1];
                const double ddNi1dtByVdck1 = JcRowi1[k+1];
                const double ddNi2dtByVdck1 = JcRowi2[k+1];
                const double ddNi3dtByVdck1 = JcRowi3[k+1];
                const double ddNi0dtByVdck2 = JcRowi0[k+2];
                const double ddNi1dtByVdck2 = JcRowi1[k+2];
                const double ddNi2dtByVdck2 = JcRowi2[k+2];
                const double ddNi3dtByVdck2 = JcRowi3[k+2];
                const double ddNi0dtByVdck3 = JcRowi0[k+3];
                const double ddNi1dtByVdck3 = JcRowi1[k+3];
                const double ddNi2dtByVdck3 = JcRowi2[k+3];
                const double ddNi3dtByVdck3 = JcRowi3[k+3];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);
                __m256d dCk3dYj = _mm256_loadu_pd(&ArrRes[k+3][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
                
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3),dCk3dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3),dCk3dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck3),dCk3dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck3),dCk3dYj,ddNi3dtByVdYj); 
            }
             __m256d WiByrhoM_0 = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidtv0 = _mm256_set1_pd(dYi0dt);
            __m256d ddYi0dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv0);
            ddYi0dtdYj = _mm256_fmadd_pd(WiByrhoM_0,ddNi0dtByVdYj,ddYi0dtdYj);
            __m256d WiByrhoM_1 = _mm256_set1_pd(Wi1ByrhoM_);
            __m256d dYidtv1 = _mm256_set1_pd(dYi1dt);
            __m256d ddYi1dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv1);
            ddYi1dtdYj = _mm256_fmadd_pd(WiByrhoM_1,ddNi1dtByVdYj,ddYi1dtdYj);
            __m256d WiByrhoM_2 = _mm256_set1_pd(Wi2ByrhoM_);
            __m256d dYidtv2 = _mm256_set1_pd(dYi2dt);
            __m256d ddYi2dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv2);
            ddYi2dtdYj = _mm256_fmadd_pd(WiByrhoM_2,ddNi2dtByVdYj,ddYi2dtdYj);
            __m256d WiByrhoM_3 = _mm256_set1_pd(Wi3ByrhoM_);
            __m256d dYidtv3 = _mm256_set1_pd(dYi3dt);
            __m256d ddYi3dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv3);
            ddYi3dtdYj = _mm256_fmadd_pd(WiByrhoM_3,ddNi3dtByVdYj,ddYi3dtdYj);
            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
        }     
    }     
}


void 
OptReaction::ddYdtdY_Vec1_1
(
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* __restrict__ invWPtr = &invW[0];
    double* const* __restrict__ ArrRes = this->ArrPtr.data();
    for(unsigned int j = 0; j < this->nSpecies - 1; j = j + 4)
    {
        __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies-1; i=i+4)
        {
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*invWPtr[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*invWPtr[i+3]*YTp[i+3];
            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvj_));
        }
        unsigned int ii = this->nSpecies-1;
        const double rhoMByWiYTPi0 = -rhoM*invWPtr[ii+0]*YTp[ii+0];
        __m256d result1 = _mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_);
        _mm256_storeu_pd(&ArrRes[ii+0][0],result1);
        ArrRes[j+0][0] += rhoM*invWPtr[j+0];
        ArrRes[j+1][1] += rhoM*invWPtr[j+1];
        ArrRes[j+2][2] += rhoM*invWPtr[j+2];
        ArrRes[j+3][3] += rhoM*invWPtr[j+3];  
        for(unsigned int i=0; i<this->nSpecies - 1; i=i+4)
        {
            const double Wi0ByrhoM_ = WiByrhoM[i+0];
            const double Wi1ByrhoM_ = WiByrhoM[i+1];
            const double Wi2ByrhoM_ = WiByrhoM[i+2];
            const double Wi3ByrhoM_ = WiByrhoM[i+3];
            const double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            const double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            const double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            const double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;
            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();
            const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i+0)*(n_)];
            const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i+1)*(n_)];
            const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i+2)*(n_)];
            const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i+3)*(n_)];
            for (unsigned int k=0; k<this->nSpecies-1; k=k+4)
            {
                const double ddNi0dtByVdck0 = JcRowi0[k+0];
                const double ddNi1dtByVdck0 = JcRowi1[k+0];
                const double ddNi2dtByVdck0 = JcRowi2[k+0];
                const double ddNi3dtByVdck0 = JcRowi3[k+0];
                const double ddNi0dtByVdck1 = JcRowi0[k+1];
                const double ddNi1dtByVdck1 = JcRowi1[k+1];
                const double ddNi2dtByVdck1 = JcRowi2[k+1];
                const double ddNi3dtByVdck1 = JcRowi3[k+1];
                const double ddNi0dtByVdck2 = JcRowi0[k+2];
                const double ddNi1dtByVdck2 = JcRowi1[k+2];
                const double ddNi2dtByVdck2 = JcRowi2[k+2];
                const double ddNi3dtByVdck2 = JcRowi3[k+2];
                const double ddNi0dtByVdck3 = JcRowi0[k+3];
                const double ddNi1dtByVdck3 = JcRowi1[k+3];
                const double ddNi2dtByVdck3 = JcRowi2[k+3];
                const double ddNi3dtByVdck3 = JcRowi3[k+3];
                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);
                __m256d dCk3dYj = _mm256_loadu_pd(&ArrRes[k+3][0]);
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3),dCk3dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3),dCk3dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck3),dCk3dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck3),dCk3dYj,ddNi3dtByVdYj); 
            }
            unsigned int k = this->nSpecies-1;
            const double ddNi0dtByVdck0 = JcRowi0[k+0];
            const double ddNi1dtByVdck0 = JcRowi1[k+0];
            const double ddNi2dtByVdck0 = JcRowi2[k+0];
            const double ddNi3dtByVdck0 = JcRowi3[k+0];
            __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
            ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
            ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
            ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
            ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);
            __m256d WiByrhoM_0 = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidtv0 = _mm256_set1_pd(dYi0dt);
            __m256d ddYi0dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv0);
            ddYi0dtdYj = _mm256_fmadd_pd(WiByrhoM_0,ddNi0dtByVdYj,ddYi0dtdYj);
            __m256d WiByrhoM_1 = _mm256_set1_pd(Wi1ByrhoM_);
            __m256d dYidtv1 = _mm256_set1_pd(dYi1dt);
            __m256d ddYi1dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv1);
            ddYi1dtdYj = _mm256_fmadd_pd(WiByrhoM_1,ddNi1dtByVdYj,ddYi1dtdYj);
            __m256d WiByrhoM_2 = _mm256_set1_pd(Wi2ByrhoM_);
            __m256d dYidtv2 = _mm256_set1_pd(dYi2dt);
            __m256d ddYi2dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv2);
            ddYi2dtdYj = _mm256_fmadd_pd(WiByrhoM_2,ddNi2dtByVdYj,ddYi2dtdYj);
            __m256d WiByrhoM_3 = _mm256_set1_pd(Wi3ByrhoM_);
            __m256d dYidtv3 = _mm256_set1_pd(dYi3dt);
            __m256d ddYi3dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv3);
            ddYi3dtdYj = _mm256_fmadd_pd(WiByrhoM_3,ddNi3dtByVdYj,ddYi3dtdYj);

            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
        }     
        const double* __restrict__ JcRowii = &ddNdtByVdcTp[ii*n_];
        const double WiByrhoM_ = WiByrhoM[ii];
        double dYidt = dYTpdt[ii];
        dYidt *= WiByrhoM_;
        __m256d ddNidtByVdYjv = _mm256_setzero_pd();
        for (unsigned int k=0; k<this->nSpecies-1; k=k+4)
        {
            __m256d Arrk0 = _mm256_loadu_pd(&ArrRes[k+0][0]);
            ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(JcRowii[k+0]),Arrk0,ddNidtByVdYjv);
            __m256d Arrk1 = _mm256_loadu_pd(&ArrRes[k+1][0]);
            ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(JcRowii[k+1]),Arrk1,ddNidtByVdYjv);
            __m256d Arrk2 = _mm256_loadu_pd(&ArrRes[k+2][0]);
            ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(JcRowii[k+2]),Arrk2,ddNidtByVdYjv);
            __m256d Arrk3 = _mm256_loadu_pd(&ArrRes[k+3][0]);
            ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(JcRowii[k+3]),Arrk3,ddNidtByVdYjv);  
        }
        unsigned int k0 = nSpecies-1;
        __m256d Arrk3 = _mm256_loadu_pd(&ArrRes[k0][0]);
        ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(JcRowii[k0]),Arrk3,ddNidtByVdYjv);          
        __m256d result = _mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYidt));
        result = _mm256_fmadd_pd(ddNidtByVdYjv,_mm256_set1_pd(WiByrhoM_),result);
        _mm256_storeu_pd(&J[ii*(n_) + j+0],result);
    }
    unsigned int j = this->nSpecies-1;
    const double rhoMvj_ = rhoMvj[j];    
    __m256d rhoMvjv = _mm256_set1_pd(-rhoMvj_);
    __m256d rhoMv = _mm256_set1_pd(rhoM);

    for(unsigned int i=0; i<this->nSpecies -1; i=i+4)
    {
        __m256d rhoMByWiv = _mm256_mul_pd(rhoMv,_mm256_loadu_pd(&invWPtr[i]));
        __m256d YTpv = _mm256_loadu_pd(&YTp[i+0]);
        __m256d result = _mm256_mul_pd(rhoMByWiv,_mm256_mul_pd(YTpv,rhoMvjv));
        _mm256_storeu_pd(&ArrRes[0][i+0],result);  
    }
    ArrRes[0][j] = rhoM*invWPtr[j]*(1-rhoMvj[j]*YTp[j]);

    for(unsigned int i=0; i<this->nSpecies; i++)
    {
        const double WiByrhoM_0 = WiByrhoM[i+0];
        double dYi0dt = WiByrhoM_0*dYTpdt[i+0];
        double ddNi0dtByVdYj = 0;
        __m256d sum = _mm256_setzero_pd();
        const double* __restrict__ JcRowi = &ddNdtByVdcTp[i*n_];

        for (unsigned int k=0; k<this->nSpecies-1; k=k+4)
        {
            __m256d a = _mm256_loadu_pd(&JcRowi[k]);
            __m256d b = _mm256_loadu_pd(&ArrRes[0][k]);
            sum = _mm256_fmadd_pd(a, b, sum); 
        }
        __m256d tmp = _mm256_permute2f128_pd(sum, sum, 0x01);
        __m256d sum1 = _mm256_add_pd(sum, tmp);
        __m128d lo = _mm256_castpd256_pd128(sum1);            
        __m128d hi = _mm_unpackhi_pd(lo, lo);                 
        __m128d sum2 = _mm_add_pd(lo, hi);
        double dot =  _mm_cvtsd_f64(sum2);
        ddNi0dtByVdYj += dot;
        unsigned int k = nSpecies-1;
        ddNi0dtByVdYj += JcRowi[k]*ArrRes[0][k];
        J[(i+0)*(n_) + j] = WiByrhoM_0*ddNi0dtByVdYj + rhoMvj_*dYi0dt;
    }
}

void 
OptReaction::ddYdtdY_Vec1_2
(
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* __restrict__ invWPtr = &invW[0];
    double* const* __restrict__ ArrRes = this->ArrPtr.data();
    unsigned int remain = 2;
    for(unsigned int j = 0; j < this->nSpecies - remain; j = j + 4)
    {
        const double rhoMvj0_ = rhoMvj[j+0];     
        const double rhoMvj1_ = rhoMvj[j+1]; 
        const double rhoMvj2_ = rhoMvj[j+2]; 
        const double rhoMvj3_ = rhoMvj[j+3]; 
        __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies-remain; i=i+4)
        {
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*invWPtr[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*invWPtr[i+3]*YTp[i+3];
            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvj_));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvj_));
        }

        {
            unsigned int i = this->nSpecies-2;
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            ArrRes[i+0][0] = rhoMByWiYTPi0*rhoMvj0_;
            ArrRes[i+0][1] = rhoMByWiYTPi0*rhoMvj1_;
            ArrRes[i+0][2] = rhoMByWiYTPi0*rhoMvj2_;
            ArrRes[i+0][3] = rhoMByWiYTPi0*rhoMvj3_;   
            ArrRes[i+1][0] = rhoMByWiYTPi1*rhoMvj0_;
            ArrRes[i+1][1] = rhoMByWiYTPi1*rhoMvj1_;
            ArrRes[i+1][2] = rhoMByWiYTPi1*rhoMvj2_;
            ArrRes[i+1][3] = rhoMByWiYTPi1*rhoMvj3_;     
        }
     
        ArrRes[j+0][0] += rhoM*invWPtr[j+0];
        ArrRes[j+1][1] += rhoM*invWPtr[j+1];
        ArrRes[j+2][2] += rhoM*invWPtr[j+2];
        ArrRes[j+3][3] += rhoM*invWPtr[j+3];  


        for(unsigned int i=0; i<this->nSpecies - remain; i=i+4)
        {
            double Wi0ByrhoM_ = WiByrhoM[i+0];
            double Wi1ByrhoM_ = WiByrhoM[i+1];
            double Wi2ByrhoM_ = WiByrhoM[i+2];
            double Wi3ByrhoM_ = WiByrhoM[i+3];

            double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;

            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();
            const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i+0)*(n_)];
            const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i+1)*(n_)];
            const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i+2)*(n_)];
            const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i+3)*(n_)];
            for (unsigned int k=0; k<this->nSpecies-remain; k=k+4)
            {

                const double ddNi0dtByVdck0 = JcRowi0[k+0];
                const double ddNi1dtByVdck0 = JcRowi1[k+0];
                const double ddNi2dtByVdck0 = JcRowi2[k+0];
                const double ddNi3dtByVdck0 = JcRowi3[k+0];

                const double ddNi0dtByVdck1 = JcRowi0[k+1];
                const double ddNi1dtByVdck1 = JcRowi1[k+1];
                const double ddNi2dtByVdck1 = JcRowi2[k+1];
                const double ddNi3dtByVdck1 = JcRowi3[k+1];

                const double ddNi0dtByVdck2 = JcRowi0[k+2];
                const double ddNi1dtByVdck2 = JcRowi1[k+2];
                const double ddNi2dtByVdck2 = JcRowi2[k+2];
                const double ddNi3dtByVdck2 = JcRowi3[k+2];

                const double ddNi0dtByVdck3 = JcRowi0[k+3];
                const double ddNi1dtByVdck3 = JcRowi1[k+3];
                const double ddNi2dtByVdck3 = JcRowi2[k+3];
                const double ddNi3dtByVdck3 = JcRowi3[k+3];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);
                __m256d dCk3dYj = _mm256_loadu_pd(&ArrRes[k+3][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
                
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3),dCk3dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3),dCk3dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck3),dCk3dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck3),dCk3dYj,ddNi3dtByVdYj); 
            }
            
            unsigned int h = this->nSpecies-2;
            const double ddNi0dtByVdck0 = JcRowi0[h+0];
            const double ddNi1dtByVdck0 = JcRowi1[h+0];
            const double ddNi2dtByVdck0 = JcRowi2[h+0];
            const double ddNi3dtByVdck0 = JcRowi3[h+0];

            const double ddNi0dtByVdck1 = JcRowi0[h+1];
            const double ddNi1dtByVdck1 = JcRowi1[h+1];
            const double ddNi2dtByVdck1 = JcRowi2[h+1];
            const double ddNi3dtByVdck1 = JcRowi3[h+1];

            __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[h+0][0]);
            __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[h+1][0]);
            ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
            ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
            ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
            ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);
            ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
            ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
            ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
            ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj); 
            __m256d WiByrhoM_0 = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidtv0 = _mm256_set1_pd(dYi0dt);
            __m256d ddYi0dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv0);
            ddYi0dtdYj = _mm256_fmadd_pd(WiByrhoM_0,ddNi0dtByVdYj,ddYi0dtdYj);
            __m256d WiByrhoM_1 = _mm256_set1_pd(Wi1ByrhoM_);
            __m256d dYidtv1 = _mm256_set1_pd(dYi1dt);
            __m256d ddYi1dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv1);
            ddYi1dtdYj = _mm256_fmadd_pd(WiByrhoM_1,ddNi1dtByVdYj,ddYi1dtdYj);
            __m256d WiByrhoM_2 = _mm256_set1_pd(Wi2ByrhoM_);
            __m256d dYidtv2 = _mm256_set1_pd(dYi2dt);
            __m256d ddYi2dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv2);
            ddYi2dtdYj = _mm256_fmadd_pd(WiByrhoM_2,ddNi2dtByVdYj,ddYi2dtdYj);
            __m256d WiByrhoM_3 = _mm256_set1_pd(Wi3ByrhoM_);
            __m256d dYidtv3 = _mm256_set1_pd(dYi3dt);
            __m256d ddYi3dtdYj = _mm256_mul_pd(rhoMvj_,dYidtv3);
            ddYi3dtdYj = _mm256_fmadd_pd(WiByrhoM_3,ddNi3dtByVdYj,ddYi3dtdYj);
            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
            {
                unsigned int i0 = this->nSpecies-2;
                unsigned int i1 = this->nSpecies-1;
                Wi0ByrhoM_ = WiByrhoM[i0];
                Wi1ByrhoM_ = WiByrhoM[i1];
                dYi0dt = dYTpdt[i0]*WiByrhoM[i0];
                dYi1dt = dYTpdt[i1]*WiByrhoM[i1];
                __m256d ddNi0dtByVdYjv = _mm256_setzero_pd();
                __m256d ddNi1dtByVdYjv = _mm256_setzero_pd();
                const double* __restrict__ JcRowi00 = &ddNdtByVdcTp[(i0)*(n_)];
                const double* __restrict__ JcRowi11 = &ddNdtByVdcTp[(i1)*(n_)];                 
                for (unsigned int k=0; k<this->nSpecies-2; k=k+4)
                {
                    const double ddNi0dtByVdck0a = JcRowi00[k+0];
                    const double ddNi1dtByVdck0b = JcRowi11[k+0];
                    __m256d Arrk00 = _mm256_loadu_pd(&ArrRes[k+0][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0a),Arrk00,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0b),Arrk00,ddNi1dtByVdYjv);
                    const double ddNi0dtByVdck1a = JcRowi00[k+1];
                    const double ddNi1dtByVdck1b = JcRowi11[k+1];
                    __m256d Arrk10 = _mm256_loadu_pd(&ArrRes[k+1][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1a),Arrk10,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1b),Arrk10,ddNi1dtByVdYjv);
                    const double ddNi0dtByVdck2a = JcRowi00[k+2];
                    const double ddNi1dtByVdck2b = JcRowi11[k+2];
                    __m256d Arrk20 = _mm256_loadu_pd(&ArrRes[k+2][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2a),Arrk20,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2b),Arrk20,ddNi1dtByVdYjv);
                    const double ddNi0dtByVdck3a = JcRowi00[k+3];
                    const double ddNi1dtByVdck3b = JcRowi11[k+3];
                    __m256d Arrk30 = _mm256_loadu_pd(&ArrRes[k+3][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3a),Arrk30,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3b),Arrk30,ddNi1dtByVdYjv);  
                }
                {
                    unsigned int k0 = this->nSpecies-2;
                    unsigned int k1 = this->nSpecies-1;
                    const double ddNi0dtByVdck0a = JcRowi00[k0];
                    const double ddNi1dtByVdck0b = JcRowi11[k0];
                    __m256d Arrk00 = _mm256_loadu_pd(&ArrRes[k0][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0a),Arrk00,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0b),Arrk00,ddNi1dtByVdYjv); 

                    const double ddNi0dtByVdck1a = JcRowi00[k1];
                    const double ddNi1dtByVdck1b = JcRowi11[k1];
                    __m256d Arrk10 = _mm256_loadu_pd(&ArrRes[k1][0]);
                    ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1a),Arrk10,ddNi0dtByVdYjv);
                    ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1b),Arrk10,ddNi1dtByVdYjv);
                }
                __m256d r0 = _mm256_mul_pd(ddNi0dtByVdYjv,_mm256_set1_pd(Wi0ByrhoM_));
                r0 = _mm256_fmadd_pd(rhoMvj_,_mm256_set1_pd(dYi0dt),r0);
                _mm256_storeu_pd(&J[i0*(n_) + j+0],r0);
                __m256d r1 = _mm256_mul_pd(ddNi1dtByVdYjv,_mm256_set1_pd(Wi1ByrhoM_));
                r1 = _mm256_fmadd_pd(rhoMvj_,_mm256_set1_pd(dYi1dt),r1);
                _mm256_storeu_pd(&J[i1*(n_) + j+0],r1);
            }
        }
    }
    
    unsigned int j0 = this->nSpecies-2;
    unsigned int j1 = this->nSpecies-1;

    __m128d rhoMvjv = _mm_loadu_pd(&rhoMvj[j0]);
    for(unsigned int i=0; i<this->nSpecies-2; i=i+4)
    {
        const double rhoMByWi0 = rhoM*invWPtr[i+0];
        const double rhoMByWi1 = rhoM*invWPtr[i+1];
        const double rhoMByWi2 = rhoM*invWPtr[i+2];
        const double rhoMByWi3 = rhoM*invWPtr[i+3];

        __m128d Arr0 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi0),rhoMvjv),_mm_set1_pd(YTp[i+0]));
        _mm_storeu_pd(&ArrRes[i+0][0],Arr0);
        __m128d Arr1 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi1),rhoMvjv),_mm_set1_pd(YTp[i+1]));
        _mm_storeu_pd(&ArrRes[i+1][0],Arr1);
        __m128d Arr2 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi2),rhoMvjv),_mm_set1_pd(YTp[i+2]));
        _mm_storeu_pd(&ArrRes[i+2][0],Arr2); 
        __m128d Arr3 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi3),rhoMvjv),_mm_set1_pd(YTp[i+3]));
        _mm_storeu_pd(&ArrRes[i+3][0],Arr3);                
    }
    {
        unsigned int i = this->nSpecies-2;
        const double rhoMByWi0 = rhoM*invWPtr[i+0];
        const double rhoMByWi1 = rhoM*invWPtr[i+1];
        __m128d Arr0 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi0),rhoMvjv),_mm_set1_pd(YTp[i+0]));
        _mm_storeu_pd(&ArrRes[i+0][0],Arr0);  
        __m128d Arr1 = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(-rhoMByWi1),rhoMvjv),_mm_set1_pd(YTp[i+1]));
        _mm_storeu_pd(&ArrRes[i+1][0],Arr1);             
    }
    ArrRes[j0][0] = ArrRes[j0][0] + rhoM*invWPtr[j0];
    ArrRes[j1][1] = ArrRes[j1][1] + rhoM*invWPtr[j1];
    for(unsigned int i=0; i<this->nSpecies; i++)
    {
        const double WiByrhoM_ = WiByrhoM[i];

        double dYidt = dYTpdt[i]*WiByrhoM_;

        const double* __restrict__ JcRowi = &ddNdtByVdcTp[i*(n_)];
        __m128d ddNidtByVdYjv = _mm_setzero_pd();
        for (unsigned int k=0; k<this->nSpecies-2; k=k+4)
        {
            const double ddNidtByVdck0 = JcRowi[k+0];
            const double ddNidtByVdck1 = JcRowi[k+1];
            const double ddNidtByVdck2 = JcRowi[k+2];
            const double ddNidtByVdck3 = JcRowi[k+3];
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck0),_mm_loadu_pd(&ArrRes[k+0][0]),ddNidtByVdYjv);
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck1),_mm_loadu_pd(&ArrRes[k+1][0]),ddNidtByVdYjv);
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck2),_mm_loadu_pd(&ArrRes[k+2][0]),ddNidtByVdYjv);
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck3),_mm_loadu_pd(&ArrRes[k+3][0]),ddNidtByVdYjv);
        }
        {
            unsigned int k = this->nSpecies-2;
            const double ddNidtByVdck0 = JcRowi[k+0];
            const double ddNidtByVdck1 = JcRowi[k+1];
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck0),_mm_loadu_pd(&ArrRes[k+0][0]),ddNidtByVdYjv);
            ddNidtByVdYjv = _mm_fmadd_pd(_mm_set1_pd(ddNidtByVdck1),_mm_loadu_pd(&ArrRes[k+1][0]),ddNidtByVdYjv);
        }
        __m128d WiByrhoMv = _mm_set1_pd(WiByrhoM_);
        __m128d rhiMvjv = _mm_loadu_pd(&rhoMvj[j0]);
        __m128d dYidtv = _mm_set1_pd(dYidt);
        __m128d result = _mm_mul_pd(WiByrhoMv,ddNidtByVdYjv);
        result = _mm_fmadd_pd(rhiMvjv,dYidtv,result);
        _mm_storeu_pd(&J[i*(n_) + j0],result);
    }
        
}

void 
OptReaction::ddYdtdY_Vec1_3
(
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    double* __restrict__ invWPtr = &invW[0];
    double* const* __restrict__ ArrRes = this->ArrPtr.data();

    for(unsigned int j = 0; j < this->nSpecies - 3; j = j + 4)
    {
        __m256d rhoMvjv = _mm256_loadu_pd(&rhoMvj[j+0]);
        for(unsigned int i = 0; i < this->nSpecies-3; i=i+4)
        {
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            const double rhoMByWiYTPi2 = -rhoM*invWPtr[i+2]*YTp[i+2];
            const double rhoMByWiYTPi3 = -rhoM*invWPtr[i+3]*YTp[i+3];

            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi0),rhoMvjv));
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi1),rhoMvjv));
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi2),rhoMvjv));
            _mm256_storeu_pd(&ArrRes[i+3][0],_mm256_mul_pd(_mm256_set1_pd(rhoMByWiYTPi3),rhoMvjv));
        }
        {
            unsigned int i = this->nSpecies-3;
            const double rhoMByWiYTPi0 = -rhoM*invWPtr[i+0]*YTp[i+0];
            __m256d rhoMByWiYTPi0v = _mm256_set1_pd(rhoMByWiYTPi0);
            _mm256_storeu_pd(&ArrRes[i+0][0],_mm256_mul_pd(rhoMByWiYTPi0v,rhoMvjv));
            const double rhoMByWiYTPi1 = -rhoM*invWPtr[i+1]*YTp[i+1];
            __m256d rhoMByWiYTPi1v = _mm256_set1_pd(rhoMByWiYTPi1);
            _mm256_storeu_pd(&ArrRes[i+1][0],_mm256_mul_pd(rhoMByWiYTPi1v,rhoMvjv));
            const double rhoMByWiYTPi2 = -rhoM*invWPtr[i+2]*YTp[i+2];
            __m256d rhoMByWiYTPi2v = _mm256_set1_pd(rhoMByWiYTPi2);
            _mm256_storeu_pd(&ArrRes[i+2][0],_mm256_mul_pd(rhoMByWiYTPi2v,rhoMvjv));             
        }
        ArrRes[j+0][0] += rhoM*invWPtr[j+0];
        ArrRes[j+1][1] += rhoM*invWPtr[j+1];
        ArrRes[j+2][2] += rhoM*invWPtr[j+2];
        ArrRes[j+3][3] += rhoM*invWPtr[j+3];  

        for(unsigned int i=0; i<this->nSpecies - 3; i=i+4)
        {
            const double Wi0ByrhoM_ = WiByrhoM[i+0];
            const double Wi1ByrhoM_ = WiByrhoM[i+1];
            const double Wi2ByrhoM_ = WiByrhoM[i+2];
            const double Wi3ByrhoM_ = WiByrhoM[i+3];

            const double dYi0dt = dYTpdt[i+0]*Wi0ByrhoM_;
            const double dYi1dt = dYTpdt[i+1]*Wi1ByrhoM_;
            const double dYi2dt = dYTpdt[i+2]*Wi2ByrhoM_;
            const double dYi3dt = dYTpdt[i+3]*Wi3ByrhoM_;

            __m256d ddNi0dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYj = _mm256_setzero_pd();
            __m256d ddNi3dtByVdYj = _mm256_setzero_pd();
            const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i+0)*(n_)];
            const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i+1)*(n_)];
            const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i+2)*(n_)];
            const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i+3)*(n_)];
            for (unsigned int k=0; k<this->nSpecies-3; k=k+4)
            {

                const double ddNi0dtByVdck0 = JcRowi0[k+0];
                const double ddNi1dtByVdck0 = JcRowi1[k+0];
                const double ddNi2dtByVdck0 = JcRowi2[k+0];
                const double ddNi3dtByVdck0 = JcRowi3[k+0];

                const double ddNi0dtByVdck1 = JcRowi0[k+1];
                const double ddNi1dtByVdck1 = JcRowi1[k+1];
                const double ddNi2dtByVdck1 = JcRowi2[k+1];
                const double ddNi3dtByVdck1 = JcRowi3[k+1];

                const double ddNi0dtByVdck2 = JcRowi0[k+2];
                const double ddNi1dtByVdck2 = JcRowi1[k+2];
                const double ddNi2dtByVdck2 = JcRowi2[k+2];
                const double ddNi3dtByVdck2 = JcRowi3[k+2];

                const double ddNi0dtByVdck3 = JcRowi0[k+3];
                const double ddNi1dtByVdck3 = JcRowi1[k+3];
                const double ddNi2dtByVdck3 = JcRowi2[k+3];
                const double ddNi3dtByVdck3 = JcRowi3[k+3];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);
                __m256d dCk3dYj = _mm256_loadu_pd(&ArrRes[k+3][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
                
                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck3),dCk3dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck3),dCk3dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck3),dCk3dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck3),dCk3dYj,ddNi3dtByVdYj); 
            }

            {
                unsigned int k = this->nSpecies-3;
                const double ddNi0dtByVdck0 = JcRowi0[k+0];
                const double ddNi1dtByVdck0 = JcRowi1[k+0];
                const double ddNi2dtByVdck0 = JcRowi2[k+0];
                const double ddNi3dtByVdck0 = JcRowi3[k+0];

                const double ddNi0dtByVdck1 = JcRowi0[k+1];
                const double ddNi1dtByVdck1 = JcRowi1[k+1];
                const double ddNi2dtByVdck1 = JcRowi2[k+1];
                const double ddNi3dtByVdck1 = JcRowi3[k+1];

                const double ddNi0dtByVdck2 = JcRowi0[k+2];
                const double ddNi1dtByVdck2 = JcRowi1[k+2];
                const double ddNi2dtByVdck2 = JcRowi2[k+2];
                const double ddNi3dtByVdck2 = JcRowi3[k+2];

                __m256d dCk0dYj = _mm256_loadu_pd(&ArrRes[k+0][0]);
                __m256d dCk1dYj = _mm256_loadu_pd(&ArrRes[k+1][0]);
                __m256d dCk2dYj = _mm256_loadu_pd(&ArrRes[k+2][0]);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck0),dCk0dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck0),dCk0dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck0),dCk0dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck0),dCk0dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck1),dCk1dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck1),dCk1dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck1),dCk1dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck1),dCk1dYj,ddNi3dtByVdYj);

                ddNi0dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck2),dCk2dYj,ddNi0dtByVdYj);
                ddNi1dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck2),dCk2dYj,ddNi1dtByVdYj);
                ddNi2dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck2),dCk2dYj,ddNi2dtByVdYj);
                ddNi3dtByVdYj = _mm256_fmadd_pd(_mm256_set1_pd(ddNi3dtByVdck2),dCk2dYj,ddNi3dtByVdYj);     
            }
            __m256d WiByrhoM_0 = _mm256_set1_pd(Wi0ByrhoM_);
            __m256d dYidtv0 = _mm256_set1_pd(dYi0dt);
            __m256d ddYi0dtdYj = _mm256_mul_pd(rhoMvjv,dYidtv0);
            ddYi0dtdYj = _mm256_fmadd_pd(WiByrhoM_0,ddNi0dtByVdYj,ddYi0dtdYj);
            __m256d WiByrhoM_1 = _mm256_set1_pd(Wi1ByrhoM_);
            __m256d dYidtv1 = _mm256_set1_pd(dYi1dt);
            __m256d ddYi1dtdYj = _mm256_mul_pd(rhoMvjv,dYidtv1);
            ddYi1dtdYj = _mm256_fmadd_pd(WiByrhoM_1,ddNi1dtByVdYj,ddYi1dtdYj);
            __m256d WiByrhoM_2 = _mm256_set1_pd(Wi2ByrhoM_);
            __m256d dYidtv2 = _mm256_set1_pd(dYi2dt);
            __m256d ddYi2dtdYj = _mm256_mul_pd(rhoMvjv,dYidtv2);
            ddYi2dtdYj = _mm256_fmadd_pd(WiByrhoM_2,ddNi2dtByVdYj,ddYi2dtdYj);
            __m256d WiByrhoM_3 = _mm256_set1_pd(Wi3ByrhoM_);
            __m256d dYidtv3 = _mm256_set1_pd(dYi3dt);
            __m256d ddYi3dtdYj = _mm256_mul_pd(rhoMvjv,dYidtv3);
            ddYi3dtdYj = _mm256_fmadd_pd(WiByrhoM_3,ddNi3dtByVdYj,ddYi3dtdYj);
            _mm256_storeu_pd(&J[(i+0)*(n_)+ j+0],ddYi0dtdYj);
            _mm256_storeu_pd(&J[(i+1)*(n_)+ j+0],ddYi1dtdYj);
            _mm256_storeu_pd(&J[(i+2)*(n_)+ j+0],ddYi2dtdYj);
            _mm256_storeu_pd(&J[(i+3)*(n_)+ j+0],ddYi3dtdYj);
        }     
        
        {
            unsigned int i0 = this->nSpecies-3;            
            unsigned int i1 = this->nSpecies-2;
            unsigned int i2 = this->nSpecies-1;
            const double Wi0ByrhoM_ = WiByrhoM[i0];
            const double Wi1ByrhoM_ = WiByrhoM[i1];
            const double Wi2ByrhoM_ = WiByrhoM[i2];
            double dYi0dt = dYTpdt[i0]*Wi0ByrhoM_;
            double dYi1dt = dYTpdt[i1]*Wi1ByrhoM_;
            double dYi2dt = dYTpdt[i2]*Wi2ByrhoM_;

            __m256d ddNi0dtByVdYjv = _mm256_setzero_pd();
            __m256d ddNi1dtByVdYjv = _mm256_setzero_pd();
            __m256d ddNi2dtByVdYjv = _mm256_setzero_pd();
            const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[i0*(n_)];
            const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[i1*(n_)];
            const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[i2*(n_)];
            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                const double ddNi0dtByVdck = JcRowi0[k];
                const double ddNi1dtByVdck = JcRowi1[k];
                const double ddNi2dtByVdck = JcRowi2[k];
                __m256d Arrkv = _mm256_loadu_pd(&ArrRes[k][0]);
                ddNi0dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi0dtByVdck),Arrkv,ddNi0dtByVdYjv);
                ddNi1dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi1dtByVdck),Arrkv,ddNi1dtByVdYjv);
                ddNi2dtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNi2dtByVdck),Arrkv,ddNi2dtByVdYjv);
            }
            __m256d result0 = _mm256_mul_pd(rhoMvjv,_mm256_set1_pd(dYi0dt));
            result0 = _mm256_fmadd_pd(_mm256_set1_pd(Wi0ByrhoM_),ddNi0dtByVdYjv,result0);
            _mm256_storeu_pd(&J[i0*(n_) + j+0],result0);
            __m256d result1 = _mm256_mul_pd(rhoMvjv,_mm256_set1_pd(dYi1dt));
            result1 = _mm256_fmadd_pd(_mm256_set1_pd(Wi1ByrhoM_),ddNi1dtByVdYjv,result1);
            _mm256_storeu_pd(&J[i1*(n_) + j+0],result1);    
            __m256d result2 = _mm256_mul_pd(rhoMvjv,_mm256_set1_pd(dYi2dt));
            result2 = _mm256_fmadd_pd(_mm256_set1_pd(Wi2ByrhoM_),ddNi2dtByVdYjv,result2);
            _mm256_storeu_pd(&J[i2*(n_) + j+0],result2);             
        }        
    }
    {
        unsigned int j0 = this->nSpecies-3;
        unsigned int j1 = this->nSpecies-2;
        unsigned int j2 = this->nSpecies-1;

        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*invWPtr[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];
            ArrRes[i][0] = rhoMByWi*(0 - rhoMvj0_*YTp[i]);
            ArrRes[i][1] = rhoMByWi*(0 - rhoMvj1_*YTp[i]);
            ArrRes[i][2] = rhoMByWi*(0 - rhoMvj2_*YTp[i]);
            ArrRes[i][3] = 0;
        }
        ArrRes[j0][0] = ArrRes[j0][0] + rhoM*invWPtr[j0];
        ArrRes[j1][1] = ArrRes[j1][1] + rhoM*invWPtr[j1];
        ArrRes[j2][2] = ArrRes[j2][2] + rhoM*invWPtr[j2];
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double WiByrhoM_ = WiByrhoM[i];
            const double rhoMvj0_ = rhoMvj[j0];
            const double rhoMvj1_ = rhoMvj[j1];
            const double rhoMvj2_ = rhoMvj[j2];

            double dYidt = dYTpdt[i]*WiByrhoM[i];

            __m256d ddNidtByVdYjv = _mm256_setzero_pd();
            const double* __restrict__ JcRowi = &ddNdtByVdcTp[i*(n_)];
            for (unsigned int k=0; k<this->nSpecies-3; k=k+4)
            {
                const double ddNidtByVdck0 = JcRowi[k+0];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck0),_mm256_loadu_pd(&ArrRes[k+0][0]),ddNidtByVdYjv);
                const double ddNidtByVdck1 = JcRowi[k+1];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck1),_mm256_loadu_pd(&ArrRes[k+1][0]),ddNidtByVdYjv);
                const double ddNidtByVdck2 = JcRowi[k+2];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck2),_mm256_loadu_pd(&ArrRes[k+2][0]),ddNidtByVdYjv);
                const double ddNidtByVdck3 = JcRowi[k+3];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck3),_mm256_loadu_pd(&ArrRes[k+3][0]),ddNidtByVdYjv);
            }
            {
                unsigned int k = this->nSpecies-3;
                const double ddNidtByVdck0 = JcRowi[k+0];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck0),_mm256_loadu_pd(&ArrRes[k+0][0]),ddNidtByVdYjv);
                const double ddNidtByVdck1 = JcRowi[k+1];
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck1),_mm256_loadu_pd(&ArrRes[k+1][0]),ddNidtByVdYjv);
                const double ddNidtByVdck2 = JcRowi[k+2];   
                ddNidtByVdYjv = _mm256_fmadd_pd(_mm256_set1_pd(ddNidtByVdck2),_mm256_loadu_pd(&ArrRes[k+2][0]),ddNidtByVdYjv);       
            }

            __m256d rhoMvjv = _mm256_setr_pd(rhoMvj0_,rhoMvj1_,rhoMvj2_,0);
            __m256d result = _mm256_mul_pd(rhoMvjv,_mm256_set1_pd(dYidt));
            result = _mm256_fmadd_pd(ddNidtByVdYjv,_mm256_set1_pd(WiByrhoM_),result);
            __m128d lo = _mm256_castpd256_pd128(result);         
            __m128d hi = _mm256_extractf128_pd(result, 1);       
            double r0 = _mm_cvtsd_f64(lo);
            double r1 = _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));
            double r2 = _mm_cvtsd_f64(hi);
            J[i*(n_) + j0] = r0;
            J[i*(n_) + j1] = r1;
            J[i*(n_) + j2] = r2;
        }
    }        
}
void 
OptReaction::ddYdtdY
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    scalarField Array0(this->nSpecies);

    for (unsigned int j=0; j<this->nSpecies; j++)
    {
        for(unsigned int i=0; i<this->nSpecies; i++)
        {
            const double rhoMByWi = rhoM*this->invW[i];
            const double rhoMvj_ = rhoMvj[j];
            Array0[i] = rhoMByWi*((i == j) - rhoMvj_*YTp[i]);
        }

        for(unsigned int i=0; i<this->nSpecies; i++)
        {

            const double WiByrhoM_ = WiByrhoM[i];

            const double rhoMvj_ = rhoMvj[j];
            double dYidt = dYTpdt[i];
            dYidt *= WiByrhoM_;

            double ddNidtByVdYj = 0;

            for (unsigned int k=0; k<this->nSpecies; k++)
            {
                 const double ddNidtByVdck = ddNdtByVdcTp[i*(n_) + k];
                 ddNidtByVdYj += ddNidtByVdck*Array0[k];            
            }

            double& ddYidtdYj = J[i*(n_) + j];
            ddYidtdYj = WiByrhoM_*ddNidtByVdYj + rhoMvj_*dYidt;
        }
    }
}

void 
OptReaction::ddYdtdY2
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    for (unsigned int i = 0; i < nSpecies; ++i)
    {
        const double WiByrhoM_ = WiByrhoM[i];
        const double dYidt = dYTpdt[i] * WiByrhoM_;

        for (unsigned int j = 0; j < nSpecies; ++j)
        {
            const double rhoMvj_ = rhoMvj[j];
            double ddNidtByVdYj = 0.0;
            for (unsigned int k = 0; k < nSpecies; ++k)
            {
                const double rhoMByWk = rhoM * this->invW[k];
                const double coeff = rhoMByWk * ((k == j) - rhoMvj_ * YTp[k]);
                ddNidtByVdYj += ddNdtByVdcTp[i*(n_) + k] * coeff;
            }

            J[i*(n_) + j] = WiByrhoM_ * ddNidtByVdYj + rhoMvj_ * dYidt;
        }
    }
}


void 
OptReaction::ddYdtdY3
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    unsigned int remain = nSpecies%4;
    std::vector<std::vector<double>> dcdY(nSpecies, std::vector<double>(nSpecies));

    for (unsigned int j = 0; j < nSpecies-remain; j=j+4)
    {    
        __m256d rhoMv = _mm256_set1_pd(-rhoM);
        __m256d rhoMvj_0v = _mm256_set1_pd(rhoMvj[j+0]);
        __m256d rhoMvj_1v = _mm256_set1_pd(rhoMvj[j+1]);
        __m256d rhoMvj_2v = _mm256_set1_pd(rhoMvj[j+2]);
        __m256d rhoMvj_3v = _mm256_set1_pd(rhoMvj[j+3]);        
        for (unsigned int k = 0; k < nSpecies-remain; k=k+4)
        {
            __m256d invWv = _mm256_loadu_pd(&invW[k+0]);
            __m256d YTpv = _mm256_loadu_pd(&YTp[k+0]);            
            __m256d dcjdYk0v = _mm256_mul_pd(rhoMv,_mm256_mul_pd(invWv,_mm256_mul_pd(YTpv,rhoMvj_0v)));
            _mm256_storeu_pd(&dcdY[j+0][k+0],dcjdYk0v);
            __m256d dcjdYk1v = _mm256_mul_pd(rhoMv,_mm256_mul_pd(invWv,_mm256_mul_pd(YTpv,rhoMvj_1v)));
            _mm256_storeu_pd(&dcdY[j+1][k+0],dcjdYk1v);
            __m256d dcjdYk2v = _mm256_mul_pd(rhoMv,_mm256_mul_pd(invWv,_mm256_mul_pd(YTpv,rhoMvj_2v)));
            _mm256_storeu_pd(&dcdY[j+2][k+0],dcjdYk2v);
            __m256d dcjdYk3v = _mm256_mul_pd(rhoMv,_mm256_mul_pd(invWv,_mm256_mul_pd(YTpv,rhoMvj_3v)));
            _mm256_storeu_pd(&dcdY[j+3][k+0],dcjdYk3v);

        }

        dcdY[j+0][j+0] += rhoM * invW[j+0];
        dcdY[j+1][j+1] += rhoM * invW[j+1];
        dcdY[j+2][j+2] += rhoM * invW[j+2];
        dcdY[j+3][j+3] += rhoM * invW[j+3];                        
    }
    if(remain==1)
    {
        for (unsigned int j = 0; j < nSpecies-1; ++j)
        {
            const double rhoMvj_ = rhoMvj[j];
            unsigned int k = nSpecies-1;
            dcdY[j][k] = -rhoM * invW[k] *  rhoMvj_ * YTp[k];    
        }
        unsigned int j = nSpecies-1; 
        const double rhoMvj_ = rhoMvj[j];
        for (unsigned int k = 0; k < nSpecies-1; ++k)
        {
            dcdY[j][k] = -rhoM * invW[k] *  rhoMvj_ * YTp[k];
        }
        dcdY[j][j] = rhoM * invW[j] - rhoM * invW[j] *  rhoMvj_ * YTp[j];
    }
    else if(remain==2)
    {
        for (unsigned int j = 0; j < nSpecies-2; ++j)
        {
            const double rhoMvj_ = rhoMvj[j];
            unsigned int k = nSpecies-2;
            dcdY[j][k+0] = -rhoM * invW[k+0] *  rhoMvj_ * YTp[k+0];
            dcdY[j][k+1] = -rhoM * invW[k+1] *  rhoMvj_ * YTp[k+1];
        }
        unsigned int j = nSpecies-2; 
        const double rhoMvj_0 = rhoMvj[j+0];
        const double rhoMvj_1 = rhoMvj[j+1];        
        for (unsigned int k = 0; k < nSpecies-2; ++k)
        {
            dcdY[j+0][k] = -rhoM * invW[k] *  rhoMvj_0 * YTp[k];
            dcdY[j+1][k] = -rhoM * invW[k] *  rhoMvj_1 * YTp[k];            
        }
        dcdY[j+0][j+0] = rhoM * invW[j+0] - rhoM * invW[j+0] *  rhoMvj_0 * YTp[j+0];
        dcdY[j+1][j+1] = rhoM * invW[j+1] - rhoM * invW[j+1] *  rhoMvj_1 * YTp[j+1];                
    }
    else if(remain==3)
    {
        for (unsigned int j = 0; j < nSpecies-3; ++j)
        {
            const double rhoMvj_ = rhoMvj[j];
            unsigned int k = nSpecies-2;
            dcdY[j][k+0] = -rhoM * invW[k+0] *  rhoMvj_ * YTp[k+0];
            dcdY[j][k+1] = -rhoM * invW[k+1] *  rhoMvj_ * YTp[k+1];
            dcdY[j][k+2] = -rhoM * invW[k+2] *  rhoMvj_ * YTp[k+2];            
        }
        unsigned int j = nSpecies-3; 
        const double rhoMvj_0 = rhoMvj[j+0];
        const double rhoMvj_1 = rhoMvj[j+1];
        const double rhoMvj_2 = rhoMvj[j+2];        
        for (unsigned int k = 0; k < nSpecies-3; ++k)
        {
            dcdY[j+0][k] = -rhoM * invW[k] *  rhoMvj_0 * YTp[k];
            dcdY[j+1][k] = -rhoM * invW[k] *  rhoMvj_1 * YTp[k];  
            dcdY[j+2][k] = -rhoM * invW[k] *  rhoMvj_2 * YTp[k];                      
        }
        dcdY[j+0][j+0] = rhoM * invW[j+0] - rhoM * invW[j+0] *  rhoMvj_0 * YTp[j+0];
        dcdY[j+1][j+1] = rhoM * invW[j+1] - rhoM * invW[j+1] *  rhoMvj_1 * YTp[j+1];  
        dcdY[j+2][j+2] = rhoM * invW[j+2] - rhoM * invW[j+2] *  rhoMvj_1 * YTp[j+2];                       
    }


    for (unsigned int i = 0; i < nSpecies; ++i)
    {
        const double WiByrhoM_ = WiByrhoM[i];
        const double dYidt = dYTpdt[i] * WiByrhoM_;
        const double* ddNidtByVdc = &ddNdtByVdcTp[i * n_];

        for (unsigned int j = 0; j < nSpecies-remain; j=j+4)
        {

            __m256d rhoMvj_v = _mm256_loadu_pd(&rhoMvj[j+0]);

            const double* dcdYj0 = dcdY[j+0].data();
            const double* dcdYj1 = dcdY[j+1].data();
            const double* dcdYj2 = dcdY[j+2].data();
            const double* dcdYj3 = dcdY[j+3].data();

            __m256d sum0v = _mm256_setzero_pd();
            __m256d sum1v = _mm256_setzero_pd();
            __m256d sum2v = _mm256_setzero_pd();
            __m256d sum3v = _mm256_setzero_pd();

            for (unsigned int k = 0; k < nSpecies-remain; k=k+4)
            {

                __m256d ddNidtByVdcv = _mm256_loadu_pd(&ddNidtByVdc[k+0]);
                __m256d dcdYj0v = _mm256_loadu_pd(&dcdYj0[k+0]);  
                sum0v = _mm256_fmadd_pd(ddNidtByVdcv,dcdYj0v,sum0v);
                __m256d dcdYj1v = _mm256_loadu_pd(&dcdYj1[k+0]);  
                sum1v = _mm256_fmadd_pd(ddNidtByVdcv,dcdYj1v,sum1v);
                __m256d dcdYj2v = _mm256_loadu_pd(&dcdYj2[k+0]);  
                sum2v = _mm256_fmadd_pd(ddNidtByVdcv,dcdYj2v,sum2v);
                __m256d dcdYj3v = _mm256_loadu_pd(&dcdYj3[k+0]);  
                sum3v = _mm256_fmadd_pd(ddNidtByVdcv,dcdYj3v,sum3v);
            }

            auto hsum_avx = [](const __m256d v) -> double {
                __m128d lo = _mm256_castpd256_pd128(v);
                __m128d hi = _mm256_extractf128_pd(v, 1);
                __m128d sum = _mm_add_pd(lo, hi);
                sum = _mm_hadd_pd(sum, sum);
                return _mm_cvtsd_f64(sum);
            };
            double sum0 = hsum_avx(sum0v);
            double sum1 = hsum_avx(sum1v);
            double sum2 = hsum_avx(sum2v);
            double sum3 = hsum_avx(sum3v);

            for (unsigned int k = nSpecies -remain; k < nSpecies; ++k)
            {
                double ddNidt = ddNidtByVdc[k];
                sum0 += ddNidt * dcdYj0[k];
                sum1 += ddNidt * dcdYj1[k];
                sum2 += ddNidt * dcdYj2[k];
                sum3 += ddNidt * dcdYj3[k];
            }
            __m256d  Sum = _mm256_setr_pd(sum0,sum1,sum2,sum3);
            __m256d Jv = _mm256_mul_pd(rhoMvj_v,_mm256_set1_pd(dYidt));
            Jv = _mm256_fmadd_pd(Sum,_mm256_set1_pd(WiByrhoM_),Jv);
            _mm256_storeu_pd(&J[i * n_ + j+0],Jv);
        }
        if(remain==1)
        {
            unsigned int j = nSpecies-1;
            const double rhoMvj_ = rhoMvj[j];
            const double* dcdYj = dcdY[j].data();
            double sum = 0;
            for (unsigned int k = 0; k < nSpecies-remain; k=k+4)
            {
                sum += ddNidtByVdc[k+0] * dcdYj[k+0];
                sum += ddNidtByVdc[k+1] * dcdYj[k+1];
                sum += ddNidtByVdc[k+2] * dcdYj[k+2];
                sum += ddNidtByVdc[k+3] * dcdYj[k+3];
            }
            unsigned int k = nSpecies-1;
            sum += ddNidtByVdc[k] * dcdYj[k];
            J[i * n_ + j] = WiByrhoM_ * sum + rhoMvj_ * dYidt;
        }
        else if(remain==2)
        {
            unsigned int j = nSpecies-2;
            const double rhoMvj_0 = rhoMvj[j+0];
            const double rhoMvj_1 = rhoMvj[j+1];

            const double* dcdYj0 = dcdY[j+0].data();
            const double* dcdYj1 = dcdY[j+1].data();

            double sum0 = 0;
            double sum1 = 0;

            for (unsigned int k = 0; k < nSpecies-remain; k=k+4)
            {
                sum0 += ddNidtByVdc[k+0] * dcdYj0[k+0];
                sum0 += ddNidtByVdc[k+1] * dcdYj0[k+1];
                sum0 += ddNidtByVdc[k+2] * dcdYj0[k+2];
                sum0 += ddNidtByVdc[k+3] * dcdYj0[k+3];   

                sum1 += ddNidtByVdc[k+0] * dcdYj1[k+0];
                sum1 += ddNidtByVdc[k+1] * dcdYj1[k+1];
                sum1 += ddNidtByVdc[k+2] * dcdYj1[k+2];
                sum1 += ddNidtByVdc[k+3] * dcdYj1[k+3]; 
            }
            {
                unsigned int k = nSpecies-2;
                sum0 += ddNidtByVdc[k+0] * dcdYj0[k+0];
                sum0 += ddNidtByVdc[k+1] * dcdYj0[k+1];     
                sum1 += ddNidtByVdc[k+0] * dcdYj1[k+0];
                sum1 += ddNidtByVdc[k+1] * dcdYj1[k+1];                 
                           
            }
            J[i * n_ + j+0] = WiByrhoM_ * sum0 + rhoMvj_0 * dYidt;
            J[i * n_ + j+1] = WiByrhoM_ * sum1 + rhoMvj_1 * dYidt;                      
        }
        else if(remain==3)
        {
            unsigned int j = nSpecies-3;
            const double rhoMvj_0 = rhoMvj[j+0];
            const double rhoMvj_1 = rhoMvj[j+1];
            const double rhoMvj_2 = rhoMvj[j+2];
            const double* dcdYj0 = dcdY[j+0].data();
            const double* dcdYj1 = dcdY[j+1].data();
            const double* dcdYj2 = dcdY[j+2].data();
            double sum0 = 0;
            double sum1 = 0;
            double sum2 = 0;
            for (unsigned int k = 0; k < nSpecies-remain; k=k+4)
            {
                sum0 += ddNidtByVdc[k+0] * dcdYj0[k+0];
                sum0 += ddNidtByVdc[k+1] * dcdYj0[k+1];
                sum0 += ddNidtByVdc[k+2] * dcdYj0[k+2];
                sum0 += ddNidtByVdc[k+3] * dcdYj0[k+3];   

                sum1 += ddNidtByVdc[k+0] * dcdYj1[k+0];
                sum1 += ddNidtByVdc[k+1] * dcdYj1[k+1];
                sum1 += ddNidtByVdc[k+2] * dcdYj1[k+2];
                sum1 += ddNidtByVdc[k+3] * dcdYj1[k+3]; 
                
                sum2 += ddNidtByVdc[k+0] * dcdYj2[k+0];
                sum2 += ddNidtByVdc[k+1] * dcdYj2[k+1];
                sum2 += ddNidtByVdc[k+2] * dcdYj2[k+2];
                sum2 += ddNidtByVdc[k+3] * dcdYj2[k+3];       
            }
            {
                unsigned int k = nSpecies-3;
                sum0 += ddNidtByVdc[k+0] * dcdYj0[k+0];
                sum0 += ddNidtByVdc[k+1] * dcdYj0[k+1];  
                sum0 += ddNidtByVdc[k+2] * dcdYj0[k+2];
                sum1 += ddNidtByVdc[k+0] * dcdYj1[k+0];
                sum1 += ddNidtByVdc[k+1] * dcdYj1[k+1];  
                sum1 += ddNidtByVdc[k+2] * dcdYj1[k+2];                
                sum2 += ddNidtByVdc[k+0] * dcdYj2[k+0];
                sum2 += ddNidtByVdc[k+1] * dcdYj2[k+1];  
                sum2 += ddNidtByVdc[k+2] * dcdYj2[k+2];                 
            }
            J[i * n_ + j+0] = WiByrhoM_ * sum0 + rhoMvj_0 * dYidt;
            J[i * n_ + j+1] = WiByrhoM_ * sum1 + rhoMvj_1 * dYidt;
            J[i * n_ + j+2] = WiByrhoM_ * sum2 + rhoMvj_2 * dYidt;                             
        }       
    }
}

void 
OptReaction::ddYdtdTP_Vec
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    double alphavM = this->invT;
    unsigned int remain = this->nSpecies%4;
    for (unsigned int i=0; i<this->nSpecies-remain; i=i+4)
    {
        unsigned int i0 = i;
        unsigned int i1 = i+1;
        unsigned int i2 = i+2;
        unsigned int i3 = i+3;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        const double Wi1ByrhoM_ = WiByrhoM[i1];
        const double Wi2ByrhoM_ = WiByrhoM[i2];
        const double Wi3ByrhoM_ = WiByrhoM[i3];
        double& dYi0dt = dYTpdt[i0];
        double& dYi1dt = dYTpdt[i1];
        double& dYi2dt = dYTpdt[i2];
        double& dYi3dt = dYTpdt[i3];
        dYi0dt *= Wi0ByrhoM_;
        dYi1dt *= Wi1ByrhoM_;
        dYi2dt *= Wi2ByrhoM_;
        dYi3dt *= Wi3ByrhoM_;
        double ddNi0dtByVdT = ddNdtByVdcTp[(i0)*(n_) + nSpecies];
        double ddNi1dtByVdT = ddNdtByVdcTp[(i1)*(n_) + nSpecies];
        double ddNi2dtByVdT = ddNdtByVdcTp[(i2)*(n_) + nSpecies];
        double ddNi3dtByVdT = ddNdtByVdcTp[(i3)*(n_) + nSpecies];

        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddNi0dtByVdcj = ddNdtByVdcTp[i0*(n_) + j];
            const double ddNi1dtByVdcj = ddNdtByVdcTp[i1*(n_) + j];
            const double ddNi2dtByVdcj = ddNdtByVdcTp[i2*(n_) + j];
            const double ddNi3dtByVdcj = ddNdtByVdcTp[i3*(n_) + j];
            ddNi0dtByVdT -= ddNi0dtByVdcj*C[j]*alphavM;
            ddNi1dtByVdT -= ddNi1dtByVdcj*C[j]*alphavM;
            ddNi2dtByVdT -= ddNi2dtByVdcj*C[j]*alphavM;
            ddNi3dtByVdT -= ddNi3dtByVdcj*C[j]*alphavM;     
        }
        double& ddYi0dtdT = J[i0*(n_) + nSpecies];
        double& ddYi1dtdT = J[i1*(n_) + nSpecies];
        double& ddYi2dtdT = J[i2*(n_) + nSpecies];
        double& ddYi3dtdT = J[i3*(n_) + nSpecies];
        ddYi0dtdT = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
        ddYi1dtdT = Wi1ByrhoM_*ddNi1dtByVdT + alphavM*dYi1dt;
        ddYi2dtdT = Wi2ByrhoM_*ddNi2dtByVdT + alphavM*dYi2dt;
        ddYi3dtdT = Wi3ByrhoM_*ddNi3dtByVdT + alphavM*dYi3dt;      
    }
    if(remain ==1)
    {
        unsigned int i0 = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        double& dYi0dt = dYTpdt[i0];
        dYi0dt *= Wi0ByrhoM_;
        double ddNi0dtByVdT = ddNdtByVdcTp[(i0)*(n_) + nSpecies];
        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddNi0dtByVdcj = ddNdtByVdcTp[i0*(n_) + j];
            ddNi0dtByVdT -= ddNi0dtByVdcj*C[j]*alphavM;                                
        }
        double& ddYi0dtdT = J[i0*(n_) + nSpecies];
        ddYi0dtdT = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
    }
    else if(remain ==2)
    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        const double Wi1ByrhoM_ = WiByrhoM[i1];
        double& dYi0dt = dYTpdt[i0];
        double& dYi1dt = dYTpdt[i1];
        dYi0dt *= Wi0ByrhoM_;
        dYi1dt *= Wi1ByrhoM_;
        double ddNi0dtByVdT = ddNdtByVdcTp[(i0)*(n_) + nSpecies];
        double ddNi1dtByVdT = ddNdtByVdcTp[(i1)*(n_) + nSpecies];
        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddNi0dtByVdcj = ddNdtByVdcTp[i0*(n_) + j];
            const double ddNi1dtByVdcj = ddNdtByVdcTp[i1*(n_) + j];
            ddNi0dtByVdT -= ddNi0dtByVdcj*C[j]*alphavM;
            ddNi1dtByVdT -= ddNi1dtByVdcj*C[j]*alphavM;                                   
        }
        double& ddYi0dtdT = J[i0*(n_) + nSpecies];
        double& ddYi1dtdT = J[i1*(n_) + nSpecies];
        ddYi0dtdT = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
        ddYi1dtdT = Wi1ByrhoM_*ddNi1dtByVdT + alphavM*dYi1dt;

    }
    else if(remain==3)
    {
        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        const double Wi1ByrhoM_ = WiByrhoM[i1];
        const double Wi2ByrhoM_ = WiByrhoM[i2];
        double& dYi0dt = dYTpdt[i0];
        double& dYi1dt = dYTpdt[i1];
        double& dYi2dt = dYTpdt[i2];
        dYi0dt *= Wi0ByrhoM_;
        dYi1dt *= Wi1ByrhoM_;
        dYi2dt *= Wi2ByrhoM_;
        double ddNi0dtByVdT = ddNdtByVdcTp[(i0)*(n_) + nSpecies];
        double ddNi1dtByVdT = ddNdtByVdcTp[(i1)*(n_) + nSpecies];
        double ddNi2dtByVdT = ddNdtByVdcTp[(i2)*(n_) + nSpecies];
        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddNi0dtByVdcj = ddNdtByVdcTp[i0*(n_) + j];
            const double ddNi1dtByVdcj = ddNdtByVdcTp[i1*(n_) + j];
            const double ddNi2dtByVdcj = ddNdtByVdcTp[i2*(n_) + j];
            ddNi0dtByVdT -= ddNi0dtByVdcj*C[j]*alphavM;
            ddNi1dtByVdT -= ddNi1dtByVdcj*C[j]*alphavM;
            ddNi2dtByVdT -= ddNi2dtByVdcj*C[j]*alphavM;                               
        }
        double& ddYi0dtdT = J[i0*(n_) + nSpecies];
        double& ddYi1dtdT = J[i1*(n_) + nSpecies];
        double& ddYi2dtdT = J[i2*(n_) + nSpecies];
        ddYi0dtdT = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
        ddYi1dtdT = Wi1ByrhoM_*ddNi1dtByVdT + alphavM*dYi1dt;
        ddYi2dtdT = Wi2ByrhoM_*ddNi2dtByVdT + alphavM*dYi2dt;
    }
}

void 
OptReaction::ddYdtdTP_Vec_3
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{

    double alphavM = this->invT;
    __m256d alphavMv = _mm256_set1_pd(this->invT);
    for (unsigned int i=0; i<this->nSpecies-3; i=i+4)
    {
        unsigned int i0 = i;
        unsigned int i1 = i+1;
        unsigned int i2 = i+2;
        unsigned int i3 = i+3;

        __m256d Wi0ByrhoM_v = _mm256_loadu_pd(&WiByrhoM[i0]);
        __m256d dYTpdtv = _mm256_loadu_pd(&dYTpdt[i0]);   
        dYTpdtv = _mm256_mul_pd(dYTpdtv,Wi0ByrhoM_v);   
        _mm256_storeu_pd(&dYTpdt[i0],dYTpdtv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i0]);

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];
        const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i2)*(n_)];
        const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i3)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        double ddNi2dtByVdT = JcRowi2[nSpecies];
        double ddNi3dtByVdT = JcRowi3[nSpecies];
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-3; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            sum0 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi0[j]),Cv),alphavMv,sum0);
            sum1 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi1[j]),Cv),alphavMv,sum1);
            sum2 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi2[j]),Cv),alphavMv,sum2);
            sum3 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi3[j]),Cv),alphavMv,sum3);
        }
        ddNi0dtByVdT = ddNi0dtByVdT - (hsum4(sum0));
        ddNi1dtByVdT = ddNi1dtByVdT - (hsum4(sum1));
        ddNi2dtByVdT = ddNi2dtByVdT - (hsum4(sum2));
        ddNi3dtByVdT = ddNi3dtByVdT - (hsum4(sum3));
        {
            unsigned int j0 = nSpecies-3;
            unsigned int j1 = nSpecies-2;
            unsigned int j2 = nSpecies-1;
            ddNi0dtByVdT -= JcRowi0[j0]*C[j0]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j0]*C[j0]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j0]*C[j0]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j0]*C[j0]*alphavM;

            ddNi0dtByVdT -= JcRowi0[j1]*C[j1]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j1]*C[j1]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j1]*C[j1]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j1]*C[j1]*alphavM;

            ddNi0dtByVdT -= JcRowi0[j2]*C[j2]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j2]*C[j2]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j2]*C[j2]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j2]*C[j2]*alphavM;
        }
        __m256d ddNi0dtByVdTv = _mm256_setr_pd(ddNi0dtByVdT,ddNi1dtByVdT,ddNi2dtByVdT,ddNi3dtByVdT);
        __m256d result = _mm256_fmadd_pd(Wi0ByrhoM_v,ddNi0dtByVdTv,_mm256_mul_pd(alphavMv,dYi0dtv));
        J[i0*(n_) + nSpecies] = get_elem0(result);
        J[i1*(n_) + nSpecies] = get_elem1(result);
        J[i2*(n_) + nSpecies] = get_elem2(result);
        J[i3*(n_) + nSpecies] = get_elem3(result);
      
    }
    {
        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        const double Wi1ByrhoM_ = WiByrhoM[i1];
        const double Wi2ByrhoM_ = WiByrhoM[i2];

        dYTpdt[i0] = dYTpdt[i0]*Wi0ByrhoM_;
        dYTpdt[i1] = dYTpdt[i1]*Wi1ByrhoM_;
        dYTpdt[i2] = dYTpdt[i2]*Wi2ByrhoM_;

        double dYi0dt = dYTpdt[i0];
        double dYi1dt = dYTpdt[i1];
        double dYi2dt = dYTpdt[i2];

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];
        const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i2)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        double ddNi2dtByVdT = JcRowi2[nSpecies];
        __m256d ddNi0dtByVdTv = _mm256_setzero_pd();
        __m256d ddNi1dtByVdTv = _mm256_setzero_pd();
        __m256d ddNi2dtByVdTv = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-3; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            __m256d JcRowi0v = _mm256_loadu_pd(&JcRowi0[j+0]);
            ddNi0dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi0v,Cv),alphavMv,ddNi0dtByVdTv);
            __m256d JcRowi1v = _mm256_loadu_pd(&JcRowi1[j+0]);
            ddNi1dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi1v,Cv),alphavMv,ddNi1dtByVdTv);
            __m256d JcRowi2v = _mm256_loadu_pd(&JcRowi2[j+0]);
            ddNi2dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi2v,Cv),alphavMv,ddNi2dtByVdTv);                            
        }
        {
            unsigned int j = nSpecies-3;
            ddNi0dtByVdT -= JcRowi0[j+0]*C[j+0]*alphavM;
            ddNi0dtByVdT -= JcRowi0[j+1]*C[j+1]*alphavM;
            ddNi0dtByVdT -= JcRowi0[j+2]*C[j+2]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j+0]*C[j+0]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j+1]*C[j+1]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j+2]*C[j+2]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j+0]*C[j+0]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j+1]*C[j+1]*alphavM; 
            ddNi2dtByVdT -= JcRowi2[j+2]*C[j+2]*alphavM;
        }
        ddNi0dtByVdT -= hsum4(ddNi0dtByVdTv);
        ddNi1dtByVdT -= hsum4(ddNi1dtByVdTv);
        ddNi2dtByVdT -= hsum4(ddNi2dtByVdTv);
        J[i0*(n_) + nSpecies] = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
        J[i1*(n_) + nSpecies] = Wi1ByrhoM_*ddNi1dtByVdT + alphavM*dYi1dt;
        J[i2*(n_) + nSpecies] = Wi2ByrhoM_*ddNi2dtByVdT + alphavM*dYi2dt;
    }
}



void 
OptReaction::ddYdtdTP_Vec_2
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{

    double alphavM = this->invT;
    __m256d alphavMv = _mm256_set1_pd(this->invT);
    for (unsigned int i=0; i<this->nSpecies-2; i=i+4)
    {
        unsigned int i0 = i;
        unsigned int i1 = i+1;
        unsigned int i2 = i+2;
        unsigned int i3 = i+3;

        __m256d Wi0ByrhoM_v = _mm256_loadu_pd(&WiByrhoM[i0]);
        __m256d dYTpdtv = _mm256_loadu_pd(&dYTpdt[i0]);   
        dYTpdtv = _mm256_mul_pd(dYTpdtv,Wi0ByrhoM_v);   
        _mm256_storeu_pd(&dYTpdt[i0],dYTpdtv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i0]);

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];
        const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i2)*(n_)];
        const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i3)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        double ddNi2dtByVdT = JcRowi2[nSpecies];
        double ddNi3dtByVdT = JcRowi3[nSpecies];
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-2; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            sum0 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi0[j]),Cv),alphavMv,sum0);
            sum1 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi1[j]),Cv),alphavMv,sum1);
            sum2 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi2[j]),Cv),alphavMv,sum2);
            sum3 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi3[j]),Cv),alphavMv,sum3);
        }
        ddNi0dtByVdT = ddNi0dtByVdT - (hsum4(sum0));
        ddNi1dtByVdT = ddNi1dtByVdT - (hsum4(sum1));
        ddNi2dtByVdT = ddNi2dtByVdT - (hsum4(sum2));
        ddNi3dtByVdT = ddNi3dtByVdT - (hsum4(sum3));
        {
            unsigned int j0 = nSpecies-2;
            unsigned int j1 = nSpecies-1;
            ddNi0dtByVdT -= JcRowi0[j0]*C[j0]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j0]*C[j0]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j0]*C[j0]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j0]*C[j0]*alphavM;
            ddNi0dtByVdT -= JcRowi0[j1]*C[j1]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j1]*C[j1]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j1]*C[j1]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j1]*C[j1]*alphavM;
        }
        __m256d ddNi0dtByVdTv = _mm256_setr_pd(ddNi0dtByVdT,ddNi1dtByVdT,ddNi2dtByVdT,ddNi3dtByVdT);
        __m256d result = _mm256_fmadd_pd(Wi0ByrhoM_v,ddNi0dtByVdTv,_mm256_mul_pd(alphavMv,dYi0dtv));
        J[i0*(n_) + nSpecies] = get_elem0(result);
        J[i1*(n_) + nSpecies] = get_elem1(result);
        J[i2*(n_) + nSpecies] = get_elem2(result);
        J[i3*(n_) + nSpecies] = get_elem3(result);
    }

    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;


        const double Wi0ByrhoM_ = WiByrhoM[i0];
        const double Wi1ByrhoM_ = WiByrhoM[i1];

        dYTpdt[i0] = dYTpdt[i0]*Wi0ByrhoM_;
        dYTpdt[i1] = dYTpdt[i1]*Wi1ByrhoM_;

        double dYi0dt = dYTpdt[i0];
        double dYi1dt = dYTpdt[i1];

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        __m256d ddNi0dtByVdTv = _mm256_setzero_pd();
        __m256d ddNi1dtByVdTv = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-2; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            __m256d JcRowi0v = _mm256_loadu_pd(&JcRowi0[j+0]);
            ddNi0dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi0v,Cv),alphavMv,ddNi0dtByVdTv);
            __m256d JcRowi1v = _mm256_loadu_pd(&JcRowi1[j+0]);
            ddNi1dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi1v,Cv),alphavMv,ddNi1dtByVdTv);
        }
        {
            unsigned int j = nSpecies-2;
            ddNi0dtByVdT -= ddNdtByVdcTp[i0*(n_) + j+0]*C[j+0]*alphavM;
            ddNi0dtByVdT -= ddNdtByVdcTp[i0*(n_) + j+1]*C[j+1]*alphavM;
            ddNi1dtByVdT -= ddNdtByVdcTp[i1*(n_) + j+0]*C[j+0]*alphavM;
            ddNi1dtByVdT -= ddNdtByVdcTp[i1*(n_) + j+1]*C[j+1]*alphavM;
        }
        ddNi0dtByVdT -= hsum4(ddNi0dtByVdTv);
        ddNi1dtByVdT -= hsum4(ddNi1dtByVdTv);

        J[i0*(n_) + nSpecies] = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
        J[i1*(n_) + nSpecies] = Wi1ByrhoM_*ddNi1dtByVdT + alphavM*dYi1dt;
    }
}


void 
OptReaction::ddYdtdTP_Vec_1
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{

    double alphavM = this->invT;
    __m256d alphavMv = _mm256_set1_pd(this->invT);
    for (unsigned int i=0; i<this->nSpecies-1; i=i+4)
    {
        unsigned int i0 = i;
        unsigned int i1 = i+1;
        unsigned int i2 = i+2;
        unsigned int i3 = i+3;

        __m256d Wi0ByrhoM_v = _mm256_loadu_pd(&WiByrhoM[i0]);
        __m256d dYTpdtv = _mm256_loadu_pd(&dYTpdt[i0]);   
        dYTpdtv = _mm256_mul_pd(dYTpdtv,Wi0ByrhoM_v);   
        _mm256_storeu_pd(&dYTpdt[i0],dYTpdtv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i0]);

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];
        const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i2)*(n_)];
        const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i3)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        double ddNi2dtByVdT = JcRowi2[nSpecies];
        double ddNi3dtByVdT = JcRowi3[nSpecies];

        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();

        for (unsigned int j=0; j<this->nSpecies-1; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            sum0 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi0[j]),Cv),alphavMv,sum0);
            sum1 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi1[j]),Cv),alphavMv,sum1);
            sum2 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi2[j]),Cv),alphavMv,sum2);
            sum3 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi3[j]),Cv),alphavMv,sum3);
        }
        ddNi0dtByVdT = ddNi0dtByVdT - hsum4(sum0);
        ddNi1dtByVdT = ddNi1dtByVdT - hsum4(sum1);
        ddNi2dtByVdT = ddNi2dtByVdT - hsum4(sum2);
        ddNi3dtByVdT = ddNi3dtByVdT - hsum4(sum3);      
        {
            unsigned int j0 = nSpecies-1;
            ddNi0dtByVdT -= JcRowi0[j0]*C[j0]*alphavM;
            ddNi1dtByVdT -= JcRowi1[j0]*C[j0]*alphavM;
            ddNi2dtByVdT -= JcRowi2[j0]*C[j0]*alphavM;
            ddNi3dtByVdT -= JcRowi3[j0]*C[j0]*alphavM;
        }
        __m256d ddNi0dtByVdTv = _mm256_setr_pd(ddNi0dtByVdT,ddNi1dtByVdT,ddNi2dtByVdT,ddNi3dtByVdT);
        __m256d result = _mm256_fmadd_pd(Wi0ByrhoM_v,ddNi0dtByVdTv,_mm256_mul_pd(alphavMv,dYi0dtv));
        J[i0*(n_) + nSpecies] = get_elem0(result);
        J[i1*(n_) + nSpecies] = get_elem1(result);
        J[i2*(n_) + nSpecies] = get_elem2(result);
        J[i3*(n_) + nSpecies] = get_elem3(result);         
    }
    {
        unsigned int i0 = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        dYTpdt[i0] = dYTpdt[i0]*Wi0ByrhoM_;
        double dYi0dt = dYTpdt[i0];
        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[i0*(n_)];
        double ddNi0dtByVdT = JcRowi0[nSpecies];
        __m256d ddNi0dtByVdTv = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-1; j=j+4)
        {
            __m256d JcRowi0v = _mm256_loadu_pd(&JcRowi0[j+0]);
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            ddNi0dtByVdTv = _mm256_fmadd_pd(_mm256_mul_pd(JcRowi0v,Cv),alphavMv,ddNi0dtByVdTv);
        }
        {
            unsigned int j = this->nSpecies-1;
            ddNi0dtByVdT = ddNi0dtByVdT - JcRowi0[j]*C[j]*alphavM - hsum4(ddNi0dtByVdTv);
        }
        
        J[i0*(n_) + nSpecies] = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
    }
}


void 
OptReaction::ddYdtdTP_Vec_0
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{

    __m256d alphavMv = _mm256_set1_pd(this->invT);
    for (unsigned int i=0; i<this->nSpecies; i=i+4)
    {
        unsigned int i0 = i;
        unsigned int i1 = i+1;
        unsigned int i2 = i+2;
        unsigned int i3 = i+3;

        __m256d Wi0ByrhoM_v = _mm256_loadu_pd(&WiByrhoM[i0]);
        __m256d dYTpdtv = _mm256_loadu_pd(&dYTpdt[i0]);   
        dYTpdtv = _mm256_mul_pd(dYTpdtv,Wi0ByrhoM_v);   
        _mm256_storeu_pd(&dYTpdt[i0],dYTpdtv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i0]);

        const double* __restrict__ JcRowi0 = &ddNdtByVdcTp[(i0)*(n_)];
        const double* __restrict__ JcRowi1 = &ddNdtByVdcTp[(i1)*(n_)];
        const double* __restrict__ JcRowi2 = &ddNdtByVdcTp[(i2)*(n_)];
        const double* __restrict__ JcRowi3 = &ddNdtByVdcTp[(i3)*(n_)];

        double ddNi0dtByVdT = JcRowi0[nSpecies];
        double ddNi1dtByVdT = JcRowi1[nSpecies];
        double ddNi2dtByVdT = JcRowi2[nSpecies];
        double ddNi3dtByVdT = JcRowi3[nSpecies];
        __m256d sum0 = _mm256_setzero_pd();
        __m256d sum1 = _mm256_setzero_pd();
        __m256d sum2 = _mm256_setzero_pd();
        __m256d sum3 = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies; j=j+4)
        {
            __m256d Cv = _mm256_loadu_pd(&C[j+0]);
            sum0 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi0[j]),Cv),alphavMv,sum0);
            sum1 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi1[j]),Cv),alphavMv,sum1);
            sum2 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi2[j]),Cv),alphavMv,sum2);
            sum3 =_mm256_fmadd_pd(_mm256_mul_pd(_mm256_loadu_pd(&JcRowi3[j]),Cv),alphavMv,sum3);
        }

        ddNi0dtByVdT = ddNi0dtByVdT - (hsum4(sum0));
        ddNi1dtByVdT = ddNi1dtByVdT - (hsum4(sum1));
        ddNi2dtByVdT = ddNi2dtByVdT - (hsum4(sum2));
        ddNi3dtByVdT = ddNi3dtByVdT - (hsum4(sum3));

        __m256d ddNi0dtByVdTv = _mm256_setr_pd(ddNi0dtByVdT,ddNi1dtByVdT,ddNi2dtByVdT,ddNi3dtByVdT);
        __m256d result = _mm256_fmadd_pd(Wi0ByrhoM_v,ddNi0dtByVdTv,_mm256_mul_pd(alphavMv,dYi0dtv));

        J[i0*(n_) + nSpecies] = get_elem0(result);
        J[i1*(n_) + nSpecies] = get_elem1(result);
        J[i2*(n_) + nSpecies] = get_elem2(result);
        J[i3*(n_) + nSpecies] = get_elem3(result);
    }
}

void 
OptReaction::ddYdtdTP
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ C,  
    double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    double alphavM = this->invT;

    for (unsigned int i=0; i<this->nSpecies; i=i+1)
    {
        unsigned int i0 = i;
        const double Wi0ByrhoM_ = WiByrhoM[i0];
        double& dYi0dt = dYTpdt[i0];
        dYi0dt *= Wi0ByrhoM_;
        double ddNi0dtByVdT = ddNdtByVdcTp[(i0)*(n_) + nSpecies];
        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddNi0dtByVdcj = ddNdtByVdcTp[i0*(n_) + j];
            ddNi0dtByVdT -= ddNi0dtByVdcj*C[j]*alphavM;                                 
        }

        double& ddYi0dtdT = J[i0*(n_) + nSpecies];
        ddYi0dtdT = Wi0ByrhoM_*ddNi0dtByVdT + alphavM*dYi0dt;
     
    }
}

void 
OptReaction::ddTdtdYT
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    double& dTdt = dYTpdt[this->nSpecies];
    dTdt = 0;

    for (unsigned int i=0; i<this->nSpecies; i++)
    {
        dTdt -= dYTpdt[i]*Ha[i];
    }

    dTdt /= CpM;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    for (unsigned int i=0; i<this->nSpecies; i++)
    {
        double& ddTdtdYi = J[this->nSpecies *(n_)+ i];
        ddTdtdYi = 0;

        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddYjdtdYi = J[j *(n_)+ i];
            ddTdtdYi -= ddYjdtdYi*Ha[j];
        }
        ddTdtdYi -= Cp[i]*dTdt;
        ddTdtdYi /= CpM;

        const double dYidt = dYTpdt[i];
        const double ddYidtdT = J[i *(n_)+ this->nSpecies];

        ddTdtdT -= dYidt*Cp[i] + ddYidtdT*Ha[i];
    }

    ddTdtdT -= dTdt*dCpMdT;
    ddTdtdT /= CpM;   
}


void 
OptReaction::ddTdtdYT_Vec
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    unsigned int remain = this->nSpecies%4;

    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    double& dTdt = dYTpdt[this->nSpecies];
    dTdt = 0;

    for (unsigned int i=0; i<this->nSpecies-remain; i=i+4)
    {
        dTdt -= dYTpdt[i+0]*Ha[i+0];
        dTdt -= dYTpdt[i+1]*Ha[i+1];
        dTdt -= dYTpdt[i+2]*Ha[i+2];
        dTdt -= dYTpdt[i+3]*Ha[i+3];                       
    }
    if(remain==1)
    {
        unsigned int i = this->nSpecies-1;
        dTdt -= dYTpdt[i+0]*Ha[i+0];       
    }
    else if(remain==2)
    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;
        dTdt -= dYTpdt[i0]*Ha[i0];   
        dTdt -= dYTpdt[i1]*Ha[i1];
    }
    else if(remain==3)
    {
        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;        
        dTdt -= dYTpdt[i0]*Ha[i0];   
        dTdt -= dYTpdt[i1]*Ha[i1];
        dTdt -= dYTpdt[i2]*Ha[i2];        
    }
    dTdt /= CpM;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    for (unsigned int i=0; i<this->nSpecies-remain; i=i+4)
    {
        double& ddTdtdYi0 = J[this->nSpecies *(n_)+ i+0];
        double& ddTdtdYi1 = J[this->nSpecies *(n_)+ i+1];
        double& ddTdtdYi2 = J[this->nSpecies *(n_)+ i+2];
        double& ddTdtdYi3 = J[this->nSpecies *(n_)+ i+3];                        
        ddTdtdYi0 = 0;
        ddTdtdYi1 = 0;
        ddTdtdYi2 = 0;
        ddTdtdYi3 = 0;
        __m256d ddTdtdYi = _mm256_set1_pd(0.0);
        for (unsigned int j=0; j<this->nSpecies-remain; j=j+4)
        {
            const double ddYj0dtdYi0 = J[(j+0) *(n_)+ (i+0)];
            const double ddYj0dtdYi1 = J[(j+0) *(n_)+ (i+1)];
            const double ddYj0dtdYi2 = J[(j+0) *(n_)+ (i+2)];
            const double ddYj0dtdYi3 = J[(j+0) *(n_)+ (i+3)];
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);

            const double ddYj1dtdYi0 = J[(j+1) *(n_)+ (i+0)];
            const double ddYj1dtdYi1 = J[(j+1) *(n_)+ (i+1)];
            const double ddYj1dtdYi2 = J[(j+1) *(n_)+ (i+2)];
            const double ddYj1dtdYi3 = J[(j+1) *(n_)+ (i+3)];
            __m256d ddYj1dtdYi = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);

            const double ddYj2dtdYi0 = J[(j+2) *(n_)+ (i+0)];
            const double ddYj2dtdYi1 = J[(j+2) *(n_)+ (i+1)];
            const double ddYj2dtdYi2 = J[(j+2) *(n_)+ (i+2)];
            const double ddYj2dtdYi3 = J[(j+2) *(n_)+ (i+3)];
            __m256d ddYj2dtdYi = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);

            const double ddYj3dtdYi0 = J[(j+3) *(n_)+ (i+0)];
            const double ddYj3dtdYi1 = J[(j+3) *(n_)+ (i+1)];
            const double ddYj3dtdYi2 = J[(j+3) *(n_)+ (i+2)];
            const double ddYj3dtdYi3 = J[(j+3) *(n_)+ (i+3)]; 
            __m256d ddYj3dtdYi = _mm256_loadu_pd(&J[(j+3) *(n_)+ (i+0)]);

            ddTdtdYi0 -= ddYj0dtdYi0*Ha[j+0];
            ddTdtdYi1 -= ddYj0dtdYi1*Ha[j+0];
            ddTdtdYi2 -= ddYj0dtdYi2*Ha[j+0];
            ddTdtdYi3 -= ddYj0dtdYi3*Ha[j+0];
            ddTdtdYi = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j+0]),ddTdtdYi);

            ddTdtdYi0 -= ddYj1dtdYi0*Ha[j+1];
            ddTdtdYi1 -= ddYj1dtdYi1*Ha[j+1];
            ddTdtdYi2 -= ddYj1dtdYi2*Ha[j+1];
            ddTdtdYi3 -= ddYj1dtdYi3*Ha[j+1];
            ddTdtdYi = _mm256_fmadd_pd(ddYj1dtdYi,_mm256_set1_pd(-Ha[j+1]),ddTdtdYi);

            ddTdtdYi0 -= ddYj2dtdYi0*Ha[j+2];
            ddTdtdYi1 -= ddYj2dtdYi1*Ha[j+2];
            ddTdtdYi2 -= ddYj2dtdYi2*Ha[j+2];
            ddTdtdYi3 -= ddYj2dtdYi3*Ha[j+2];
            ddTdtdYi = _mm256_fmadd_pd(ddYj2dtdYi,_mm256_set1_pd(-Ha[j+2]),ddTdtdYi);

            ddTdtdYi0 -= ddYj3dtdYi0*Ha[j+3];
            ddTdtdYi1 -= ddYj3dtdYi1*Ha[j+3];  
            ddTdtdYi2 -= ddYj3dtdYi2*Ha[j+3];  
            ddTdtdYi3 -= ddYj3dtdYi3*Ha[j+3];   
            ddTdtdYi = _mm256_fmadd_pd(ddYj3dtdYi,_mm256_set1_pd(-Ha[j+3]),ddTdtdYi);                     
        }
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double ddYj0dtdYi0 = J[(j+0) *(n_)+ (i+0)];
            const double ddYj0dtdYi1 = J[(j+0) *(n_)+ (i+1)];
            const double ddYj0dtdYi2 = J[(j+0) *(n_)+ (i+2)];
            const double ddYj0dtdYi3 = J[(j+0) *(n_)+ (i+3)];
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);

            ddTdtdYi0 -= ddYj0dtdYi0*Ha[j+0];
            ddTdtdYi1 -= ddYj0dtdYi1*Ha[j+0];
            ddTdtdYi2 -= ddYj0dtdYi2*Ha[j+0];
            ddTdtdYi3 -= ddYj0dtdYi3*Ha[j+0];
            ddTdtdYi = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j+0]),ddTdtdYi);             
        }
        else if(remain==2)
        {
            unsigned int j0 = this->nSpecies-2;
            unsigned int j1 = this->nSpecies-1;
            const double ddYj0dtdYi0 = J[(j0) *(n_)+ (i+0)];
            const double ddYj0dtdYi1 = J[(j0) *(n_)+ (i+1)];
            const double ddYj0dtdYi2 = J[(j0) *(n_)+ (i+2)];
            const double ddYj0dtdYi3 = J[(j0) *(n_)+ (i+3)];
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j0) *(n_)+ (i+0)]);

            const double ddYj1dtdYi0 = J[(j1) *(n_)+ (i+0)];
            const double ddYj1dtdYi1 = J[(j1) *(n_)+ (i+1)];
            const double ddYj1dtdYi2 = J[(j1) *(n_)+ (i+2)];
            const double ddYj1dtdYi3 = J[(j1) *(n_)+ (i+3)];
            __m256d ddYj1dtdYi = _mm256_loadu_pd(&J[(j1) *(n_)+ (i+0)]);

            ddTdtdYi0 -= ddYj0dtdYi0*Ha[j0];
            ddTdtdYi1 -= ddYj0dtdYi1*Ha[j0];
            ddTdtdYi2 -= ddYj0dtdYi2*Ha[j0];
            ddTdtdYi3 -= ddYj0dtdYi3*Ha[j0];
            ddTdtdYi = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j0]),ddTdtdYi);

            ddTdtdYi0 -= ddYj1dtdYi0*Ha[j1];
            ddTdtdYi1 -= ddYj1dtdYi1*Ha[j1];  
            ddTdtdYi2 -= ddYj1dtdYi2*Ha[j1];
            ddTdtdYi3 -= ddYj1dtdYi3*Ha[j1];    
            ddTdtdYi = _mm256_fmadd_pd(ddYj1dtdYi,_mm256_set1_pd(-Ha[j1]),ddTdtdYi);                    
        }
        else if(remain==3)
        {
            unsigned int j0 =this->nSpecies-3;            
            unsigned int j1 =this->nSpecies-2;            
            unsigned int j2 =this->nSpecies-1;

            const double ddYj0dtdYi0 = J[(j0) *(n_)+ (i+0)];
            const double ddYj0dtdYi1 = J[(j0) *(n_)+ (i+1)];
            const double ddYj0dtdYi2 = J[(j0) *(n_)+ (i+2)];
            const double ddYj0dtdYi3 = J[(j0) *(n_)+ (i+3)];
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j0) *(n_)+ (i+0)]);

            const double ddYj1dtdYi0 = J[(j1) *(n_)+ (i+0)];
            const double ddYj1dtdYi1 = J[(j1) *(n_)+ (i+1)];
            const double ddYj1dtdYi2 = J[(j1) *(n_)+ (i+2)];
            const double ddYj1dtdYi3 = J[(j1) *(n_)+ (i+3)];
            __m256d ddYj1dtdYi = _mm256_loadu_pd(&J[(j1) *(n_)+ (i+0)]);

            const double ddYj2dtdYi0 = J[(j2) *(n_)+ (i+0)];
            const double ddYj2dtdYi1 = J[(j2) *(n_)+ (i+1)];
            const double ddYj2dtdYi2 = J[(j2) *(n_)+ (i+2)];
            const double ddYj2dtdYi3 = J[(j2) *(n_)+ (i+3)];
            __m256d ddYj2dtdYi = _mm256_loadu_pd(&J[(j2) *(n_)+ (i+0)]);

            ddTdtdYi0 -= ddYj0dtdYi0*Ha[j0];
            ddTdtdYi1 -= ddYj0dtdYi1*Ha[j0];
            ddTdtdYi2 -= ddYj0dtdYi2*Ha[j0];
            ddTdtdYi3 -= ddYj0dtdYi3*Ha[j0];
            ddTdtdYi = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j0]),ddTdtdYi);

            ddTdtdYi0 -= ddYj1dtdYi0*Ha[j1];
            ddTdtdYi1 -= ddYj1dtdYi1*Ha[j1];
            ddTdtdYi2 -= ddYj1dtdYi2*Ha[j1];
            ddTdtdYi3 -= ddYj1dtdYi3*Ha[j1];
            ddTdtdYi = _mm256_fmadd_pd(ddYj1dtdYi,_mm256_set1_pd(-Ha[j1]),ddTdtdYi);

            ddTdtdYi0 -= ddYj2dtdYi0*Ha[j2];
            ddTdtdYi1 -= ddYj2dtdYi1*Ha[j2]; 
            ddTdtdYi2 -= ddYj2dtdYi2*Ha[j2]; 
            ddTdtdYi3 -= ddYj2dtdYi3*Ha[j2];    
            ddTdtdYi = _mm256_fmadd_pd(ddYj2dtdYi,_mm256_set1_pd(-Ha[j2]),ddTdtdYi);

        }
        ddTdtdYi0 -= Cp[i+0]*dTdt;
        ddTdtdYi1 -= Cp[i+1]*dTdt;
        ddTdtdYi2 -= Cp[i+2]*dTdt;
        ddTdtdYi3 -= Cp[i+3]*dTdt;

        ddTdtdYi0 /= CpM;
        ddTdtdYi1 /= CpM;
        ddTdtdYi2 /= CpM;
        ddTdtdYi3 /= CpM; 

        const double dYi0dt = dYTpdt[i+0];
        const double dYi1dt = dYTpdt[i+1];
        const double dYi2dt = dYTpdt[i+2];
        const double dYi3dt = dYTpdt[i+3];

        const double ddYi0dtdT = J[(i+0) *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[(i+1) *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[(i+2) *(n_)+ this->nSpecies];
        const double ddYi3dtdT = J[(i+3) *(n_)+ this->nSpecies];                

        ddTdtdT -= dYi0dt*Cp[i+0] + ddYi0dtdT*Ha[i+0];
        ddTdtdT -= dYi1dt*Cp[i+1] + ddYi1dtdT*Ha[i+1];
        ddTdtdT -= dYi2dt*Cp[i+2] + ddYi2dtdT*Ha[i+2];
        ddTdtdT -= dYi3dt*Cp[i+3] + ddYi3dtdT*Ha[i+3];   
            
    }
    if(remain==1)
    {
        unsigned int i = this->nSpecies-1;
        double& ddTdtdYi = J[this->nSpecies *(n_)+ i];
        ddTdtdYi = 0;
    
        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddYjdtdYi = J[j *(n_)+ i];
            ddTdtdYi -= ddYjdtdYi*Ha[j];
        }
        ddTdtdYi -= Cp[i]*dTdt;
        ddTdtdYi /= CpM;
    
        const double dYidt = dYTpdt[i];
        const double ddYidtdT = J[i *(n_)+ this->nSpecies];
    
        ddTdtdT -= dYidt*Cp[i] + ddYidtdT*Ha[i];
    
    }
    else if(remain==2)
    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;

        double& ddTdtdYi0 = J[this->nSpecies *(n_)+ i0];
        double& ddTdtdYi1 = J[this->nSpecies *(n_)+ i1];        
        ddTdtdYi0 = 0;
        ddTdtdYi1 = 0;   

        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddYjdtdYi0 = J[j *(n_)+ i0];
            const double ddYjdtdYi1 = J[j *(n_)+ i1];

            ddTdtdYi0 -= ddYjdtdYi0*Ha[j];
            ddTdtdYi1 -= ddYjdtdYi1*Ha[j];            
        }
        ddTdtdYi0 -= Cp[i0]*dTdt;
        ddTdtdYi1 -= Cp[i1]*dTdt;

        ddTdtdYi0 /= CpM;
        ddTdtdYi1 /= CpM;

        const double dYi0dt = dYTpdt[i0];
        const double dYi1dt = dYTpdt[i1];

        const double ddYi0dtdT = J[i0 *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[i1 *(n_)+ this->nSpecies];

        ddTdtdT -= dYi0dt*Cp[i0] + ddYi0dtdT*Ha[i0];
        ddTdtdT -= dYi1dt*Cp[i1] + ddYi1dtdT*Ha[i1];        
    }
    else if(remain==3)
    {

        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;

        double& ddTdtdYi0 = J[this->nSpecies *(n_)+ i0];
        double& ddTdtdYi1 = J[this->nSpecies *(n_)+ i1];
        double& ddTdtdYi2 = J[this->nSpecies *(n_)+ i2];

        ddTdtdYi0 = 0;
        ddTdtdYi1 = 0;
        ddTdtdYi2 = 0;

        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double ddYjdtdYi0 = J[j *(n_)+ i0];
            const double ddYjdtdYi1 = J[j *(n_)+ i1];
            const double ddYjdtdYi2 = J[j *(n_)+ i2];

            ddTdtdYi0 -= ddYjdtdYi0*Ha[j];
            ddTdtdYi1 -= ddYjdtdYi1*Ha[j];
            ddTdtdYi2 -= ddYjdtdYi2*Ha[j];                        
        }

        ddTdtdYi0 -= Cp[i0]*dTdt;
        ddTdtdYi1 -= Cp[i1]*dTdt;
        ddTdtdYi2 -= Cp[i2]*dTdt;

        ddTdtdYi0 /= CpM;
        ddTdtdYi1 /= CpM;
        ddTdtdYi2 /= CpM;  

        const double dYi0dt = dYTpdt[i0];
        const double dYi1dt = dYTpdt[i1];
        const double dYi2dt = dYTpdt[i2];

        const double ddYi0dtdT = J[i0 *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[i1 *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[i2 *(n_)+ this->nSpecies];

        ddTdtdT -= dYi0dt*Cp[i0] + ddYi0dtdT*Ha[i0];
        ddTdtdT -= dYi1dt*Cp[i1] + ddYi1dtdT*Ha[i1];
        ddTdtdT -= dYi2dt*Cp[i2] + ddYi2dtdT*Ha[i2];                
    }
    ddTdtdT -= dTdt*dCpMdT;
    ddTdtdT /= CpM;   
}

void 
OptReaction::ddTdtdYT_Vec_0
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    const double invCpM = 1.0/CpM;
    __m256d dTdtv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies; i=i+4)
    {
        dTdtv = _mm256_fmadd_pd(-_mm256_loadu_pd(&Ha[i+0]),_mm256_loadu_pd(&dYTpdt[i+0]),dTdtv);
    }

    double dTdt = hsum4(dTdtv);    
    dTdt *= invCpM;
    dYTpdt[this->nSpecies] = dTdt;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    __m256d ddTdtdTv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies; i=i+4)
    {
        __m256d ddTdtdYiv = _mm256_set1_pd(0.0);
        for (unsigned int j=0; j<this->nSpecies; j=j+4)
        {
            __m256d ddYj0dtdYiv = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYiv = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            __m256d ddYj2dtdYiv = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);
            __m256d ddYj3dtdYiv = _mm256_loadu_pd(&J[(j+3) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYiv,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYiv,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj2dtdYiv,_mm256_set1_pd(-Ha[j+2]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj3dtdYiv,_mm256_set1_pd(-Ha[j+3]),ddTdtdYiv);                     
        }
        ddTdtdYiv = _mm256_fmadd_pd(_mm256_loadu_pd(&Cp[i+0]),_mm256_set1_pd(-dTdt),ddTdtdYiv);
        ddTdtdYiv =_mm256_mul_pd(ddTdtdYiv,_mm256_set1_pd(invCpM));
        _mm256_storeu_pd(&J[this->nSpecies *(n_)+ i+0],ddTdtdYiv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i+0]);
        const double ddYi0dtdT = J[(i+0) *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[(i+1) *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[(i+2) *(n_)+ this->nSpecies];
        const double ddYi3dtdT = J[(i+3) *(n_)+ this->nSpecies];
        __m256d ddYi0dtdTv = _mm256_setr_pd(ddYi0dtdT,ddYi1dtdT,ddYi2dtdT,ddYi3dtdT);
        ddTdtdTv = _mm256_fmadd_pd(dYi0dtv,_mm256_loadu_pd(&Cp[i+0]),ddTdtdTv);
        ddTdtdTv = _mm256_fmadd_pd(ddYi0dtdTv,_mm256_loadu_pd(&Ha[i+0]),ddTdtdTv);
    }

    ddTdtdT = ddTdtdT - hsum4(ddTdtdTv);
    ddTdtdT -= dTdt*dCpMdT; 
    ddTdtdT *= invCpM;
}

void 
OptReaction::ddTdtdYT_Vec_1
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{
    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    const double invCpM = 1.0/CpM;
    __m256d dTdtv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-1; i=i+4)
    {
        dTdtv = _mm256_fmadd_pd(-_mm256_loadu_pd(&Ha[i+0]),_mm256_loadu_pd(&dYTpdt[i+0]),dTdtv);
    }
    double dTdt = hsum4(dTdtv);
    {
        unsigned int i = this->nSpecies-1;
        dTdt -= dYTpdt[i+0]*Ha[i+0];       
    }
    dTdt *= invCpM;
    dYTpdt[this->nSpecies] = dTdt;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    __m256d ddTdtdTv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-1; i=i+4)
    {
        __m256d ddTdtdYiv = _mm256_set1_pd(0.0);
        for (unsigned int j=0; j<this->nSpecies-1; j=j+4)
        {
            __m256d ddYj0dtdYiv = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYiv = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            __m256d ddYj2dtdYiv = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);
            __m256d ddYj3dtdYiv = _mm256_loadu_pd(&J[(j+3) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYiv,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYiv,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj2dtdYiv,_mm256_set1_pd(-Ha[j+2]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj3dtdYiv,_mm256_set1_pd(-Ha[j+3]),ddTdtdYiv);                     
        }
        {
            unsigned int j = this->nSpecies-1;
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);             
        }
        ddTdtdYiv = _mm256_fmadd_pd(_mm256_loadu_pd(&Cp[i+0]),_mm256_set1_pd(-dTdt),ddTdtdYiv);
        ddTdtdYiv =_mm256_mul_pd(ddTdtdYiv,_mm256_set1_pd(invCpM));
        _mm256_storeu_pd(&J[this->nSpecies *(n_)+ i+0],ddTdtdYiv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i+0]);

        const double ddYi0dtdT = J[(i+0) *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[(i+1) *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[(i+2) *(n_)+ this->nSpecies];
        const double ddYi3dtdT = J[(i+3) *(n_)+ this->nSpecies];
        __m256d ddYi0dtdTv = _mm256_setr_pd(ddYi0dtdT,ddYi1dtdT,ddYi2dtdT,ddYi3dtdT);

        ddTdtdTv = _mm256_fmadd_pd(dYi0dtv,_mm256_loadu_pd(&Cp[i+0]),ddTdtdTv);
        ddTdtdTv = _mm256_fmadd_pd(ddYi0dtdTv,_mm256_loadu_pd(&Ha[i+0]),ddTdtdTv);
    }  
    ddTdtdT = ddTdtdT - hsum4(ddTdtdTv);

    {
        unsigned int i = this->nSpecies-1;
        double& ddTdtdYi = J[this->nSpecies *(n_)+ i];
        ddTdtdYi = 0;
        __m256d ddTdtdYiv = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-1; j=j+4)
        {
            __m256d Av = _mm256_setr_pd(J[(j+0) *(n_)+ i],J[(j+1) *(n_)+ i],J[(j+2) *(n_)+ i],J[(j+3) *(n_)+ i]);
            __m256d Hav = _mm256_loadu_pd(&Ha[j+0]);
            ddTdtdYiv = _mm256_fmadd_pd(Av,Hav,ddTdtdYiv);
        }
        {
            ddTdtdYi = ddTdtdYi - hsum4(ddTdtdYiv);
            unsigned int j = nSpecies-1;
            ddTdtdYi -= J[(j+0) *(n_)+ i]*Ha[j+0];
        }

        ddTdtdYi -= Cp[i]*dTdt;
        ddTdtdYi *= invCpM;
        const double dYidt = dYTpdt[i];
        const double ddYidtdT = J[i *(n_)+ this->nSpecies];
        ddTdtdT -= dYidt*Cp[i] + ddYidtdT*Ha[i];
    }
    ddTdtdT -= dTdt*dCpMdT;
    ddTdtdT *= invCpM;   
}

void 
OptReaction::ddTdtdYT_Vec_2
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{

    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    const double invCpM = 1.0/CpM;
    __m256d dTdtv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-2; i=i+4)
    {
        dTdtv = _mm256_fmadd_pd(-_mm256_loadu_pd(&Ha[i+0]),_mm256_loadu_pd(&dYTpdt[i+0]),dTdtv);
    }
    double dTdt = hsum4(dTdtv);
    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;
        dTdt -= dYTpdt[i0]*Ha[i0];
        dTdt -= dYTpdt[i1]*Ha[i1];
    }
    dTdt *= invCpM;
    dYTpdt[this->nSpecies] = dTdt;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    __m256d ddTdtdTv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-2; i=i+4)
    {
        __m256d ddTdtdYiv = _mm256_set1_pd(0.0);
        for (unsigned int j=0; j<this->nSpecies-2; j=j+4)
        {
            __m256d ddYj0dtdYiv = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYiv = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            __m256d ddYj2dtdYiv = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);
            __m256d ddYj3dtdYiv = _mm256_loadu_pd(&J[(j+3) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYiv,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYiv,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj2dtdYiv,_mm256_set1_pd(-Ha[j+2]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj3dtdYiv,_mm256_set1_pd(-Ha[j+3]),ddTdtdYiv);                     
        }
        {
            unsigned int j = this->nSpecies-2;
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYi = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYi,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
        }
        ddTdtdYiv = _mm256_fmadd_pd(_mm256_loadu_pd(&Cp[i+0]),_mm256_set1_pd(-dTdt),ddTdtdYiv);
        ddTdtdYiv =_mm256_mul_pd(ddTdtdYiv,_mm256_set1_pd(invCpM));
        _mm256_storeu_pd(&J[this->nSpecies *(n_)+ i+0],ddTdtdYiv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i+0]);

        const double ddYi0dtdT = J[(i+0) *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[(i+1) *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[(i+2) *(n_)+ this->nSpecies];
        const double ddYi3dtdT = J[(i+3) *(n_)+ this->nSpecies];
        __m256d ddYi0dtdTv = _mm256_setr_pd(ddYi0dtdT,ddYi1dtdT,ddYi2dtdT,ddYi3dtdT);
        ddTdtdTv = _mm256_fmadd_pd(dYi0dtv,_mm256_loadu_pd(&Cp[i+0]),ddTdtdTv);
        ddTdtdTv = _mm256_fmadd_pd(ddYi0dtdTv,_mm256_loadu_pd(&Ha[i+0]),ddTdtdTv);
    }
    ddTdtdT = ddTdtdT - hsum4(ddTdtdTv);
    {
        unsigned int i0 = this->nSpecies-2;
        unsigned int i1 = this->nSpecies-1;
        double& ddTdtdYi0 = J[this->nSpecies *(n_)+ i0];
        double& ddTdtdYi1 = J[this->nSpecies *(n_)+ i1];        
        ddTdtdYi0 = 0;
        ddTdtdYi1 = 0;   
        __m256d result0 = _mm256_setzero_pd();
        __m256d result1 = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-2; j=j+4)
        {
            __m256d A0 = _mm256_setr_pd(J[(j+0) *(n_)+ i0],J[(j+1) *(n_)+ i0],J[(j+2) *(n_)+ i0],J[(j+3) *(n_)+ i0]);
            __m256d A1 = _mm256_setr_pd(J[(j+0) *(n_)+ i1],J[(j+1) *(n_)+ i1],J[(j+2) *(n_)+ i1],J[(j+3) *(n_)+ i1]);
            __m256d Hav = _mm256_loadu_pd(&Ha[j+0]);
            result0 = _mm256_fmadd_pd(A0,Hav,result0);
            result1 = _mm256_fmadd_pd(A1,Hav,result1);
        }
        {
            ddTdtdYi0 = ddTdtdYi0 - hsum4(result0);
            ddTdtdYi1 = ddTdtdYi1 - hsum4(result1);
            unsigned int j = nSpecies-2;
            ddTdtdYi0 -= J[(j+0) *(n_)+ i0]*Ha[j+0];
            ddTdtdYi0 -= J[(j+1) *(n_)+ i0]*Ha[j+1];
            ddTdtdYi1 -= J[(j+0) *(n_)+ i1]*Ha[j+0];
            ddTdtdYi1 -= J[(j+1) *(n_)+ i1]*Ha[j+1];
        }
        ddTdtdYi0 -= Cp[i0]*dTdt;
        ddTdtdYi1 -= Cp[i1]*dTdt;

        ddTdtdYi0 = ddTdtdYi0*invCpM;
        ddTdtdYi1 = ddTdtdYi1*invCpM;

        const double dYi0dt = dYTpdt[i0];
        const double dYi1dt = dYTpdt[i1];

        const double ddYi0dtdT = J[i0 *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[i1 *(n_)+ this->nSpecies];

        ddTdtdT -= dYi0dt*Cp[i0] + ddYi0dtdT*Ha[i0];
        ddTdtdT -= dYi1dt*Cp[i1] + ddYi1dtdT*Ha[i1];        
    }
    ddTdtdT -= dTdt*dCpMdT;
    ddTdtdT = ddTdtdT*invCpM;   
}

void 
OptReaction::ddTdtdYT_Vec_3
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,
    const double* __restrict__ Cp,
    const double* __restrict__ dCpdT,
    const double* __restrict__ Ha,      
    double* __restrict__ dYTpdt,  
    const double* __restrict__ YTp, 
    double* __restrict__ J
) const noexcept
{

    const double& CpM = Cp[this->nSpecies];
    const double& dCpMdT = dCpdT[this->nSpecies];
    const double invCpM = 1.0/CpM;
    __m256d dTdtv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-3; i=i+4)
    {
        dTdtv = _mm256_fmadd_pd(-_mm256_loadu_pd(&Ha[i+0]),_mm256_loadu_pd(&dYTpdt[i+0]),dTdtv);
    }
    double dTdt = hsum4(dTdtv);
    {
        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;
        dTdt -= dYTpdt[i0]*Ha[i0];
        dTdt -= dYTpdt[i1]*Ha[i1];
        dTdt -= dYTpdt[i2]*Ha[i2];
    }
    dTdt *= invCpM;
    dYTpdt[this->nSpecies] = dTdt;
    double& ddTdtdT = J[this->nSpecies *(n_)+ this->nSpecies];
    ddTdtdT = 0;
    __m256d ddTdtdTv = _mm256_setzero_pd();
    for (unsigned int i=0; i<this->nSpecies-3; i=i+4)
    {
        __m256d ddTdtdYiv = _mm256_set1_pd(0.0);
        for (unsigned int j=0; j<this->nSpecies-3; j=j+4)
        {
            __m256d ddYj0dtdYiv = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYiv = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            __m256d ddYj2dtdYiv = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);
            __m256d ddYj3dtdYiv = _mm256_loadu_pd(&J[(j+3) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYiv,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYiv,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj2dtdYiv,_mm256_set1_pd(-Ha[j+2]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj3dtdYiv,_mm256_set1_pd(-Ha[j+3]),ddTdtdYiv);                     
        }
        {
            unsigned int j = this->nSpecies-3;
            __m256d ddYj0dtdYi = _mm256_loadu_pd(&J[(j+0) *(n_)+ (i+0)]);
            __m256d ddYj1dtdYi = _mm256_loadu_pd(&J[(j+1) *(n_)+ (i+0)]);
            __m256d ddYj2dtdYi = _mm256_loadu_pd(&J[(j+2) *(n_)+ (i+0)]);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj0dtdYi,_mm256_set1_pd(-Ha[j+0]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj1dtdYi,_mm256_set1_pd(-Ha[j+1]),ddTdtdYiv);
            ddTdtdYiv = _mm256_fmadd_pd(ddYj2dtdYi,_mm256_set1_pd(-Ha[j+2]),ddTdtdYiv);
        }
        ddTdtdYiv = _mm256_fmadd_pd(_mm256_loadu_pd(&Cp[i+0]),_mm256_set1_pd(-dTdt),ddTdtdYiv);
        ddTdtdYiv =_mm256_mul_pd(ddTdtdYiv,_mm256_set1_pd(invCpM));
        _mm256_storeu_pd(&J[this->nSpecies *(n_)+ i+0],ddTdtdYiv);
        __m256d dYi0dtv = _mm256_loadu_pd(&dYTpdt[i+0]);

        const double ddYi0dtdT = J[(i+0) *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[(i+1) *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[(i+2) *(n_)+ this->nSpecies];
        const double ddYi3dtdT = J[(i+3) *(n_)+ this->nSpecies];
        __m256d ddYi0dtdTv = _mm256_setr_pd(ddYi0dtdT,ddYi1dtdT,ddYi2dtdT,ddYi3dtdT);

        ddTdtdTv = _mm256_fmadd_pd(dYi0dtv,_mm256_loadu_pd(&Cp[i+0]),ddTdtdTv);
        ddTdtdTv = _mm256_fmadd_pd(ddYi0dtdTv,_mm256_loadu_pd(&Ha[i+0]),ddTdtdTv);
    }
   
    ddTdtdT = ddTdtdT - hsum4(ddTdtdTv);
    {
        unsigned int i0 = this->nSpecies-3;
        unsigned int i1 = this->nSpecies-2;
        unsigned int i2 = this->nSpecies-1;

        double& ddTdtdYi0 = J[this->nSpecies *(n_)+ i0];
        double& ddTdtdYi1 = J[this->nSpecies *(n_)+ i1];  
        double& ddTdtdYi2 = J[this->nSpecies *(n_)+ i2];                
        ddTdtdYi0 = 0;
        ddTdtdYi1 = 0;   
        ddTdtdYi2 = 0; 
        __m256d result0 = _mm256_setzero_pd();
        __m256d result1 = _mm256_setzero_pd();
        __m256d result2 = _mm256_setzero_pd();
        for (unsigned int j=0; j<this->nSpecies-3; j=j+4)
        {
            __m256d A0 = _mm256_setr_pd(J[(j+0) *(n_)+ i0],J[(j+1) *(n_)+ i0],J[(j+2) *(n_)+ i0],J[(j+3) *(n_)+ i0]);
            __m256d A1 = _mm256_setr_pd(J[(j+0) *(n_)+ i1],J[(j+1) *(n_)+ i1],J[(j+2) *(n_)+ i1],J[(j+3) *(n_)+ i1]);
            __m256d A2 = _mm256_setr_pd(J[(j+0) *(n_)+ i2],J[(j+1) *(n_)+ i2],J[(j+2) *(n_)+ i2],J[(j+3) *(n_)+ i2]);
            __m256d Hav = _mm256_loadu_pd(&Ha[j+0]);
            result0 = _mm256_fmadd_pd(A0,Hav,result0);
            result1 = _mm256_fmadd_pd(A1,Hav,result1);
            result2 = _mm256_fmadd_pd(A2,Hav,result2);
        }
        {

            ddTdtdYi0 = ddTdtdYi0 - hsum4(result0);
            ddTdtdYi1 = ddTdtdYi1 - hsum4(result1);
            ddTdtdYi2 = ddTdtdYi2 - hsum4(result2);

            unsigned int j = nSpecies-3;
            ddTdtdYi0 -= J[(j+0) *(n_)+ i0]*Ha[j+0];
            ddTdtdYi0 -= J[(j+1) *(n_)+ i0]*Ha[j+1];
            ddTdtdYi0 -= J[(j+2) *(n_)+ i0]*Ha[j+2];

            ddTdtdYi1 -= J[(j+0) *(n_)+ i1]*Ha[j+0];
            ddTdtdYi1 -= J[(j+1) *(n_)+ i1]*Ha[j+1];
            ddTdtdYi1 -= J[(j+2) *(n_)+ i1]*Ha[j+2];

            ddTdtdYi2 -= J[(j+0) *(n_)+ i2]*Ha[j+0];
            ddTdtdYi2 -= J[(j+1) *(n_)+ i2]*Ha[j+1];
            ddTdtdYi2 -= J[(j+2) *(n_)+ i2]*Ha[j+2];
        }
        ddTdtdYi0 -= Cp[i0]*dTdt;
        ddTdtdYi1 -= Cp[i1]*dTdt;
        ddTdtdYi2 -= Cp[i2]*dTdt;

        ddTdtdYi0 *= invCpM;
        ddTdtdYi1 *= invCpM;
        ddTdtdYi2 *= invCpM;

        const double dYi0dt = dYTpdt[i0];
        const double dYi1dt = dYTpdt[i1];
        const double dYi2dt = dYTpdt[i2];

        const double ddYi0dtdT = J[i0 *(n_)+ this->nSpecies];
        const double ddYi1dtdT = J[i1 *(n_)+ this->nSpecies];
        const double ddYi2dtdT = J[i2 *(n_)+ this->nSpecies];

        ddTdtdT -= dYi0dt*Cp[i0] + ddYi0dtdT*Ha[i0];
        ddTdtdT -= dYi1dt*Cp[i1] + ddYi1dtdT*Ha[i1];        
        ddTdtdT -= dYi2dt*Cp[i2] + ddYi2dtdT*Ha[i2]; 
    }
    ddTdtdT -= dTdt*dCpMdT;
    ddTdtdT = ddTdtdT*invCpM;   
}

void 
OptReaction::FastddYdtdY
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{
    for(unsigned int i=0; i<this->nSpecies; i++)
    {

        const double WiByrhoM_ = WiByrhoM[i];
        double dYidt = dYTpdt[i]*WiByrhoM[i];

        for (unsigned int j=0; j<this->nSpecies; j++)
        {
            const double dCjdYj = rhoM*this->invW[j];
            const double ddNidtByVdYj = ddNdtByVdcTp[i*(n_) + j]*dCjdYj;            
            double& ddYidtdYj = J[i*(n_) + j];
            ddYidtdYj = WiByrhoM_*ddNidtByVdYj + rhoMvj[j]*dYidt;
        }
    }
}


void 
OptReaction::FastddYdtdY_Vec
(
    const double* __restrict__ ExpNegGstdByRT,
    const double* __restrict__ ddNdtByVdcTp,
    const double* __restrict__ rhoMvj,
    const double* __restrict__ WiByrhoM,  
    const double* __restrict__ dYTpdt, 
    const double* __restrict__ YTp,  
    double* __restrict__ J
) const noexcept
{ 
    unsigned int remain = this->nSpecies%4;
    for(unsigned int i=0; i<this->nSpecies-remain; i=i+4)
    {
        const double Wi0ByrhoM_ = WiByrhoM[i+0];
        const double Wi1ByrhoM_ = WiByrhoM[i+1];
        const double Wi2ByrhoM_ = WiByrhoM[i+2];
        const double Wi3ByrhoM_ = WiByrhoM[i+3];
        double dYi0dt = dYTpdt[i+0]*WiByrhoM[i+0];
        double dYi1dt = dYTpdt[i+1]*WiByrhoM[i+1];
        double dYi2dt = dYTpdt[i+2]*WiByrhoM[i+2];
        double dYi3dt = dYTpdt[i+3]*WiByrhoM[i+3];
        for (unsigned int j=0; j<this->nSpecies-remain; j=j+4)
        {
            __m256d dCjdYj = _mm256_mul_pd(_mm256_set1_pd(rhoM),_mm256_loadu_pd(&this->invW[j+0]));
            __m256d ddNi0dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+0)*(n_)+j+0]),dCjdYj);
            __m256d ddNi1dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+1)*(n_)+j+0]),dCjdYj);
            __m256d ddNi2dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+2)*(n_)+j+0]),dCjdYj);
            __m256d ddNi3dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+3)*(n_)+j+0]),dCjdYj);
            __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
            __m256d ddYi0dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi0ByrhoM_),ddNi0dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi0dt)));
            _mm256_storeu_pd(&J[(i+0)*(n_) + j+0],ddYi0dtdYj);
            __m256d ddYi1dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi1ByrhoM_),ddNi1dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi1dt)));
            _mm256_storeu_pd(&J[(i+1)*(n_) + j+0],ddYi1dtdYj);
            __m256d ddYi2dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi2ByrhoM_),ddNi2dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi2dt)));
            _mm256_storeu_pd(&J[(i+2)*(n_) + j+0],ddYi2dtdYj);
            __m256d ddYi3dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi3ByrhoM_),ddNi3dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi3dt)));
            _mm256_storeu_pd(&J[(i+3)*(n_) + j+0],ddYi3dtdYj);
        }
        if(remain==1)
        {
            unsigned int j = this->nSpecies-1;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            const double ddNi1dtByVdYj0 = ddNdtByVdcTp[(i+1)*(n_) + j+0]*dCj0dYj0;
            const double ddNi2dtByVdYj0 = ddNdtByVdcTp[(i+2)*(n_) + j+0]*dCj0dYj0;
            const double ddNi3dtByVdYj0 = ddNdtByVdcTp[(i+3)*(n_) + j+0]*dCj0dYj0;
            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            double& ddYi1dtdYj0 = J[(i+1)*(n_) + j+0];
            double& ddYi2dtdYj0 = J[(i+2)*(n_) + j+0];
            double& ddYi3dtdYj0 = J[(i+3)*(n_) + j+0];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj[j+0]*dYi1dt;
            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj[j+0]*dYi2dt;
            ddYi3dtdYj0 = Wi3ByrhoM_*ddNi3dtByVdYj0 + rhoMvj[j+0]*dYi3dt;
        }
        else if(remain==2)
        {
            unsigned int j = this->nSpecies-2;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double dCj1dYj1 = rhoM*this->invW[j+1];
            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            const double ddNi0dtByVdYj1 = ddNdtByVdcTp[(i+0)*(n_) + j+1]*dCj1dYj1;
            const double ddNi1dtByVdYj0 = ddNdtByVdcTp[(i+1)*(n_) + j+0]*dCj0dYj0;
            const double ddNi1dtByVdYj1 = ddNdtByVdcTp[(i+1)*(n_) + j+1]*dCj1dYj1;
            const double ddNi2dtByVdYj0 = ddNdtByVdcTp[(i+2)*(n_) + j+0]*dCj0dYj0;
            const double ddNi2dtByVdYj1 = ddNdtByVdcTp[(i+2)*(n_) + j+1]*dCj1dYj1;
            const double ddNi3dtByVdYj0 = ddNdtByVdcTp[(i+3)*(n_) + j+0]*dCj0dYj0;
            const double ddNi3dtByVdYj1 = ddNdtByVdcTp[(i+3)*(n_) + j+1]*dCj1dYj1;
            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            double& ddYi0dtdYj1 = J[(i+0)*(n_) + j+1];
            double& ddYi1dtdYj0 = J[(i+1)*(n_) + j+0];
            double& ddYi1dtdYj1 = J[(i+1)*(n_) + j+1];
            double& ddYi2dtdYj0 = J[(i+2)*(n_) + j+0];
            double& ddYi2dtdYj1 = J[(i+2)*(n_) + j+1];
            double& ddYi3dtdYj0 = J[(i+3)*(n_) + j+0];
            double& ddYi3dtdYj1 = J[(i+3)*(n_) + j+1];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj[j+1]*dYi0dt;
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj[j+0]*dYi1dt;
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj[j+1]*dYi1dt;
            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj[j+0]*dYi2dt;
            ddYi2dtdYj1 = Wi2ByrhoM_*ddNi2dtByVdYj1 + rhoMvj[j+1]*dYi2dt;
            ddYi3dtdYj0 = Wi3ByrhoM_*ddNi3dtByVdYj0 + rhoMvj[j+0]*dYi3dt;
            ddYi3dtdYj1 = Wi3ByrhoM_*ddNi3dtByVdYj1 + rhoMvj[j+1]*dYi3dt;
        }
        else if(remain==3)
        {
            unsigned int j = this->nSpecies-3;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double dCj1dYj1 = rhoM*this->invW[j+1];
            const double dCj2dYj2 = rhoM*this->invW[j+2];
            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            const double ddNi0dtByVdYj1 = ddNdtByVdcTp[(i+0)*(n_) + j+1]*dCj1dYj1;
            const double ddNi0dtByVdYj2 = ddNdtByVdcTp[(i+0)*(n_) + j+2]*dCj2dYj2;
            const double ddNi1dtByVdYj0 = ddNdtByVdcTp[(i+1)*(n_) + j+0]*dCj0dYj0;
            const double ddNi1dtByVdYj1 = ddNdtByVdcTp[(i+1)*(n_) + j+1]*dCj1dYj1;
            const double ddNi1dtByVdYj2 = ddNdtByVdcTp[(i+1)*(n_) + j+2]*dCj2dYj2;
            const double ddNi2dtByVdYj0 = ddNdtByVdcTp[(i+2)*(n_) + j+0]*dCj0dYj0;
            const double ddNi2dtByVdYj1 = ddNdtByVdcTp[(i+2)*(n_) + j+1]*dCj1dYj1;
            const double ddNi2dtByVdYj2 = ddNdtByVdcTp[(i+2)*(n_) + j+2]*dCj2dYj2;
            const double ddNi3dtByVdYj0 = ddNdtByVdcTp[(i+3)*(n_) + j+0]*dCj0dYj0;
            const double ddNi3dtByVdYj1 = ddNdtByVdcTp[(i+3)*(n_) + j+1]*dCj1dYj1;
            const double ddNi3dtByVdYj2 = ddNdtByVdcTp[(i+3)*(n_) + j+2]*dCj2dYj2;
            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            double& ddYi0dtdYj1 = J[(i+0)*(n_) + j+1];
            double& ddYi0dtdYj2 = J[(i+0)*(n_) + j+2];
            double& ddYi1dtdYj0 = J[(i+1)*(n_) + j+0];
            double& ddYi1dtdYj1 = J[(i+1)*(n_) + j+1];
            double& ddYi1dtdYj2 = J[(i+1)*(n_) + j+2];
            double& ddYi2dtdYj0 = J[(i+2)*(n_) + j+0];
            double& ddYi2dtdYj1 = J[(i+2)*(n_) + j+1];
            double& ddYi2dtdYj2 = J[(i+2)*(n_) + j+2];
            double& ddYi3dtdYj0 = J[(i+3)*(n_) + j+0];
            double& ddYi3dtdYj1 = J[(i+3)*(n_) + j+1];
            double& ddYi3dtdYj2 = J[(i+3)*(n_) + j+2];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj[j+1]*dYi0dt;
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj[j+2]*dYi0dt;
            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj[j+0]*dYi1dt;
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj[j+1]*dYi1dt;
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj[j+2]*dYi1dt;
            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj[j+0]*dYi2dt;
            ddYi2dtdYj1 = Wi2ByrhoM_*ddNi2dtByVdYj1 + rhoMvj[j+1]*dYi2dt;
            ddYi2dtdYj2 = Wi2ByrhoM_*ddNi2dtByVdYj2 + rhoMvj[j+2]*dYi2dt;
            ddYi3dtdYj0 = Wi3ByrhoM_*ddNi3dtByVdYj0 + rhoMvj[j+0]*dYi3dt;
            ddYi3dtdYj1 = Wi3ByrhoM_*ddNi3dtByVdYj1 + rhoMvj[j+1]*dYi3dt;
            ddYi3dtdYj2 = Wi3ByrhoM_*ddNi3dtByVdYj2 + rhoMvj[j+2]*dYi3dt;
        }
    }
    if(remain==1)
    {
        unsigned int i = this->nSpecies-1;
        const double Wi0ByrhoM_ = WiByrhoM[i+0];
        double dYi0dt = dYTpdt[i+0]*WiByrhoM[i+0];

        for (unsigned int j=0; j<this->nSpecies-remain; j=j+4)
        {
            __m256d dCjdYj = _mm256_mul_pd(_mm256_set1_pd(rhoM),_mm256_loadu_pd(&this->invW[j+0]));
            __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);

            __m256d ddNi0dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+0)*(n_) + j+0]),dCjdYj);
            __m256d ddYi0dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi0ByrhoM_),ddNi0dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi0dt)));
            _mm256_storeu_pd(&J[(i+0)*(n_) + j+0],ddYi0dtdYj);
        }
        {
            unsigned int j = this->nSpecies-1;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
        }
    }
    else if(remain==2)
    {
        unsigned int i = this->nSpecies-2;
        const double Wi0ByrhoM_ = WiByrhoM[i+0];
        const double Wi1ByrhoM_ = WiByrhoM[i+1];

        double dYi0dt = dYTpdt[i+0]*WiByrhoM[i+0];
        double dYi1dt = dYTpdt[i+1]*WiByrhoM[i+1];
        
        for (unsigned int j=0; j<this->nSpecies-remain; j=j+4)
        {
            __m256d dCjdYj = _mm256_mul_pd(_mm256_set1_pd(rhoM),_mm256_loadu_pd(&this->invW[j+0]));
            __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);

            __m256d ddNi0dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+0)*(n_) + j+0]),dCjdYj);
            __m256d ddNi1dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+1)*(n_) + j+0]),dCjdYj);
            __m256d ddYi0dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi0ByrhoM_),ddNi0dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi0dt)));
            _mm256_storeu_pd(&J[(i+0)*(n_) + j+0],ddYi0dtdYj);
            __m256d ddYi1dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi1ByrhoM_),ddNi1dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi1dt)));
            _mm256_storeu_pd(&J[(i+1)*(n_) + j+0],ddYi1dtdYj);

        }
        {
            unsigned int j = this->nSpecies-2;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double dCj1dYj1 = rhoM*this->invW[j+1];

            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            const double ddNi0dtByVdYj1 = ddNdtByVdcTp[(i+0)*(n_) + j+1]*dCj1dYj1;

            const double ddNi1dtByVdYj0 = ddNdtByVdcTp[(i+1)*(n_) + j+0]*dCj0dYj0;
            const double ddNi1dtByVdYj1 = ddNdtByVdcTp[(i+1)*(n_) + j+1]*dCj1dYj1;


            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            double& ddYi0dtdYj1 = J[(i+0)*(n_) + j+1];

            double& ddYi1dtdYj0 = J[(i+1)*(n_) + j+0];
            double& ddYi1dtdYj1 = J[(i+1)*(n_) + j+1];

            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj[j+1]*dYi0dt;

            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj[j+0]*dYi1dt;
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj[j+1]*dYi1dt;
        }
    }
    else if(remain==3)
    {
        unsigned int i = this->nSpecies - 3;
        const double Wi0ByrhoM_ = WiByrhoM[i+0];
        const double Wi1ByrhoM_ = WiByrhoM[i+1];
        const double Wi2ByrhoM_ = WiByrhoM[i+2];

        double dYi0dt = dYTpdt[i+0]*WiByrhoM[i+0];
        double dYi1dt = dYTpdt[i+1]*WiByrhoM[i+1];
        double dYi2dt = dYTpdt[i+2]*WiByrhoM[i+2];
        
        for (unsigned int j=0; j<this->nSpecies-remain; j=j+4)
        {
            __m256d dCjdYj = _mm256_mul_pd(_mm256_set1_pd(rhoM),_mm256_loadu_pd(&this->invW[j+0]));
            __m256d rhoMvj_ = _mm256_loadu_pd(&rhoMvj[j+0]);
            __m256d ddNi0dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+0)*(n_) + j+0]),dCjdYj);
            __m256d ddNi1dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+1)*(n_) + j+0]),dCjdYj);
            __m256d ddNi2dtByVdYj = _mm256_mul_pd(_mm256_loadu_pd(&ddNdtByVdcTp[(i+2)*(n_) + j+0]),dCjdYj);

            __m256d ddYi0dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi0ByrhoM_),ddNi0dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi0dt)));
            _mm256_storeu_pd(&J[(i+0)*(n_) + j+0],ddYi0dtdYj);

            __m256d ddYi1dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi1ByrhoM_),ddNi1dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi1dt)));
            _mm256_storeu_pd(&J[(i+1)*(n_) + j+0],ddYi1dtdYj);

            __m256d ddYi2dtdYj = _mm256_fmadd_pd(_mm256_set1_pd(Wi2ByrhoM_),ddNi2dtByVdYj,_mm256_mul_pd(rhoMvj_,_mm256_set1_pd(dYi2dt)));
            _mm256_storeu_pd(&J[(i+2)*(n_) + j+0],ddYi2dtdYj);
        }
        {
            unsigned int j = this->nSpecies-3;
            const double dCj0dYj0 = rhoM*this->invW[j+0];
            const double dCj1dYj1 = rhoM*this->invW[j+1];
            const double dCj2dYj2 = rhoM*this->invW[j+2];

            const double ddNi0dtByVdYj0 = ddNdtByVdcTp[(i+0)*(n_) + j+0]*dCj0dYj0;
            const double ddNi0dtByVdYj1 = ddNdtByVdcTp[(i+0)*(n_) + j+1]*dCj1dYj1;
            const double ddNi0dtByVdYj2 = ddNdtByVdcTp[(i+0)*(n_) + j+2]*dCj2dYj2;

            const double ddNi1dtByVdYj0 = ddNdtByVdcTp[(i+1)*(n_) + j+0]*dCj0dYj0;
            const double ddNi1dtByVdYj1 = ddNdtByVdcTp[(i+1)*(n_) + j+1]*dCj1dYj1;
            const double ddNi1dtByVdYj2 = ddNdtByVdcTp[(i+1)*(n_) + j+2]*dCj2dYj2;

            const double ddNi2dtByVdYj0 = ddNdtByVdcTp[(i+2)*(n_) + j+0]*dCj0dYj0;
            const double ddNi2dtByVdYj1 = ddNdtByVdcTp[(i+2)*(n_) + j+1]*dCj1dYj1;
            const double ddNi2dtByVdYj2 = ddNdtByVdcTp[(i+2)*(n_) + j+2]*dCj2dYj2;

            double& ddYi0dtdYj0 = J[(i+0)*(n_) + j+0];
            double& ddYi0dtdYj1 = J[(i+0)*(n_) + j+1];
            double& ddYi0dtdYj2 = J[(i+0)*(n_) + j+2];

            double& ddYi1dtdYj0 = J[(i+1)*(n_) + j+0];
            double& ddYi1dtdYj1 = J[(i+1)*(n_) + j+1];
            double& ddYi1dtdYj2 = J[(i+1)*(n_) + j+2];

            double& ddYi2dtdYj0 = J[(i+2)*(n_) + j+0];
            double& ddYi2dtdYj1 = J[(i+2)*(n_) + j+1];
            double& ddYi2dtdYj2 = J[(i+2)*(n_) + j+2];

            ddYi0dtdYj0 = Wi0ByrhoM_*ddNi0dtByVdYj0 + rhoMvj[j+0]*dYi0dt;
            ddYi0dtdYj1 = Wi0ByrhoM_*ddNi0dtByVdYj1 + rhoMvj[j+1]*dYi0dt;
            ddYi0dtdYj2 = Wi0ByrhoM_*ddNi0dtByVdYj2 + rhoMvj[j+2]*dYi0dt;

            ddYi1dtdYj0 = Wi1ByrhoM_*ddNi1dtByVdYj0 + rhoMvj[j+0]*dYi1dt;
            ddYi1dtdYj1 = Wi1ByrhoM_*ddNi1dtByVdYj1 + rhoMvj[j+1]*dYi1dt;
            ddYi1dtdYj2 = Wi1ByrhoM_*ddNi1dtByVdYj2 + rhoMvj[j+2]*dYi1dt;

            ddYi2dtdYj0 = Wi2ByrhoM_*ddNi2dtByVdYj0 + rhoMvj[j+0]*dYi2dt;
            ddYi2dtdYj1 = Wi2ByrhoM_*ddNi2dtByVdYj1 + rhoMvj[j+1]*dYi2dt;
            ddYi2dtdYj2 = Wi2ByrhoM_*ddNi2dtByVdYj2 + rhoMvj[j+2]*dYi2dt;
        }
    }
}
