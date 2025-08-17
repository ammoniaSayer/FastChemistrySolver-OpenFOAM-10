
#include "OptReaction.H"

void 
OptReaction::Tc
(
    int celli,
    double p,
    double Temperature,
    double* C,
    double& sumW,
    double& sumWRateByCTot
) const noexcept
{
    Temperature = Temperature<TlowMin?TlowMin:Temperature;
    Temperature = Temperature>ThighMax?ThighMax:Temperature;
    this->logT = std::log(Temperature);
    this->invT = 1/Temperature;
    this->sqrT = Temperature*Temperature;


    this->setPtrCoeffs(Temperature);
    this->ExpNegGstdByRT(Temperature,&this->tmp_Exp[0]);
    this->update_Pow_pByRT_SumVki(Temperature);
    this->update_Pow_pByRT_SumVki2(Temperature);

    {
        for(size_t i = 0; i <this->Troe.size();i++)
        {
            size_t j0 = i + this->nSpecies;
            size_t j1 = i + this->nSpecies + this->Troe.size();
            size_t j2 = i + this->nSpecies + this->Troe.size()*2;         
            this->tmp_Exp[j0] = -Temperature*this->invTsss_[i];
            this->tmp_Exp[j1] = -this->Tss_[i]*invT;    
            this->tmp_Exp[j2] = -Temperature*this->invTs_[i];
        }
    }

    {
        for(size_t i = 0; i <this->SRI.size();i++)
        {
            size_t j0 = i + this->nSpecies + this->Troe.size()*3;
            size_t j1 = i + this->nSpecies + this->Troe.size()*3 + this->SRI.size();
            this->tmp_Exp[j0] = -this->b_[i]*invT;
            this->tmp_Exp[j1] = -Temperature*this->invc_[i];
        }       
    }


    {
         unsigned int remain = this->tmp_Exp.size()%4;
        for(unsigned int i = 0; i < this->tmp_Exp.size()-remain;i=i+4)
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



    {
        __m256d LogT = _mm256_set1_pd(logT);
        __m256d InvT = _mm256_set1_pd(-invT);
        unsigned int remain = (this->Ikf[11]-n_Temperature_Independent_Reaction)%4;
        unsigned int times = (this->Ikf[11]-n_Temperature_Independent_Reaction)/4;
        for(unsigned int z = 0; z <times;z=z+1)
        {
            unsigned int i = z*4 + this->n_Temperature_Independent_Reaction;
            __m256d A_ = _mm256_loadu_pd(&this->A[i]);
            __m256d beta_ = _mm256_loadu_pd(&this->beta[i]);
            __m256d Ta_ = _mm256_loadu_pd(&this->Ta[i]);
            __m256d Kf = _mm256_mul_pd(Ta_,InvT);
            Kf = _mm256_fmadd_pd(beta_,LogT,Kf);
            Kf = vec256_expd(Kf);
            Kf = _mm256_mul_pd(A_,Kf);
            _mm256_storeu_pd(&this->Kf_[i],Kf);
        }
        if(remain==1)
        {
            unsigned int i = this->Ikf[11]-1;
            this->Kf_[i] = this->A[i]*std::exp(this->beta[i+0]*logT-this->Ta[i+0]*invT);   
        }
        else if(remain==2)
        {
            unsigned int i0 = this->Ikf[11]-2;
            unsigned int i1 = this->Ikf[11]-1;
            __m256d A_ = _mm256_setr_pd(this->A[i0],this->A[i1],0,0);
            __m256d beta_ = _mm256_setr_pd(this->beta[i0],this->beta[i1],0,0);
            __m256d Ta_ = _mm256_setr_pd(this->Ta[i0],this->Ta[i1],0,0);
            __m256d Kf = _mm256_mul_pd(Ta_,InvT);
            Kf = _mm256_fmadd_pd(beta_,LogT,Kf);
            Kf = vec256_expd(Kf);
            Kf = _mm256_mul_pd(A_,Kf);
            this->Kf_[i0] = get_elem0(Kf);
            this->Kf_[i1] = get_elem1(Kf);
        }
        else if(remain==3)
        {
            unsigned int i0 = this->Ikf[11]-3;
            unsigned int i1 = this->Ikf[11]-2;
            unsigned int i2 = this->Ikf[11]-1;
            __m256d A_ = _mm256_setr_pd(this->A[i0],this->A[i1],this->A[i2],0);
            __m256d beta_ = _mm256_setr_pd(this->beta[i0],this->beta[i1],this->beta[i2],0);
            __m256d Ta_ = _mm256_setr_pd(this->Ta[i0],this->Ta[i1],this->Ta[i2],0);
            __m256d Kf = _mm256_mul_pd(Ta_,InvT);
            Kf = _mm256_fmadd_pd(beta_,LogT,Kf);
            Kf = vec256_expd(Kf);
            Kf = _mm256_mul_pd(A_,Kf);
            this->Kf_[i0] = get_elem0(Kf);
            this->Kf_[i1] = get_elem1(Kf); 
            this->Kf_[i2] = get_elem2(Kf); 
        }
    }



    {
        unsigned int Tremain = (Itbr[5])%4;
        unsigned int remain = this->nSpecies%4;
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
            for(unsigned int j  = 0;j<this->nSpecies-remain;j=j+4)
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

            if(remain==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M3 = M3 + TBF1DRowi3[j+0]*C[j+0];
            }
            else if(remain==2)
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
            else if(remain==3)
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
            for(unsigned int j  = 0;j<this->nSpecies-remain;j=j+4)
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

            if(remain==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
            }
            else if(remain==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M2 = M2 + TBF1DRowi2[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
                M2 = M2 + TBF1DRowi2[j+1]*C[j+1];
            }
            else if(remain==3)
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
            for(unsigned int j  = 0;j<this->nSpecies-remain;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d Factor1 = _mm256_loadu_pd(&TBF1DRowi1[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
                arrM_1 = _mm256_fmadd_pd(Factor1,C_,arrM_1);
            }

            M0 = M0 + hsum4(arrM_0);
            M1 = M1 + hsum4(arrM_1);

            if(remain==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
            }
            else if(remain==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M1 = M1 + TBF1DRowi1[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M1 = M1 + TBF1DRowi1[j+1]*C[j+1];
            }
            else if(remain==3)
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
            for(unsigned int j  = 0;j<this->nSpecies-remain;j=j+4)
            {
                __m256d Factor0 = _mm256_loadu_pd(&TBF1DRowi0[j+0]);
                __m256d C_ = _mm256_loadu_pd(&C[j+0]);
                arrM_0 = _mm256_fmadd_pd(Factor0,C_,arrM_0);
            }

            M0 = M0 + hsum4(arrM_0);

            if(remain==1)
            {
                unsigned int j = this->nSpecies-1;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
            }
            else if(remain==2)
            {
                unsigned int j = this->nSpecies-2;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
            }
            else if(remain==3)
            {
                unsigned int j = this->nSpecies-3;
                M0 = M0 + TBF1DRowi0[j+0]*C[j+0];
                M0 = M0 + TBF1DRowi0[j+1]*C[j+1];
                M0 = M0 + TBF1DRowi0[j+2]*C[j+2];
            }
            this->tmp_M[i+0] = M0;
        }
    }






    {
        for(unsigned int i = 0; i < this->n_ThirdBodyReaction; i++)
        {
            const unsigned int j = i + Ikf[3];
            this->Kf_[j] = this->Kf_[j]*this->tmp_M[i+this->Itbr[1]];
        }
    }


    {
        for(unsigned int i = 0; i < this->n_NonEquilibriumThirdBodyReaction; i++)
        {
            double Mfwd = this->tmp_M[i];
            double Mrev = this->tmp_M[this->Itbr[4]+i];
            this->Kf_[Ikf[2]+i] = this->Kf_[Ikf[2]+i]*Mfwd;
            this->Kf_[Ikf[10]+i] = this->Kf_[Ikf[10]+i]*Mrev;
        } 
    }





    {

        unsigned int remain = this->Lindemann.size()%4;

        for (unsigned int i = 0;i<Lindemann.size()-remain;i=i+4)
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
            __m256d Kinf = _mm256_loadu_pd(&this->Kf_[j0+offset_kinf]);
            __m256d M = _mm256_loadu_pd(&this->tmp_M[m0]);
            __m256d K0 = _mm256_loadu_pd(&this->Kf_[j0]);         
            __m256d Pr = _mm256_div_pd(_mm256_mul_pd(K0,M),Kinf);
            __m256d N = _mm256_div_pd(K0,_mm256_add_pd(Pr,_mm256_set1_pd(1.0)));
            __m256d k = _mm256_setr_pd(k0,k1,k2,k3);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d Kf = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);
            _mm256_storeu_pd(&this->Kf_[j0],Kf);
        }


        if(remain==1)
        {
            size_t i = Lindemann.size()-1;
            const unsigned int j = this->Lindemann[i];
            const unsigned int m = j - Ikf[4] + Itbr[2];
            const unsigned int k = j - Ikf[4];
            const double Kinf = this->Kf_[j+offset_kinf];
            double M = this->tmp_M[m];     
            const double K0 = this->Kf_[j];
            const double Pr = K0*M/Kinf;   
            const double N          = 1/(1+Pr)*K0;
            this->Kf_[j] = k<this->n_Fall_Off_Reaction ? M*N : N;            
        }
        else if(remain==2)
        {
            size_t i = Lindemann.size()-2;
            const unsigned int j0 = this->Lindemann[i+0]+0;
            const unsigned int j1 = this->Lindemann[i+0]+1;
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            __m128d Kinf = _mm_loadu_pd(&this->Kf_[j0+offset_kinf]);
            __m128d M = _mm_loadu_pd(&this->tmp_M[m0]);
            __m128d K0 = _mm_loadu_pd(&this->Kf_[j0]);
            __m128d Pr = _mm_div_pd(_mm_mul_pd(K0,M),Kinf);
            __m128d N = _mm_div_pd(K0,_mm_add_pd(Pr,_mm_set1_pd(1.0)));
            __m128d k = _mm_setr_pd(k0,k1);
            __m128d cmp = _mm_cmp_pd(k,_mm_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m128d Kf = _mm_blendv_pd(N,_mm_mul_pd(M,N),cmp);
            _mm_storeu_pd(&this->Kf_[j0],Kf);
        }
        else if(remain==3)
        {
            size_t i = Lindemann.size()-3;
            const unsigned int j0 = this->Lindemann[i+0];
            const unsigned int j1 = this->Lindemann[i+1];
            const unsigned int j2 = this->Lindemann[i+2];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            const unsigned int m1 = j1 - Ikf[4] + Itbr[2];
            const unsigned int m2 = j2 - Ikf[4] + Itbr[2];
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            const unsigned int k2 = j2 - Ikf[4];

            const double Kinf0 = this->Kf_[j0+offset_kinf];
            const double Kinf1 = this->Kf_[j1+offset_kinf];
            const double Kinf2 = this->Kf_[j2+offset_kinf];
            double M0 = this->tmp_M[m0];
            double M1 = this->tmp_M[m1];
            double M2 = this->tmp_M[m2];
            const double K00 = this->Kf_[j0];
            const double K01 = this->Kf_[j1];
            const double K02 = this->Kf_[j2];

            __m256d Kinf = _mm256_setr_pd(Kinf0,Kinf1,Kinf2,1);
            __m256d M = _mm256_setr_pd(M0,M1,M2,0);
            __m256d K0 = _mm256_setr_pd(K00,K01,K02,0);         
            __m256d Pr = _mm256_div_pd(_mm256_mul_pd(K0,M),Kinf);
            __m256d N = _mm256_div_pd(K0,_mm256_add_pd(Pr,_mm256_set1_pd(1.0)));
            __m256d k = _mm256_setr_pd(k0,k1,k2,0);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d Kf = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);
            double tmp[4] = {0,0,0,0};
            _mm256_storeu_pd(&tmp[0],Kf);  
            this->Kf_[j0] = tmp[0];
            this->Kf_[j1] = tmp[1];
            this->Kf_[j2] = tmp[2];
 

        }

    }


    {
    
        unsigned int remain = this->Troe.size()%4;
        for(unsigned int i = 0; i < this->Troe.size()-remain;i=i+4)
        {
            const unsigned int j0 = this->Troe[i+0];
            const unsigned int j1 = j0 + 1;
            const unsigned int j2 = j0 + 2;
            const unsigned int j3 = j0 + 3;
            const unsigned int k0 = j0 - Ikf[4];
            const unsigned int k1 = j1 - Ikf[4];
            const unsigned int k2 = j2 - Ikf[4];
            const unsigned int k3 = j3 - Ikf[4];
            const unsigned int m0 = j0 - Ikf[4] + Itbr[2];
            __m256d Kinf = _mm256_loadu_pd(&this->Kf_[j0+offset_kinf]);
            __m256d M = _mm256_loadu_pd(&this->tmp_M[m0]);
            __m256d K0 = _mm256_loadu_pd(&this->Kf_[j0]);    
            __m256d Pr_ = _mm256_div_pd(_mm256_mul_pd(K0,M),Kinf);
            __m256d small = _mm256_set1_pd(2.2e-16);
            Pr_ = _mm256_max_pd(small,Pr_);
            const double invLog10 = 0.43429448190325182765112891891661;
            __m256d logPr_ = _mm256_mul_pd(vec256_logd(Pr_),_mm256_set1_pd(invLog10));
            __m256d alpha = _mm256_loadu_pd(&this->alpha_[i]);
            __m256d one = _mm256_set1_pd(1.0);
            __m256d expTTsss = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies]);
            __m256d expTTss = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies+this->Troe.size()]);
            __m256d expTTs = _mm256_loadu_pd(&this->tmp_Exp[i+this->nSpecies+this->Troe.size()*2]);
            __m256d Fcent  = _mm256_mul_pd(_mm256_sub_pd(one,alpha), expTTsss);
            Fcent = _mm256_fmadd_pd(alpha, expTTs,Fcent);
            Fcent = _mm256_add_pd(expTTss,Fcent);
            __m256d logFcent = _mm256_mul_pd(vec256_logd(_mm256_max_pd(Fcent,small)),_mm256_set1_pd(invLog10));
            __m256d c = _mm256_fmadd_pd(logFcent,_mm256_set1_pd(0.67),_mm256_set1_pd(0.4));
            __m256d n = _mm256_fmadd_pd(logFcent,_mm256_set1_pd(-1.27),_mm256_set1_pd(0.75));
            __m256d x1 = _mm256_fmadd_pd(_mm256_sub_pd(c,logPr_),_mm256_set1_pd(0.14),n);
            __m256d x2 = _mm256_div_pd(_mm256_sub_pd(logPr_,c),x1);
            __m256d x3 = _mm256_fmadd_pd(x2,x2,one);
            __m256d x4 = _mm256_div_pd(logFcent,x3);
            __m256d F_ = vec256_powd(_mm256_set1_pd(10),x4);
            __m256d N = _mm256_div_pd(_mm256_mul_pd(K0,F_),_mm256_add_pd(_mm256_set1_pd(1.0),Pr_));
            __m256d k = _mm256_setr_pd(k0,k1,k2,k3);
            __m256d cmp = _mm256_cmp_pd(k,_mm256_set1_pd(this->n_Fall_Off_Reaction),_CMP_LT_OQ);
            __m256d Kf = _mm256_blendv_pd(N,_mm256_mul_pd(M,N),cmp);
            _mm256_storeu_pd(&this->Kf_[j0],Kf);         
        }        
        if(remain==1)       {this->Troe_F_1();}
        else if(remain==2)  {this->Troe_F_2();}
        else if(remain==3)  {this->Troe_F_3();}
    
    }
    {
        for (unsigned int i = 0;i<SRI.size();i++)
        {
            const unsigned int j = this->SRI[i];

            const unsigned int m = j - Ikf[4] + Itbr[2];
            const unsigned int k = j - Ikf[4];
            const double Kinf = this->Kf_[j+offset_kinf];
            double M = this->tmp_M[m]; 
            const double K0 = this->Kf_[j];
            const double Pr = K0*M/Kinf;   
            const double F  = this->SRI_F(Temperature,Pr,i);
            const double N  = 1/(1+Pr)*F*K0;
            this->Kf_[j] = k<this->n_Fall_Off_Reaction ? M*N : N;   
        }
    }


    sumW = 0;
    sumWRateByCTot = 0;
    for (unsigned int i = 0; i < Ikf[7]; i++) 
    {
        double Kf = Kf_[i];
        int sumVki = 0;
        
        double Kp = 1.0;
        double CR = 1.0;
        double CF = 1.0;
        double Kc_ = 1.0;
        double Kr = 0;
        if(this->isIrreversible[i]==0)
        {
            for(unsigned int j = 0; j < this->lhsIndex[i].size();j++)
            {
                const unsigned int si = this->lhsIndex[i][j];
                const unsigned int sl = this->lhsstoichCoeff[i][j];
                Kp = Kp / this->tmp_Exp[si];
                sumVki = sumVki - sl;
                CF = CF * C[si];
            }
            
            for(unsigned int j = 0; j < this->rhsIndex[i].size();j++)
            {
                const unsigned int si = this->rhsIndex[i][j];
                const unsigned int sr = this->rhsstoichCoeff[i][j];
                Kp = Kp * this->tmp_Exp[si];
                sumVki = sumVki + sr;
                CR = CR * C[si];
            }
            double x = this->Pow_pByRT_SumVki_[sumVki];
            Kc_ = Kp*x;
            Kc_ = std::max(Kc_,1.49011611938476E-08);
            Kr = Kf/Kc_;
        }
        else
        {
            for(unsigned int j = 0; j < this->lhsIndex[i].size();j++)
            {
                const unsigned int si = this->lhsIndex[i][j];
                CF = CF * C[si];
            }
            
            for(unsigned int j = 0; j < this->rhsIndex[i].size();j++)
            {
                const unsigned int si = this->rhsIndex[i][j];
                CR = CR * C[si];
            }
        }


        const double omegaf = Kf*CF;
        const double omegar = Kr*CR;

        double wf = 0;
        for(unsigned int j = 0; j < rhsIndex[i].size();j++)
        {
            wf+=omegaf;
        }
        sumW+=wf;
        sumWRateByCTot += (wf*wf);

        double wr = 0;       
        for(unsigned int j = 0; j < lhsIndex[i].size();j++)
        {
            wr+=omegar;
        }
        sumW+=wr;
        sumWRateByCTot += (wr*wr);        
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
                }
                else if(p>=this->Prange[i][length-1])
                {
                    this->Kf_Plog[i] =   this->logAPlog[i][length-1] + 
                                                this->betaPlog[i][length-1]*this->logT - 
                                                this->TaPlog[i][length-1]*invT;
                }
                else
                {
                    unsigned int index = 0;
                    for(unsigned int j = 0; j < length-1;j++)
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
                                            (this->logAPlog[i][index+1] - this->logAPlog[i][index]) +
                                            (this->betaPlog[i][index+1] - this->betaPlog[i][index])*this->logT -
                                            (this->TaPlog[i][index+1] - this->TaPlog[i][index])*invT
                                        )*
                                        (
                                            (logP - this->logPi[i][index])/
                                            (this->logPi[i][index+1] - this->logPi[i][index])
                                        ); 
    
                }
            }
        }
        else 
        {
            const size_t length = this->Prange[0].size();
            unsigned int index = 0;
            if(p<=this->Prange[0][0])
            {
                for(unsigned int i = 0; i< this->n_PlogReaction; i++)
                {
                    this->Kf_Plog[i] =  this->logAPlog[i][0] + 
                                        this->betaPlog[i][0]*this->logT - 
                                        this->TaPlog[i][0]*invT;
                }     
            }
            else if(p>=this->Prange[0][length-1])
            {
                for(unsigned int i = 0; i< this->n_PlogReaction; i++)
                {
                    this->Kf_Plog[i] =  this->logAPlog[i][length -1] + 
                                        this->betaPlog[i][length -1]*this->logT - 
                                        this->TaPlog[i][length -1]*invT;
                }             
            }
            else
            {
                for(unsigned int j = 0; j < length-1;j++)
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
                }

            }
        }

        unsigned int remain = this->n_PlogReaction%4;
        for(unsigned int i = 0; i < this->n_PlogReaction-remain; i=i+4)
        {
            __m256d A_ = _mm256_loadu_pd(&this->Kf_Plog[i]);
            A_ = vec256_expd(A_);
            _mm256_storeu_pd(&this->Kf_Plog[i],A_);
        }
        if(remain==1)
        {
            this->Kf_Plog[this->n_PlogReaction-1] = std::exp(this->Kf_Plog[this->n_PlogReaction-1]);
        }
        else if(remain==2)
        {
            __m256d A_ = _mm256_setr_pd(Kf_Plog[this->n_PlogReaction-2],Kf_Plog[this->n_PlogReaction-1],0,0);
            A_ = vec256_expd(A_);
            this->Kf_Plog[this->n_PlogReaction-2] = get_elem0(A_);
            this->Kf_Plog[this->n_PlogReaction-1] = get_elem1(A_);
        }
        else if(remain==3)
        {
            __m256d A_ = _mm256_setr_pd(Kf_Plog[this->n_PlogReaction-3],Kf_Plog[this->n_PlogReaction-2],Kf_Plog[this->n_PlogReaction-1],0);
            A_ = vec256_expd(A_);
            this->Kf_Plog[this->n_PlogReaction-3] = get_elem0(A_);
            this->Kf_Plog[this->n_PlogReaction-2] = get_elem1(A_);
            this->Kf_Plog[this->n_PlogReaction-1] = get_elem2(A_);
        }

        for(unsigned int i = 0; i < this->n_PlogReaction; i ++)
        {
            unsigned int n = Ikf[7] + i ;
            double Kf = Kf_Plog[i];
            int sumVki = 0;
            
            double Kp = 1.0;
            double CR = 1.0;
            double CF = 1.0;
            double Kc_ = 1.0;
            double Kr = 0;
            if(this->isIrreversible[n]==0)
            {
                for(unsigned int j = 0; j < this->lhsIndex[n].size();j++)
                {
                    const unsigned int si = this->lhsIndex[n][j];
                    const unsigned int sl = this->lhsstoichCoeff[n][j];
                    Kp = Kp / this->tmp_Exp[si];
                    sumVki = sumVki - sl;
                    CF = CF * C[si];
                }
                
                for(unsigned int j = 0; j < this->rhsIndex[n].size();j++)
                {
                    const unsigned int si = this->rhsIndex[n][j];
                    const unsigned int sr = this->rhsstoichCoeff[n][j];
                    Kp = Kp * this->tmp_Exp[si];
                    sumVki = sumVki + sr;
                    CR = CR * C[si];
                }

                Kc_ = Kp*this->Pow_pByRT_SumVki_[sumVki];
                Kc_ = std::max(Kc_,1.49011611938476E-08);
                Kr = Kf/Kc_;
            }
            else
            {
                for(unsigned int j = 0; j < this->lhsIndex[n].size();j++)
                {
                    const unsigned int si = this->lhsIndex[n][j];
                    CF = CF * C[si];
                }
                
                for(unsigned int j = 0; j < this->rhsIndex[n].size();j++)
                {
                    const unsigned int si = this->rhsIndex[n][j];
                    CR = CR * C[si];
                }
            }


            const double omegaf = Kf*CF;
            const double omegar = Kr*CR;
            double wf = 0;
            for(unsigned int j = 0; j < rhsIndex[n].size();j++)
            {
                wf+=omegaf;
            }
            sumW+=wf;
            sumWRateByCTot += (wf)*(wf);

            double wr = 0;       
            for(unsigned int j = 0; j < lhsIndex[n].size();j++)
            {
                wr+=omegar;
            }
            sumW+=wr;
            sumWRateByCTot += (wr)*(wr);                 
        }
    }
}
