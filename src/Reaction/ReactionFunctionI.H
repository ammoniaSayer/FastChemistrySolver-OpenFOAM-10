void inline OptReaction::RF11
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    double Kr = 0;
    if(this->isIrreversible[i]==0)
    {
        const double Kp = ExpNegGbyRT[sr0]/ExpNegGbyRT[sl0];
        double Kc = Kp;
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;       
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);

    if(sameSpecies[i]==0)
    {
        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        this->update11(q,Sl0,Sr0);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sr0] = dNdtByV[sr0] + q;
    }
}
void inline OptReaction::update11
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sr0    
)const noexcept
{
    *sl0 = *sl0 - q;
    *sr0 = *sr0 + q;
}
void inline OptReaction::RF12
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]);
        double Kc = Kp*this->Pow_pByRT_SumVki[3];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;       
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1];

    const double q = (Kf*CF) - (Kr*CR);

 

    if(sameSpecies[i]==0)
    {
        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        this->update12(q,Sl0,Sr0,Sr1);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q; 
    }
      
}             
void inline OptReaction::update12
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1
)const noexcept
{
    *sl0 = *sl0 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
}

void inline OptReaction::RF13
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;
    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]);
        double Kc = Kp*this->Pow_pByRT_SumVki[4];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;         
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0];
    const double CR = C[sr0]*C[sr1]*C[sr2];

    const double q = (Kf*CF) - (Kr*CR);
    if(sameSpecies[i]==0)
    {
        /*double r0 = dNdtByV[sl0];
        double r2 = dNdtByV[sr0];
        double r3 = dNdtByV[sr1];
        double r4 = dNdtByV[sr2];
        r0 -= q;
        r2 += q;
        r3 += q;
        r4 += q;
        dNdtByV[sl0] = r0; 
        dNdtByV[sr0] = r2;
        dNdtByV[sr1] = r3;
        dNdtByV[sr2] = r4;*/


        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        double* __restrict__ Sr2 = &dNdtByV[sr2];

        update13(q,Sl0,Sr0,Sr1,Sr2);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q;
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q;
        dNdtByV[sr2] = dNdtByV[sr2] + q; 
    }
}             
void inline OptReaction::update13
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1,
    double*  __restrict__ sr2
)const noexcept
{
    *sl0 = *sl0 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
    *sr2 = *sr2 + q;
}
void inline OptReaction::RF21
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];


    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = ExpNegGbyRT[sr0]/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp*this->Pow_pByRT_SumVki[1];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;        
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }
    
    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);


    if(sameSpecies[i]==0)
    {
        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sr0 = &dNdtByV[sr0]; 
        update21(q,Sl0,Sl1,Sr0);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sl1] = dNdtByV[sl1] - q;
        dNdtByV[sr0] = dNdtByV[sr0] + q;
    }
}             
void inline OptReaction::update21
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sr0
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sr0 = *sr0 + q;
}
void inline OptReaction::RF22
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];

    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp;
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;         
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1];


    const double q = (Kf*CF) - (Kr*CR);
    if(sameSpecies[i]==0)
    {
        /*double r0 = dNdtByV[sl0];
        double r1 = dNdtByV[sl1];
        double r2 = dNdtByV[sr0];
        double r3 = dNdtByV[sr1];
        r0 -= q;
        r1 -= q;
        r2 += q;
        r3 += q;
        dNdtByV[sl0] = r0; 
        dNdtByV[sl1] = r1;
        dNdtByV[sr0] = r2;
        dNdtByV[sr1] = r3;*/

        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        update22(q,Sl0,Sl1,Sr0,Sr1);

    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sl1] = dNdtByV[sl1] - q;
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q; 
    }
}

void inline OptReaction::update22
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
}

void inline OptReaction::RF23
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];

    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]);
        double Kc = Kp*this->Pow_pByRT_SumVki[3];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;    
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0]*C[sl1];
    const double CR = C[sr0]*C[sr1]*C[sr2];
    const double q = (Kf*CF) - (Kr*CR);

    

    if(sameSpecies[i]==0)
    {
        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        double* __restrict__ Sr2 = &dNdtByV[sr2];

        update23(q,Sl0,Sl1,Sr0,Sr1,Sr2);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q;
        dNdtByV[sl1] = dNdtByV[sl1] - q;
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q;
        dNdtByV[sr2] = dNdtByV[sr2] + q;
    }
       
}

void inline OptReaction::update23
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1,
    double*  __restrict__ sr2
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
    *sr2 = *sr2 + q;
}

void inline OptReaction::RF31
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{

    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];

    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp*this->Pow_pByRT_SumVki[0];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;         
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0];

    const double q = (Kf*CF) - (Kr*CR);

    
    if(sameSpecies[i]==0)
    {

        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sl2 = &dNdtByV[sl2];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        update31(q,Sl0,Sl1,Sl2,Sr0);        
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sl1] = dNdtByV[sl1] - q; 
        dNdtByV[sl2] = dNdtByV[sl2] - q; 
        dNdtByV[sr0] = dNdtByV[sr0] + q;
    }
    
}             


void inline OptReaction::update31
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sl2,
    double*  __restrict__ sr0
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sl2 = *sl2 - q;
    *sr0 = *sr0 + q;
}



void inline OptReaction::RF32
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];

    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp*this->Pow_pByRT_SumVki[1];
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1];

    const double q = (Kf*CF) - (Kr*CR);


    if(sameSpecies[i]==0)
    {
        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sl2 = &dNdtByV[sl2];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        update32(q,Sl0,Sl1,Sl2,Sr0,Sr1);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sl1] = dNdtByV[sl1] - q; 
        dNdtByV[sl2] = dNdtByV[sl2] - q; 
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q;
    }
}             

void inline OptReaction::update32
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sl2,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sl2 = *sl2 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
}


void inline OptReaction::RF33
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{
    const unsigned int sl0 = lhsSpeciesIndex[lhsOffset[i]+0];
    const unsigned int sl1 = lhsSpeciesIndex[lhsOffset[i]+1];
    const unsigned int sl2 = lhsSpeciesIndex[lhsOffset[i]+2];
    const unsigned int sr0 = rhsSpeciesIndex[rhsOffset[i]+0];
    const unsigned int sr1 = rhsSpeciesIndex[rhsOffset[i]+1];
    const unsigned int sr2 = rhsSpeciesIndex[rhsOffset[i]+2];
    double Kr = 0;

    if(this->isIrreversible[i]==0)
    {
        const double Kp = (ExpNegGbyRT[sr0]*ExpNegGbyRT[sr1]*ExpNegGbyRT[sr2])/(ExpNegGbyRT[sl0]*ExpNegGbyRT[sl1]*ExpNegGbyRT[sl2]);
        double Kc = Kp;
        Kc = std::max(Kc,1.4901171103413047e-8);
        Kr = Kf/Kc;   
    }
    else if(this->isIrreversible[i]==2)
    {
        unsigned int j = i - this->Ikf[1] + this->Ikf[9];
        Kr = this->Kf_[j];
    }

    const double CF = C[sl0]*C[sl1]*C[sl2];
    const double CR = C[sr0]*C[sr1];

    const double q = (Kf*CF) - (Kr*CR);
    if(sameSpecies[i]==0)
    {
        /*double r0 = dNdtByV[sl0];
        double r1 = dNdtByV[sl1];
        double r2 = dNdtByV[sl2];
        double r3 = dNdtByV[sr0];
        double r4 = dNdtByV[sr1];
        double r5 = dNdtByV[sr2];
        r0 -= q;
        r1 -= q;
        r2 -= q;
        r3 += q;
        r4 += q;
        r5 += q;
        dNdtByV[sl0] = r0; 
        dNdtByV[sl1] = r1;
        dNdtByV[sl2] = r2;
        dNdtByV[sr0] = r3;
        dNdtByV[sr1] = r4;
        dNdtByV[sr2] = r5;*/

        double* __restrict__ Sl0 = &dNdtByV[sl0];
        double* __restrict__ Sl1 = &dNdtByV[sl1];
        double* __restrict__ Sl2 = &dNdtByV[sl2];
        double* __restrict__ Sr0 = &dNdtByV[sr0];
        double* __restrict__ Sr1 = &dNdtByV[sr1];
        double* __restrict__ Sr2 = &dNdtByV[sr2];
        update33(q,Sl0,Sl1,Sl2,Sr0,Sr1,Sr2);
    }
    else if(sameSpecies[i]==1)
    {
        dNdtByV[sl0] = dNdtByV[sl0] - q; 
        dNdtByV[sl1] = dNdtByV[sl1] - q; 
        dNdtByV[sl2] = dNdtByV[sl2] - q; 
        dNdtByV[sr0] = dNdtByV[sr0] + q;
        dNdtByV[sr1] = dNdtByV[sr1] + q;
        dNdtByV[sr2] = dNdtByV[sr2] + q;
    }
   
}        


void inline OptReaction::update33
(
    double q,
    double*  __restrict__ sl0,
    double*  __restrict__ sl1,
    double*  __restrict__ sl2,
    double*  __restrict__ sr0,
    double*  __restrict__ sr1,
    double*  __restrict__ sr2
)const noexcept
{
    *sl0 = *sl0 - q;
    *sl1 = *sl1 - q;
    *sl2 = *sl2 - q;
    *sr0 = *sr0 + q;
    *sr1 = *sr1 + q;
    *sr2 = *sr2 + q;
}

void inline OptReaction::RFG
(
    unsigned int i,
    double Kf,
    const double* __restrict__ C,
    double*  __restrict__ dNdtByV,
    const double*  __restrict__ ExpNegGbyRT
)const noexcept
{


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
        this->update_Pow_pByRT_SumVki(T);
        Kc_ = Kp*this->Pow_pByRT_SumVki_[sumVki];
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

    const double q = Kf*CF-Kr*CR;

    for(unsigned int j = 0; j < lhsIndex[i].size();j++)
    {
        const unsigned int si = lhsIndex[i][j];
        dNdtByV[si] = dNdtByV[si] - q;
    }
    
    for(unsigned int j = 0; j < rhsIndex[i].size();j++)
    {
        const unsigned int si = rhsIndex[i][j];
        dNdtByV[si] = dNdtByV[si] + q;
    }       

}             

