/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2022 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/


#include "FastChemistryModel.H"
#include "UniformField.H"
#include "localEulerDdtScheme.H"
#include "cpuLoad.H"
#include "basicChemistryModel.H"

#include "OptReaction.H"
#include <chrono>
#include <vector>
#include <utility>
#include <algorithm>

#include <iostream>
#include <assert.h>


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::FastChemistryModel<ThermoType>::FastChemistryModel
(
    const fluidReactionThermo& thermo
)
:   basicChemistryModel(thermo),
    Yvf_(this->thermo().composition().Y()),
    nSpecie_(Yvf_.size()),
    n_(nSpecie_+1),
    jacobianType_
    (
        this->found("jacobian")
      ? jacobianTypeNames_.read(this->lookup("jacobian"))
      : jacobianType::exact
    ),
    mixture_(refCast<const multiComponentMixture<ThermoType>>(this->thermo())),
    specieThermos_(mixture_.specieThermos()),
    reaction(false),
    RR_(nSpecie_),
    Y_(nSpecie_),
    c_(nSpecie_),
    Treact(this->lookupOrDefault("Treact",0)),
    DLBthreshold(this->lookupOrDefault("DLBthreshold",1.0)),
    MaxIter(this->lookupOrDefault("Iter",1)),
    cpuLoadTransferTable(Pstream::nProcs()),
    CPUtimeField(thermo.T().mesh().C().size()),
    chemistryIntegrationTime(Pstream::nProcs()),
    sendBufferSize_(Pstream::nProcs()),
    recvBufferSize_(Pstream::nProcs()),
    sendBuffer_(Pstream::nProcs()),
    recvBuffer_(Pstream::nProcs()),
    recvBufPos_(Pstream::nProcs()),
    firstTime(true),
    skip(thermo.T().mesh().C().size(),false),
    IamBusyProcess(Pstream::nProcs(),true),
    Balance(this->lookupOrDefault("balance", false))
{

    const IOdictionary thermoDict
    (
        physicalProperties::findModelDict(this->mesh(), word::null)
    );
    const IOdictionary chemistryProperties
    (
        IOobject
        (
            "chemistryProperties",
            this->mesh().time().constant(),
            this->mesh(),
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );


    const word defaultSpecie = thermoDict.lookup("defaultSpecie");
    Info<<"The default specie is "<<defaultSpecie<<endl;
    this->defaultIndex = mixture_.species()[defaultSpecie];

    if(this->defaultIndex<0 ||this->defaultIndex>=this->nSpecie())
    {
        FatalErrorInFunction
                    << "Index of default species is wrong!"
                    << Foam::abort(FatalError);;
    }
    reaction.readInfo(chemistryProperties,thermoDict);

    // Create the fields for the chemistry sources
    forAll(RR_, fieldi)
    {
        RR_.set
        (
            fieldi,
            new volScalarField
            (
                IOobject
                (
                    "RR." + Yvf_[fieldi].name(),
                    this->mesh().time().timeName(),
                    this->mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                thermo.T().mesh(),
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            )
        );
    }

    Info<< "FastChemistryModel: Number of species = " << nSpecie_
        << " and reactions = " << nReaction() << endl;

    {
        
        size_t N = nSpecie()+1;
        
        size_t totalSize = 12*N + 3*N*N;
        size_t bytes = totalSize * sizeof(double);
        if (posix_memalign(reinterpret_cast<void**>(&this->buffer), 32, bytes))
        {
            throw std::bad_alloc();
        }
        std::memset(this->buffer, 0, bytes);
        size_t pos = 0;

        for (int i = 0; i < 12; i++)
        {
            YTpWork[i] = buffer + pos;
            pos   += N;
        }
        for (int i = 0; i < 3; i++)
        {
            YTpYTpWork[i] = buffer + pos;
            pos   += N * N;
        }
        assert(pos == totalSize);
    }
    forAll(cpuLoadTransferTable,i)
    {
        cpuLoadTransferTable[i].resize(Pstream::nProcs(),0);
    }

    
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::FastChemistryModel<ThermoType>::~FastChemistryModel()
{
    free(this->buffer);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::derivatives
(
    const scalar time,
    const scalarField& YTp,
    const label li,
    scalarField& dYTpdt
) const
{
}

template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::derivatives
(
    const scalar t,
    const label li,
    double* __restrict__ YT,    
    double* __restrict__ dYTdt,
    double* __restrict__ Cp,
    double* __restrict__ Ha,
    double* __restrict__ ExpGbyRT
) const
{
    int remain = nSpecie_%4;

    const double T = YT[nSpecie_];
    const volScalarField& p0vf = this->thermo().p().oldTime();
    double p = p0vf[li];
    
    double Ysum = 0;
    for (int i=0; i<nSpecie_; i++)
    {
        YT[i] = max(YT[i], 0);
        Ysum += YT[i];
    }
    
    double rhoM = 0;
    double RuTByP = reaction.Ru*T/p;
    __m256d RuTByPv = _mm256_set1_pd(RuTByP);
    __m256d rhoMv = _mm256_setzero_pd();
    for (int i=0; i<nSpecie_-remain; i=i+4)
    {
        __m256d YTpv = _mm256_loadu_pd(&YT[i+0]);
        __m256d invWv = _mm256_loadu_pd(&reaction.invW[i+0]);
        rhoMv = _mm256_fmadd_pd(_mm256_mul_pd(YTpv,invWv),RuTByPv,rhoMv);
    }
    for(int i = nSpecie_-remain; i<nSpecie_;i++)
    {
        rhoM += YT[i]*reaction.invW[i]*RuTByP;            
    }
    rhoM += reaction.hsum4(rhoMv);
    double invrhoM = rhoM;
    rhoM = 1/rhoM;

    for (label i=0; i<nSpecie_; i ++)
    {
        c_[i] = rhoM*reaction.invW[i]*YT[i];
    }

    std::memset(dYTdt, 0, n_ * sizeof(double));
    reaction.dNdtByV(p,T,c_.data(),dYTdt,ExpGbyRT,Cp,Ha);

    double CpM = 0;
    double dTdt = 0;

    __m256d CpMv = _mm256_setzero_pd();
    __m256d dTdtv = _mm256_setzero_pd();
    for (label i=0; i<nSpecie_-remain; i=i+4)
    {
        __m256d Wv = _mm256_loadu_pd(&reaction.W[i]);
        __m256d dYTdtv = _mm256_loadu_pd(&dYTdt[i]);
        __m256d invrhoMv = _mm256_set1_pd(invrhoM);
        dYTdtv = _mm256_mul_pd(_mm256_mul_pd(Wv,invrhoMv),dYTdtv);
        _mm256_storeu_pd(&dYTdt[i],dYTdtv);

        __m256d YTv = _mm256_loadu_pd(&YT[i]);
        __m256d Cpv = _mm256_loadu_pd(&Cp[i]);
        CpMv = _mm256_fmadd_pd(YTv,Cpv,CpMv);
        __m256d Hav = _mm256_loadu_pd(&Ha[i]);
        dTdtv = _mm256_fmadd_pd(Hav,dYTdtv,dTdtv);
    }
    for(label i = nSpecie_-remain;i<nSpecie_;i++)
    {
        dYTdt[i] =dYTdt[i]*reaction.W[i]/rhoM;
        CpM += YT[i]*Cp[i];
        dTdt -= dYTdt[i]*Ha[i];
    }
    CpM = CpM + reaction.hsum4(CpMv);
    dTdt = dTdt -(reaction.hsum4(dTdtv));
    dTdt /= CpM;
    dYTdt[nSpecie_] = dTdt;
}

template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::derivativesC
(
    const scalar t,
    const scalarField& __restrict__ CTp,
    const label li,
    scalarField& __restrict__ dCTpdt,
    scalarField& __restrict__ Cp,
    scalarField& __restrict__ Ha,
    scalarField& __restrict__ ExpGbyRT
) const
{
        /*const scalar T = CTp[nSpecie_];
        const volScalarField& p0vf = this->thermo().p().oldTime();
        double p = p0vf[li];
        
        // Evaluate the concentrations
        for (label i=0; i<nSpecie_; i ++)
        {
            c_[i] = CTp[i];
        }

        // Evaluate contributions from reactions
        dCTpdt = Zero;
        reaction.dNdtByV
        (
            p,
            T,
            c_.data(),
            dCTpdt.data(),
            ExpGbyRT.data(),
            Cp.data(),
            Ha.data()
        );



        // dT/dt = ...
        double rho = 0;
        double cSum = 0;
        for (int i = 0; i < nSpecie_; i++)
        {
            const double W = reaction.W[i];
            cSum += c_[i];
            rho += W*c_[i];
        }
        double cp = 0;
        for (int i=0; i<nSpecie_; i++)
        {
            cp += c_[i]*Cp[i];
        }
        cp /= rho;

        double dT = 0;
        for (int i = 0; i < nSpecie_; i++)
        {
            const double hi = Ha[i];
            dT += hi*dCTpdt[i];
        }
        dT /= rho*cp;

        dCTpdt[nSpecie_] = -dT;*/
}

template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::derivatives
(
    const scalar t,
    const label li,
    double* __restrict__ YT,    
    double* __restrict__ dYTdt,
    double* __restrict__ Cp,
    double* __restrict__ Ha
) const
{
    int remain = nSpecie_%4;
    const double T = YT[nSpecie_];
    const volScalarField& p0vf = this->thermo().p().oldTime();
    double p = p0vf[li];
    
    double Ysum = 0;
    for (int i=0; i<nSpecie_; i++)
    {
        YT[i] = max(YT[i], 0);
        Ysum += YT[i];
    }

    double rhoM = 0;
    double RuTByP = reaction.Ru*T/p;
    __m256d RuTByPv = _mm256_set1_pd(RuTByP);
    __m256d rhoMv = _mm256_setzero_pd();
    for (int i=0; i<nSpecie_-remain; i=i+4)
    {
        __m256d YTpv = _mm256_loadu_pd(&YT[i+0]);
        __m256d invWv = _mm256_loadu_pd(&reaction.invW[i+0]);
        rhoMv = _mm256_fmadd_pd(_mm256_mul_pd(YTpv,invWv),RuTByPv,rhoMv);
    }
    for(int i = nSpecie_-remain; i<nSpecie_;i++)
    {
        rhoM += YT[i]*reaction.invW[i]*RuTByP;            
    }
    rhoM += reaction.hsum4(rhoMv);
    double invrhoM = rhoM;
    rhoM = 1/rhoM;

    for (label i=0; i<nSpecie_; i ++)
    {
        c_[i] = rhoM*reaction.invW[i]*YT[i];
    }

    std::memset(dYTdt, 0, n_ * sizeof(double));
    reaction.dNdtByV(p,T,c_.data(),dYTdt,Cp);
    reaction.CpHa(T,Cp,Ha);

    double CpM = 0;
    double dTdt = 0;
    __m256d CpMv = _mm256_setzero_pd();
    __m256d dTdtv = _mm256_setzero_pd();
    for (label i=0; i<nSpecie_-remain; i=i+4)
    {
        __m256d Wv = _mm256_loadu_pd(&reaction.W[i]);
        __m256d dYTdtv = _mm256_loadu_pd(&dYTdt[i]);
        __m256d invrhoMv = _mm256_set1_pd(invrhoM);
        dYTdtv = _mm256_mul_pd(_mm256_mul_pd(Wv,invrhoMv),dYTdtv);
        _mm256_storeu_pd(&dYTdt[i],dYTdtv);

        __m256d YTv = _mm256_loadu_pd(&YT[i]);
        __m256d Cpv = _mm256_loadu_pd(&Cp[i]);
        CpMv = _mm256_fmadd_pd(YTv,Cpv,CpMv);
        __m256d Hav = _mm256_loadu_pd(&Ha[i]);
        dTdtv = _mm256_fmadd_pd(Hav,dYTdtv,dTdtv);
    }
    for(label i = nSpecie_-remain;i<nSpecie_;i++)
    {
        dYTdt[i] =dYTdt[i]*reaction.W[i]/rhoM;
        CpM += YT[i]*Cp[i];
        dTdt -= dYTdt[i]*Ha[i];
    }
    CpM = CpM + reaction.hsum4(CpMv);
    dTdt = dTdt -(reaction.hsum4(dTdtv));
    dTdt /= CpM;
    dYTdt[nSpecie_] = dTdt;
}


template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::derivativesC
(
    const scalar t,
    const scalarField& __restrict__ CTp,
    const label li,
    scalarField& __restrict__ dCTpdt,
    scalarField& __restrict__ Cp,
    scalarField& __restrict__ Ha
) const
{
        /*const double T = CTp[nSpecie_];
        const volScalarField& p0vf = this->thermo().p().oldTime();
        double p = p0vf[li];
        

        // Evaluate the concentrations
        for (label i=0; i<nSpecie_; i ++)
        {
            c_[i] = CTp[i];
        }

        // Evaluate contributions from reactions
        dCTpdt = Zero;

        //scalarField& ExpGbyRT = YTpWork_[5];
        //scalarField& Cp = YTpWork_[6];
        //scalarField& Ha = YTpWork_[7];

        scalarField& ExpGbyRT = Cp;

        reaction.dNdtByV
        (
            p,
            T,
            c_.data(),
            dCTpdt.data(),
            ExpGbyRT.data()
        );

        reaction.CpHa(T,Cp.data(),Ha.data());

        double rho = 0;
        double cSum = 0;
        for (label i = 0; i < nSpecie_; i++)
        {
            const double W = reaction.W[i];
            cSum += c_[i];
            rho += W*c_[i];
        }
        double cp = 0;
        for (label i=0; i<nSpecie_; i++)
        {
            cp += c_[i]*Cp[i];
        }
        cp /= rho;

        double dT = 0;
        for (label i = 0; i < nSpecie_; i++)
        {
            const double hi = Ha[i];
            dT += hi*dCTpdt[i];
        }
        dT /= rho*cp;

        dCTpdt[nSpecie_] = -dT;*/
}

template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::jacobian
(
    const scalar t,
    double* __restrict__ YT_,
    const label li,
    double* __restrict__ dYTdt,
    double* __restrict__ Jy
) const 
{
    forAll(c_, i)
    {
        YT_[i] = max(YT_[i], 0);
    }

    const double T = YT_[nSpecie_];
    const volScalarField& p0vf = this->thermo().p().oldTime();
    double p = p0vf[li];
        
    double* __restrict__ Jc = YTpYTpWork[0];
    {size_t size = n_*n_;std::memset(Jc, 0, size * sizeof(double));}
    {size_t size = n_;std::memset(dYTdt, 0, size * sizeof(double));}

    double* __restrict__ ExpNegGbyRT = YTpWork[3];
    double* __restrict__ dBdT        = YTpWork[4];
    double* __restrict__ dCpdT       = YTpWork[5];
    double* __restrict__ Cp          = YTpWork[6];
    double* __restrict__ Ha          = YTpWork[7];
    double* __restrict__ WiByrhoM    = YTpWork[8];
    double* __restrict__ rhoMvj      = YTpWork[10];
       
    reaction.ddNdtByVdcTp
    (
        p,
        T,
        YT_,
        c_.data(),
        dYTdt,
        ExpNegGbyRT,
        dBdT,
        dCpdT,
        Cp,
        Ha,
        rhoMvj,
        WiByrhoM,
        Jc,
        true
    );

    unsigned int remain = reaction.nSpecies%4;
    switch (jacobianType_)
    {
        case jacobianType::fast:
            reaction.FastddYdtdY_Vec(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,dYTdt,YT_,Jy);
            reaction.ddYdtdTP_Vec(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,c_.data(),dYTdt,YT_,Jy);
            reaction.ddTdtdYT_Vec(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,Cp,dCpdT,Ha,dYTdt,YT_,Jy);                
            break;
        case jacobianType::exact:
        if(remain==0)
        {
            reaction.ddYdtdY_Vec1_0(Jc,rhoMvj,WiByrhoM,dYTdt,YT_,Jy);
            reaction.ddYdtdTP_Vec_0(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,c_.data(),dYTdt,YT_,Jy);
            reaction.ddTdtdYT_Vec_0(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,Cp,dCpdT,Ha,dYTdt,YT_,Jy);
        }
        else if(remain==1)
        {
            reaction.ddYdtdY_Vec1_1(Jc,rhoMvj,WiByrhoM,dYTdt,YT_,Jy);
            reaction.ddYdtdTP_Vec_1(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,c_.data(),dYTdt,YT_,Jy);  
            reaction.ddTdtdYT_Vec_1(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,Cp,dCpdT,Ha,dYTdt,YT_,Jy);
        }
        else if(remain==2)
        {
            reaction.ddYdtdY_Vec1_2(Jc,rhoMvj,WiByrhoM,dYTdt,YT_,Jy);
            reaction.ddYdtdTP_Vec_2(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,c_.data(),dYTdt,YT_,Jy);
            reaction.ddTdtdYT_Vec_2(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,Cp,dCpdT,Ha,dYTdt,YT_,Jy);
        }
        else
        {
            reaction.ddYdtdY_Vec1_3(Jc,rhoMvj,WiByrhoM,dYTdt,YT_,Jy);
            reaction.ddYdtdTP_Vec_3(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,c_.data(),dYTdt,YT_,Jy); 
            reaction.ddTdtdYT_Vec_3(ExpNegGbyRT,Jc,rhoMvj,WiByrhoM,Cp,dCpdT,Ha,dYTdt,YT_,Jy);
        }
        break;
    }
}


template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::jacobian
(
    const scalar t,
    const scalarField& __restrict__ YTp_,
    const label li,
    scalarField& __restrict__ dcdt,
    scalarSquareMatrix& J,
    int Placeholder
) const
{}


template<class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::FastChemistryModel<ThermoType>::tc() const
{
    tmp<volScalarField> ttc
    (
        volScalarField::New
        (
            "tc",
            this->mesh(),
            dimensionedScalar(dimTime, small),
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
    );
    scalarField& tc = ttc.ref();

    tmp<volScalarField> trho(this->thermo().rho());
    const scalarField& rho = trho();

    const scalarField& T = this->thermo().T();
    const scalarField& p = this->thermo().p();

    if (this->chemistry_)
    {
        reactionEvaluationScope scope(*this);

        forAll(rho, celli)
        {
            const scalar rhoi = rho[celli];
            const scalar Ti = T[celli];
            const scalar pi = p[celli];

            for (label i=0; i<nSpecie_; i++)
            {
                c_[i] = rhoi*Yvf_[i][celli]/specieThermos_[i].W();
            }

            // A reaction's rate scale is calculated as it's molar
            // production rate divided by the total number of moles in the
            // system.
            //
            // The system rate scale is the average of the reactions' rate
            // scales weighted by the reactions' molar production rates. This
            // weighting ensures that dominant reactions provide the largest
            // contribution to the system rate scale.
            //
            // The system time scale is then the reciprocal of the system rate
            // scale.
            //
            // Contributions from forward and reverse reaction rates are
            // handled independently and identically so that reversible
            // reactions produce the same result as the equivalent pair of
            // irreversible reactions.
            scalar sumW = 0, sumWRateByCTot = 0;
            reaction.Tc(celli,pi,Ti,c_.data(),sumW,sumWRateByCTot);
            double sumc = sum(c_);
            tc[celli] =
                sumWRateByCTot == 0 ? vGreat : sumW/sumWRateByCTot*sumc;

        }
    }
    ttc.ref().correctBoundaryConditions();
    return ttc;
}


template<class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::FastChemistryModel<ThermoType>::Qdot() const
{
    tmp<volScalarField> tQdot
    (
        volScalarField::New
        (
            "Qdot",
            this->mesh_,
            dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
        )
    );

    if (this->chemistry_)
    {
        reactionEvaluationScope scope(*this);

        scalarField& Qdot = tQdot.ref();

        forAll(Yvf_, i)
        {
            forAll(Qdot, celli)
            {
                const scalar hi = specieThermos_[i].Hf();
                Qdot[celli] -= hi*RR_[i][celli];
            }
        }
    }

    return tQdot;
}


template<class ThermoType>
Foam::tmp<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::FastChemistryModel<ThermoType>::calculateRR
(
    const label ri,
    const label si
) const
{

    FatalErrorInFunction
                    << "This function is not supported and should not be used"
                    << Foam::abort(FatalError);

    tmp<volScalarField::Internal> tRR
    (
        volScalarField::Internal::New
        (
            "RR",
            this->mesh(),
            dimensionedScalar(dimMass/dimVolume/dimTime, 0)
        )
    );
    return tRR;
}


template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::calculate()
{

    if (!this->chemistry_)
    {
        return;
    }

    tmp<volScalarField> trho(this->thermo().rho());
    const scalarField& rho = trho();

    const scalarField& T = this->thermo().T();
    const scalarField& p = this->thermo().p();

    double* dNdtByV = YTpWork[3];
    double* ExpGbyRT = YTpWork[4];
    double* Cp = YTpWork[5];
    double* Ha = YTpWork[6];

    reactionEvaluationScope scope(*this);

    forAll(rho, celli)
    {
        const scalar rhoi = rho[celli];
        const scalar Ti = T[celli];
        const scalar pi = p[celli];

        for (label i=0; i<nSpecie_; i++)
        {
            const scalar Yi = Yvf_[i][celli];
            c_[i] = rhoi*Yi/specieThermos_[i].W();
        }
        std::memset(dNdtByV, 0, n_ * sizeof(double));
        reaction.dNdtByV(pi,Ti,c_.data(),dNdtByV,ExpGbyRT,Cp,Ha);
        for (label i=0; i<nSpecie_; i++)
        {
            RR_[i][celli] = dNdtByV[i]*specieThermos_[i].W();
        }
    }
    return;
}


#include "FastChemistryModel_transientSolve.H"
#include "FastChemistryModel_localEulerSolve.H"
template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::exchange
(
    const UList<DynamicList<char>>& sendBufs,
    const List<std::streamsize>& recvSizes,
    List<DynamicList<char>>& recvBufs,
    const int tag,
    const label comm,
    const bool block
)
{
    if (!contiguous<char>())
    {
        FatalErrorInFunction
            << "Continuous data only." << sizeof(char) << Foam::abort(FatalError);
    }

    if (sendBufs.size() != UPstream::nProcs(comm))
    {
        FatalErrorInFunction
            << "Size of list " << sendBufs.size()
            << " does not equal the number of processors "
            << UPstream::nProcs(comm)
            << Foam::abort(FatalError);
    }

    recvBufs.setSize(sendBufs.size());

    if (UPstream::parRun() && UPstream::nProcs(comm) > 1)
    {
        label startOfRequests = Pstream::nRequests();

        forAll(recvSizes, proci)
        {
            std::streamsize nRecv = recvSizes[proci]; 


            if (proci != Pstream::myProcNo(comm) && nRecv > 0)
            {

                recvBufs[proci].setSize(static_cast<Foam::label>(nRecv)); 
                UIPstream::read
                (
                    UPstream::commsTypes::nonBlocking,
                    proci,
                    reinterpret_cast<char*>(recvBufs[proci].begin()),
                    nRecv*sizeof(char),
                    tag,
                    comm
                );
            }
        }

        forAll(sendBufs, proci)
        {
            if (proci != Pstream::myProcNo(comm) && sendBufs[proci].size() > 0)
            {

                if
                (
                   !UOPstream::write
                    (
                        UPstream::commsTypes::nonBlocking,
                        proci,
                        reinterpret_cast<const char*>(sendBufs[proci].begin()),
                        sendBufs[proci].size()*sizeof(char),
                        tag,
                        comm
                    )
                )
                {
                    FatalErrorInFunction
                        << "Cannot send outgoing message. "
                        << "to:" << proci << " nBytes:"
                        << label(sendBufs[proci].size()*sizeof(char))
                        << Foam::abort(FatalError);
                }
            }
        }

        if (block)
        {
            Pstream::waitRequests(startOfRequests); 
        }
    }

    recvBufs[Pstream::myProcNo(comm)] = sendBufs[Pstream::myProcNo(comm)];
}
template<class ThermoType>
void Foam::FastChemistryModel<ThermoType>::exchangeSizes
(
    const UList<DynamicList<char>>& sendBufs,
    labelList& recvSizes,
    const label comm
)
{
    if (sendBufs.size() != UPstream::nProcs(comm))
    {
        FatalErrorInFunction
            << "Size of container " << sendBufs.size()
            << " does not equal the number of processors "
            << UPstream::nProcs(comm)
            << Foam::abort(FatalError);
    }

    labelList sendSizes(sendBufs.size());
    forAll(sendBufs, proci)
    {
        sendSizes[proci] = sendBufs[proci].size();
    }
    recvSizes.setSize(sendSizes.size());
    Foam::UPstream::allToAll(sendSizes, recvSizes, comm);
    Pout<<"exchangesize: recvSizes: "<<recvSizes<<endl;
}



// ************************************************************************* //
