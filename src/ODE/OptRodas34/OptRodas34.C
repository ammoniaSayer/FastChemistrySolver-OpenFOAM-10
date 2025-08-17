/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2021 OpenFOAM Foundation
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

#include "OptRodas34.H"
#include "SubField.H"
#include "addToRunTimeSelectionTable.H"
#include <chrono>
#include <immintrin.h>  
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ChemistryModel>
Foam::OptRodas34<ChemistryModel>::OptRodas34
(
    const fluidReactionThermo& thermo
)
:
    chemistrySolver<ChemistryModel>(thermo),
    coeffsDict_(this->subDict("OptRodas34Coeffs")),
    absTol_(coeffsDict_.lookup<scalar>("absTol")),
    relTol_(coeffsDict_.lookup<scalar>("relTol")),
    maxSteps_(coeffsDict_.lookupOrDefault("maxSteps",10000)),
    n_(this->nSpecie()+1),
    cTp_(n_),
    pivotIndices_(n_),
    LU(this->YTpYTpWork[1],n_)
{}
// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ChemistryModel>
Foam::OptRodas34<ChemistryModel>::~OptRodas34()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
template<class ChemistryModel>
void Foam::OptRodas34<ChemistryModel>::solve
(
    scalar& __restrict__ p,
    scalar& __restrict__ T,
    scalarField& y,
    const label li,
    scalar& __restrict__ deltaT,
    scalar& __restrict__ subDeltaT
) const
{

    double* __restrict__ y00 = this->YTpWork[0];
    double* __restrict__ y0  = this->YTpWork[1];


    for (int i=0; i<this->nSpecie(); i++)
    {
        y00[i] = max(0, y[i]);
    }
    y00[this->nSpecie()] = T;

    for (label i=0; i<n_; i++)
    {
        y0[i] = y00[i];
    }
    double* __restrict__ yTemp = this->YTpWork[2];
    double* __restrict__ k1 = this->YTpWork[3];
    double* __restrict__ k2 = this->YTpWork[4];
    double* __restrict__ k3 = this->YTpWork[5];
    double* __restrict__ k4 = this->YTpWork[6];
    double* __restrict__ k5 = this->YTpWork[7];
    double* __restrict__ dy = this->YTpWork[8];
    double* __restrict__ err = this->YTpWork[9];
    double* __restrict__ dydx = this->YTpWork[10];
    double* __restrict__ dfdx = this->YTpWork[11];
    double* __restrict__ a_ = this->YTpYTpWork[1];

    this->ODESolve
    (
        0,              
        deltaT,         
        li,             
        subDeltaT,      
        y0,              
        yTemp,
        k1,
        k2,
        k3,
        k4,
        k5,
        dy,
        err,
        dydx,
        dfdx,
        a_        
    );

    for (int i=0; i<this->nSpecie(); i++)
    {
        y[i] = max(0.0, y0[i]);
    }
    T = y0[this->nSpecie()];
}


template<class ChemistryModel>
void Foam::OptRodas34<ChemistryModel>::ODESolve
(
    const scalar xStart,
    const scalar xEnd,
    const label li,
    scalar& dxTry,
    double* __restrict__ y0,
    double* __restrict__ yTemp,    
    double* __restrict__ k1,
    double* __restrict__ k2,
    double* __restrict__ k3,
    double* __restrict__ k4,
    double* __restrict__ k5,
    double* __restrict__ dy,
    double* __restrict__ err,
    double* __restrict__ dydx,
    double* __restrict__ dfdx,
    double* __restrict__ a_    
) const
{

    stepState step(dxTry);
    scalar x = xStart;

    for (label nStep=0; nStep<maxSteps_; nStep++)
    {
        // Store previous iteration dxTry
        scalar dxTry0 = step.dxTry;

        // Check if this is a truncated step and set dxTry to integrate to xEnd
        if ((x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0)
        {
            step.last = true;
            step.dxTry = xEnd - x;
        }

        // Integrate as far as possible up to step.dxTry
        //solve(x, y, li, step);
        adaptiveSolve
        (
            x,              
            li,             
            step.dxTry,     
            y0,             
            yTemp,
            k1,
            k2,
            k3,
            k4,
            k5,
            dy,
            err,
            dydx,
            dfdx,
            a_     
        );

        // Check if reached xEnd
        if ((x - xEnd)*(xEnd - xStart) >= 0)
        {
            if (nStep > 0 && step.last)
            {
                step.dxTry = dxTry0;
            }

            dxTry = step.dxTry;

            return;
        }
    }
    FatalErrorInFunction
        << "Integration steps greater than maximum " << maxSteps_ << nl
        << exit(FatalError);

}


template<class ChemistryModel>
void Foam::OptRodas34<ChemistryModel>::adaptiveSolve
(
    scalar& __restrict__ x,
    const label li,
    scalar& __restrict__ dxTry,
    double* __restrict__ y0,
    double* __restrict__ yTemp,
    double* __restrict__ k1,
    double* __restrict__ k2,
    double* __restrict__ k3,
    double* __restrict__ k4,
    double* __restrict__ k5,
    double* __restrict__ dy,
    double* __restrict__ err,
    double* __restrict__ dydx,
    double* __restrict__ dfdx,
    double* __restrict__ a_
) const
{

    scalar dx = dxTry;
    scalar Err = 0.0;
    scalar invdx = 1.0/dx;

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error

    Err = Rodas34Solve
    (
        x,
        li,
        dx,
        invdx,
        y0,
        yTemp,
        k1,
        k2,
        k3,
        k4,
        k5,
        dy,
        err,
        dydx,
        dfdx,
        a_     
    );
    while(Err > 1)
    {

        scalar scale = max(safeScale_*pow(Err, -alphaDec_), minScale_);
        dx *= scale;
        invdx = 1.0/dx;

        Err = Rodas34Solve
        (
            x,
            li,
            dx,
            invdx,
            y0,
            yTemp,
            k1,
            k2,
            k3,
            k4,
            k5,
            dy,
            err,
            dydx,
            dfdx,
            a_     
        );
    } 

    // Update the state
    x += dx;
    for(label i = 0; i < n_;i++)
    {
        y0[i] = yTemp[i];
    }

    dxTry = (Err>ratio)?
            min(max(safeScale_*pow(Err, -alphaInc_), minScale_), maxScale_)*dx
            :safeMaxScale*dx;

}



template<class ChemistryModel>
Foam::scalar Foam::OptRodas34<ChemistryModel>::Rodas34Solve
(
    const scalar x0,
    const label li,
    const scalar dx,
    const scalar invdx,
    double* __restrict__ y0,    
    double* __restrict__ yTemp,
    double* __restrict__ k1,
    double* __restrict__ k2,
    double* __restrict__ k3,
    double* __restrict__ k4,
    double* __restrict__ k5,
    double* __restrict__ dy,
    double* __restrict__ err,
    double* __restrict__ dydx,
    double* __restrict__ dfdx,
    double* __restrict__ a_
) const
{

    this->jacobian(x0, y0, li, dfdx, a_);

    {
        const label NN = n_*n_;
        label remain = NN%16;
        for(label i = 0 ; i<NN-remain;i=i+16)
        {
            __m256d Av0 = _mm256_loadu_pd(&a_[i+0]);
            __m256d Av1 = _mm256_loadu_pd(&a_[i+4]);
            __m256d Av2 = _mm256_loadu_pd(&a_[i+8]);
            __m256d Av3 = _mm256_loadu_pd(&a_[i+12]);

            _mm256_storeu_pd(&a_[i+0],-Av0);
            _mm256_storeu_pd(&a_[i+4],-Av1);
            _mm256_storeu_pd(&a_[i+8],-Av2);
            _mm256_storeu_pd(&a_[i+12],-Av3);
        }
        for(label i = NN-remain; i < NN;i++)
        {
            a_[i] = -a_[i];
        }
    }
    for (label i=0; i<n_; i++)
    {
        a_[i*n_+i] += 1.0*invdx*Invgamma;
    }

    LU.Block4LUDecompose();
          
    // Calculate k1:

    unsigned int remain = n_%4;
    {
        const double dxd1 = dx*d1+1;
        __m256d dxd1v = _mm256_set1_pd(dxd1);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dfdxv = _mm256_loadu_pd(&dfdx[i+0]);
            __m256d k1v = _mm256_mul_pd(dxd1v,dfdxv);
            _mm256_storeu_pd(&k1[i+0],k1v);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            k1[i] = dxd1*dfdx[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            k1[i+0] = dxd1*dfdx[i+0];
            k1[i+1] = dxd1*dfdx[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            k1[i+0] = dxd1*dfdx[i+0];
            k1[i+1] = dxd1*dfdx[i+1];
            k1[i+2] = dxd1*dfdx[i+2];
        }
    }

    LU.xSolve(k1);

    {
        __m256d a21v = _mm256_set1_pd(a21);
        for(unsigned int i = 0 ;i < n_ - remain;i=i+4)
        {
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d yTempv = _mm256_fmadd_pd(a21v,k1v,y0v);
            _mm256_storeu_pd(&yTemp[i],yTempv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            yTemp[i] = y0[i] + a21*k1[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            yTemp[i+0] = y0[i+0] + a21*k1[i+0];
            yTemp[i+1] = y0[i+1] + a21*k1[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            yTemp[i+0] = y0[i+0] + a21*k1[i+0];
            yTemp[i+1] = y0[i+1] + a21*k1[i+1];
            yTemp[i+2] = y0[i+2] + a21*k1[i+2];
        }
    }

    this->derivatives(x0 + c2*dx, li, yTemp, dydx, k2, k3, k4);

    {
        double dxd2 = dx*d2;
        double c21invdx = c21*invdx;
        __m256d dxd2v = _mm256_set1_pd(dxd2);
        __m256d c21invdxv = _mm256_set1_pd(c21invdx);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dydxv = _mm256_loadu_pd(&dydx[i]);
            __m256d dfdxv = _mm256_loadu_pd(&dfdx[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_fmadd_pd(dxd2v,dfdxv,dydxv);
            k2v = _mm256_fmadd_pd(c21invdxv,k1v,k2v);
            _mm256_storeu_pd(&k2[i],k2v);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            k2[i] = dydx[i] + dxd2*dfdx[i] + c21invdx*k1[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            k2[i+0] = dydx[i+0] + dxd2*dfdx[i+0] + c21invdx*k1[i+0];
            k2[i+1] = dydx[i+1] + dxd2*dfdx[i+1] + c21invdx*k1[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            k2[i+0] = dydx[i+0] + dxd2*dfdx[i+0] + c21invdx*k1[i+0];
            k2[i+1] = dydx[i+1] + dxd2*dfdx[i+1] + c21invdx*k1[i+1];
            k2[i+2] = dydx[i+2] + dxd2*dfdx[i+2] + c21invdx*k1[i+2];
        }
    }

    LU.xSolve(k2);

    {
        __m256d a31v = _mm256_set1_pd(a31);
        __m256d a32v = _mm256_set1_pd(a32);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d yTempv = _mm256_fmadd_pd(a31v,k1v,y0v);
            yTempv = _mm256_fmadd_pd(a32v,k2v,yTempv);
            _mm256_storeu_pd(&yTemp[i],yTempv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            yTemp[i] = y0[i] + a31*k1[i] + a32*k2[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            yTemp[i+0] = y0[i+0] + a31*k1[i+0] + a32*k2[i+0];
            yTemp[i+1] = y0[i+1] + a31*k1[i+1] + a32*k2[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            yTemp[i+0] = y0[i+0] + a31*k1[i+0] + a32*k2[i+0];
            yTemp[i+1] = y0[i+1] + a31*k1[i+1] + a32*k2[i+1];
            yTemp[i+2] = y0[i+2] + a31*k1[i+2] + a32*k2[i+2];
        }
    }

    this->derivatives(x0 + c3*dx, li, yTemp, dydx, k3, k4, k5);

    {
        double dxd3 = dx*d3;
        double c31invdx = c31*invdx;
        double c32invdx = c32*invdx;
        __m256d dxd3v = _mm256_set1_pd(dxd3);
        __m256d c31invdxv = _mm256_set1_pd(c31invdx);
        __m256d c32invdxv = _mm256_set1_pd(c32invdx);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dydxv = _mm256_loadu_pd(&dydx[i]);
            __m256d dfdxv = _mm256_loadu_pd(&dfdx[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_fmadd_pd(dxd3v,dfdxv,dydxv);
            k3v = _mm256_fmadd_pd(c31invdxv,k1v,k3v);
            k3v = _mm256_fmadd_pd(c32invdxv,k2v,k3v);
            _mm256_storeu_pd(&k3[i],k3v);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            k3[i] = dydx[i] + dxd3*dfdx[i] + c31invdx*k1[i] + c32invdx*k2[i];   
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            k3[i+0] = dydx[i+0] + dxd3*dfdx[i+0] + c31invdx*k1[i+0] + c32invdx*k2[i+0];
            k3[i+1] = dydx[i+1] + dxd3*dfdx[i+1] + c31invdx*k1[i+1] + c32invdx*k2[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            k3[i+0] = dydx[i+0] + dxd3*dfdx[i+0] + c31invdx*k1[i+0] + c32invdx*k2[i+0];
            k3[i+1] = dydx[i+1] + dxd3*dfdx[i+1] + c31invdx*k1[i+1] + c32invdx*k2[i+1];
            k3[i+2] = dydx[i+2] + dxd3*dfdx[i+2] + c31invdx*k1[i+2] + c32invdx*k2[i+2];
        }
    }

    LU.xSolve(k3);

    {
        __m256d a41v = _mm256_set1_pd(a41);
        __m256d a42v = _mm256_set1_pd(a42);
        __m256d a43v = _mm256_set1_pd(a43);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_loadu_pd(&k3[i]);
            __m256d yTempv = _mm256_fmadd_pd(a41v,k1v,y0v);
            yTempv = _mm256_fmadd_pd(a42v,k2v,yTempv);
            yTempv = _mm256_fmadd_pd(a43v,k3v,yTempv);
            _mm256_storeu_pd(&yTemp[i],yTempv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            yTemp[i] = y0[i] + a41*k1[i] + a42*k2[i] + a43*k3[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            yTemp[i+0] = y0[i+0] + a41*k1[i+0] + a42*k2[i+0] + a43*k3[i+0];
            yTemp[i+1] = y0[i+1] + a41*k1[i+1] + a42*k2[i+1] + a43*k3[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            yTemp[i+0] = y0[i+0] + a41*k1[i+0] + a42*k2[i+0] + a43*k3[i+0];
            yTemp[i+1] = y0[i+1] + a41*k1[i+1] + a42*k2[i+1] + a43*k3[i+1];
            yTemp[i+2] = y0[i+2] + a41*k1[i+2] + a42*k2[i+2] + a43*k3[i+2];            
        }
    }

    this->derivatives(x0 + c4*dx, li, yTemp, dydx, k4, k5, dy);

    {
        double dxd4 = dx*d4;
        double c41invdx = c41*invdx;
        double c42invdx = c42*invdx;
        double c43invdx = c43*invdx;
        __m256d dxd4v = _mm256_set1_pd(dxd4);
        __m256d c41invdxv = _mm256_set1_pd(c41invdx);
        __m256d c42invdxv = _mm256_set1_pd(c42invdx);
        __m256d c43invdxv = _mm256_set1_pd(c43invdx);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dydxv = _mm256_loadu_pd(&dydx[i]);
            __m256d dfdxv = _mm256_loadu_pd(&dfdx[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_loadu_pd(&k3[i]);

            __m256d k4v = _mm256_fmadd_pd(dxd4v,dfdxv,dydxv);
            k4v = _mm256_fmadd_pd(c41invdxv,k1v,k4v);
            k4v = _mm256_fmadd_pd(c42invdxv,k2v,k4v);
            k4v = _mm256_fmadd_pd(c43invdxv,k3v,k4v);
            _mm256_storeu_pd(&k4[i],k4v);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            k4[i] = dydx[i] + dxd4*dfdx[i] + c41invdx*k1[i] + c42invdx*k2[i] + c43invdx*k3[i];         
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            k4[i+0] = dydx[i+0] + dxd4*dfdx[i+0] + c41invdx*k1[i+0] + c42invdx*k2[i+0] + c43invdx*k3[i+0];
            k4[i+1] = dydx[i+1] + dxd4*dfdx[i+1] + c41invdx*k1[i+1] + c42invdx*k2[i+1] + c43invdx*k3[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            k4[i+0] = dydx[i+0] + dxd4*dfdx[i+0] + c41invdx*k1[i+0] + c42invdx*k2[i+0] + c43invdx*k3[i+0];
            k4[i+1] = dydx[i+1] + dxd4*dfdx[i+1] + c41invdx*k1[i+1] + c42invdx*k2[i+1] + c43invdx*k3[i+1];
            k4[i+2] = dydx[i+2] + dxd4*dfdx[i+2] + c41invdx*k1[i+2] + c42invdx*k2[i+2] + c43invdx*k3[i+2];
        }
    }



    LU.xSolve(k4);

    // Calculate k5:
    {

        __m256d a51v = _mm256_set1_pd(a51);
        __m256d a52v = _mm256_set1_pd(a52);
        __m256d a53v = _mm256_set1_pd(a53);
        __m256d a54v = _mm256_set1_pd(a54);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_loadu_pd(&k3[i]);
            __m256d k4v = _mm256_loadu_pd(&k4[i]);

            __m256d dyv = _mm256_mul_pd(a51v,k1v);
            dyv = _mm256_fmadd_pd(a52v,k2v,dyv);
            dyv = _mm256_fmadd_pd(a53v,k3v,dyv);
            dyv = _mm256_fmadd_pd(a54v,k4v,dyv);
            _mm256_storeu_pd(&dy[i],dyv);
            __m256d yTempv = _mm256_add_pd(y0v,dyv);
            _mm256_storeu_pd(&yTemp[i],yTempv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            dy[i] = a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i];
            yTemp[i] = y0[i] + dy[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            dy[i+0] = a51*k1[i+0] + a52*k2[i+0] + a53*k3[i+0] + a54*k4[i+0];
            yTemp[i+0] = y0[i+0] + dy[i+0];

            dy[i+1] = a51*k1[i+1] + a52*k2[i+1] + a53*k3[i+1] + a54*k4[i+1];
            yTemp[i+1] = y0[i+1] + dy[i+1];            
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            dy[i+0] = a51*k1[i+0] + a52*k2[i+0] + a53*k3[i+0] + a54*k4[i+0];
            yTemp[i+0] = y0[i+0] + dy[i+0];

            dy[i+1] = a51*k1[i+1] + a52*k2[i+1] + a53*k3[i+1] + a54*k4[i+1];
            yTemp[i+1] = y0[i+1] + dy[i+1];

            dy[i+2] = a51*k1[i+2] + a52*k2[i+2] + a53*k3[i+2] + a54*k4[i+2];
            yTemp[i+2] = y0[i+2] + dy[i+2];            
        }
    }


    this->derivatives(x0 + dx, li, yTemp, dydx, k5, err, dfdx);

    {
        double c51invdx = c51*invdx;
        double c52invdx = c52*invdx;
        double c53invdx = c53*invdx;
        double c54invdx = c54*invdx;
        __m256d c51invdxv = _mm256_set1_pd(c51invdx);
        __m256d c52invdxv = _mm256_set1_pd(c52invdx);
        __m256d c53invdxv = _mm256_set1_pd(c53invdx);
        __m256d c54invdxv = _mm256_set1_pd(c54invdx);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dydxv = _mm256_loadu_pd(&dydx[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_loadu_pd(&k3[i]);
            __m256d k4v = _mm256_loadu_pd(&k4[i]);

            __m256d k5v = _mm256_fmadd_pd(c51invdxv,k1v,dydxv);
            k5v = _mm256_fmadd_pd(c52invdxv,k2v,k5v);
            k5v = _mm256_fmadd_pd(c53invdxv,k3v,k5v);
            k5v = _mm256_fmadd_pd(c54invdxv,k4v,k5v);
            _mm256_storeu_pd(&k5[i],k5v);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            k5[i] = dydx[i] + c51invdx*k1[i] + c52invdx*k2[i]+ c53invdx*k3[i] + c54invdx*k4[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            k5[i+0] = dydx[i+0] + c51invdx*k1[i+0] + c52invdx*k2[i+0]+ c53invdx*k3[i+0] + c54invdx*k4[i+0];
            k5[i+1] = dydx[i+1] + c51invdx*k1[i+1] + c52invdx*k2[i+1]+ c53invdx*k3[i+1] + c54invdx*k4[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            k5[i+0] = dydx[i+0] + c51invdx*k1[i+0] + c52invdx*k2[i+0]+ c53invdx*k3[i+0] + c54invdx*k4[i+0];
            k5[i+1] = dydx[i+1] + c51invdx*k1[i+1] + c52invdx*k2[i+1]+ c53invdx*k3[i+1] + c54invdx*k4[i+1];
            k5[i+2] = dydx[i+2] + c51invdx*k1[i+2] + c52invdx*k2[i+2]+ c53invdx*k3[i+2] + c54invdx*k4[i+2];            
        }
    }

    LU.xSolve(k5);

    {
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dyv = _mm256_loadu_pd(&dy[i]);
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d k5v = _mm256_loadu_pd(&k5[i]);

            dyv = _mm256_add_pd(k5v,dyv);
            __m256d yTempv = _mm256_add_pd(y0v,dyv);

            _mm256_storeu_pd(&dy[i],dyv);
            _mm256_storeu_pd(&yTemp[i],yTempv);            
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            dy[i] += k5[i];
            yTemp[i] = y0[i] + dy[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            dy[i+0] += k5[i+0];
            yTemp[i+0] = y0[i+0] + dy[i+0];
            dy[i+1] += k5[i+1];
            yTemp[i+1] = y0[i+1] + dy[i+1];                 
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            dy[i+0] += k5[i+0];
            yTemp[i+0] = y0[i+0] + dy[i+0];
            dy[i+1] += k5[i+1];
            yTemp[i+1] = y0[i+1] + dy[i+1];
            dy[i+2] += k5[i+2];
            yTemp[i+2] = y0[i+2] + dy[i+2];             
        }
    }

    this->derivatives(x0 + dx, li, yTemp, dydx,err, dfdx);
  

    {
        double c61invdx = c61*invdx;
        double c62invdx = c62*invdx;
        double c63invdx = c63*invdx;
        double c64invdx = c64*invdx;
        double c65invdx = c65*invdx;
        __m256d c61invdxv = _mm256_set1_pd(c61invdx);
        __m256d c62invdxv = _mm256_set1_pd(c62invdx);
        __m256d c63invdxv = _mm256_set1_pd(c63invdx);
        __m256d c64invdxv = _mm256_set1_pd(c64invdx);
        __m256d c65invdxv = _mm256_set1_pd(c65invdx);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d dydxv = _mm256_loadu_pd(&dydx[i]);
            __m256d k1v = _mm256_loadu_pd(&k1[i]);
            __m256d k2v = _mm256_loadu_pd(&k2[i]);
            __m256d k3v = _mm256_loadu_pd(&k3[i]);
            __m256d k4v = _mm256_loadu_pd(&k4[i]);
            __m256d k5v = _mm256_loadu_pd(&k5[i]);

            __m256d errv = _mm256_fmadd_pd(c61invdxv,k1v,dydxv);
            errv = _mm256_fmadd_pd(c62invdxv,k2v,errv);
            errv = _mm256_fmadd_pd(c63invdxv,k3v,errv);
            errv = _mm256_fmadd_pd(c64invdxv,k4v,errv);
            errv = _mm256_fmadd_pd(c65invdxv,k5v,errv);
            _mm256_storeu_pd(&err[i],errv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            err[i] = dydx[i] + c61invdx*k1[i] + c62invdx*k2[i] + c63invdx*k3[i] + c64invdx*k4[i]+ c65invdx*k5[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            err[i+0] = dydx[i+0] + c61invdx*k1[i+0] + c62invdx*k2[i+0] + c63invdx*k3[i+0] + c64invdx*k4[i+0]+ c65invdx*k5[i+0];
            err[i+1] = dydx[i+1] + c61invdx*k1[i+1] + c62invdx*k2[i+1] + c63invdx*k3[i+1] + c64invdx*k4[i+1]+ c65invdx*k5[i+1];
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            err[i+0] = dydx[i+0] + c61invdx*k1[i+0] + c62invdx*k2[i+0] + c63invdx*k3[i+0] + c64invdx*k4[i+0]+ c65invdx*k5[i+0];
            err[i+1] = dydx[i+1] + c61invdx*k1[i+1] + c62invdx*k2[i+1] + c63invdx*k3[i+1] + c64invdx*k4[i+1]+ c65invdx*k5[i+1];
            err[i+2] = dydx[i+2] + c61invdx*k1[i+2] + c62invdx*k2[i+2] + c63invdx*k3[i+2] + c64invdx*k4[i+2]+ c65invdx*k5[i+2];
        }
    }

    LU.xSolve(err);

    {
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d dyv = _mm256_loadu_pd(&dy[i]);
            __m256d errv = _mm256_loadu_pd(&err[i]);

            __m256d yTempv = _mm256_add_pd(_mm256_add_pd(y0v,dyv),errv);
            _mm256_storeu_pd(&yTemp[i],yTempv);
        }
        if(remain==1)
        {
            unsigned int i = n_-1;
            yTemp[i] = y0[i] + dy[i] + err[i];
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            yTemp[i+0] = y0[i+0] + dy[i+0] + err[i+0];
            yTemp[i+1] = y0[i+1] + dy[i+1] + err[i+1];

        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            yTemp[i+0] = y0[i+0] + dy[i+0] + err[i+0];
            yTemp[i+1] = y0[i+1] + dy[i+1] + err[i+1];
            yTemp[i+2] = y0[i+2] + dy[i+2] + err[i+2];
        }
    }

    double maxErr = 0;
    {
        __m256d maxErrv = _mm256_setzero_pd();
        __m256d absTolv = _mm256_set1_pd(absTol_);
        __m256d relTolv = _mm256_set1_pd(relTol_);
        for(unsigned int i = 0 ;i < n_-remain;i=i+4)
        {
            __m256d y0v = _mm256_loadu_pd(&y0[i]);
            __m256d yTempv = _mm256_loadu_pd(&yTemp[i]);
            __m256d errv = _mm256_loadu_pd(&err[i]);
            __m256d signMask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
            __m256d abserrv = _mm256_andnot_pd(signMask, errv);
            __m256d absy0v = _mm256_andnot_pd(signMask, y0v);
            __m256d absyTempv  = _mm256_andnot_pd(signMask, yTempv);

            __m256d maxY = _mm256_max_pd(absyTempv,absy0v);
            __m256d tolv = _mm256_fmadd_pd(relTolv,maxY,absTolv);

            __m256d errByTolv = _mm256_div_pd(abserrv,tolv);
            maxErrv = _mm256_max_pd(maxErrv,errByTolv);
        }

        __m128d low128  = _mm256_castpd256_pd128(maxErrv);           
        __m128d high128 = _mm256_extractf128_pd(maxErrv, 1);          
        __m128d max128  = _mm_max_pd(low128, high128);            

        __m128d shuffled = _mm_shuffle_pd(max128, max128, 0x1);   
        __m128d finalMax = _mm_max_sd(max128, shuffled);          

        maxErr = _mm_cvtsd_f64(finalMax);  
        if(remain==1)
        {
            unsigned int i = n_-1;
            double tol = absTol_ + relTol_*std::max(std::fabs(y0[i]), std::fabs(yTemp[i]));
            maxErr = std::max(maxErr, std::fabs(err[i])/tol);
        }
        else if(remain==2)
        {
            unsigned int i = n_-2;
            double tol = absTol_ + relTol_*std::max(std::fabs(y0[i+0]), std::fabs(yTemp[i+0]));
            maxErr = std::max(maxErr, std::fabs(err[i+0])/tol);
            tol = absTol_ + relTol_*std::max(std::fabs(y0[i+1]), std::fabs(yTemp[i+1]));
            maxErr = std::max(maxErr, std::fabs(err[i+1])/tol);
        }
        else if(remain==3)
        {
            unsigned int i = n_-3;
            double  tol = absTol_ + relTol_*std::max(std::fabs(y0[i+0]), std::fabs(yTemp[i+0]));
            maxErr = std::max(maxErr, std::fabs(err[i+0])/tol);
                    tol = absTol_ + relTol_*std::max(std::fabs(y0[i+1]), std::fabs(yTemp[i+1]));
            maxErr = std::max(maxErr, std::fabs(err[i+1])/tol);
                    tol = absTol_ + relTol_*std::max(std::fabs(y0[i+2]), std::fabs(yTemp[i+2]));
            maxErr = std::max(maxErr, std::fabs(err[i+2])/tol);            
        }        
    }
    return maxErr;

}



template<class ChemistryModel>
Foam::scalar Foam::OptRodas34<ChemistryModel>::normaliseError
(
    const double* __restrict__ y0,
    const double* __restrict__ y,
    const double* __restrict__ err
) const
{

    scalar maxErr = 0.0;
    for (label i=0; i<n_; i++)
    {
        scalar tol = absTol_ + relTol_*std::max(std::fabs(y0[i]), std::fabs(y[i]));
        maxErr = std::max(maxErr, std::fabs(err[i])/tol);
    }
    return maxErr;
}

// ************************************************************************* //
