/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2020 OpenFOAM Foundation
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

#include "OptRosenbrock23.H"
#include "SubField.H"
#include "addToRunTimeSelectionTable.H"

#include <immintrin.h>  
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ChemistryModel>
Foam::OptRosenbrock23<ChemistryModel>::OptRosenbrock23
(
    const fluidReactionThermo& thermo
)
:
    chemistrySolver<ChemistryModel>(thermo),
    coeffsDict_(this->subDict("OptRosenbrock23Coeffs")),
    absTol_(coeffsDict_.lookup<scalar>("absTol")),
    relTol_(coeffsDict_.lookup<scalar>("relTol")),
    maxSteps_(coeffsDict_.lookupOrDefault("maxSteps",10000)),
    n_(this->nSpecie()+1),
    cTp_(n_),
    pivotIndices_(n_),
    LU(this->YTpYTpWork[1],n_)
{
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ChemistryModel>
Foam::OptRosenbrock23<ChemistryModel>::~OptRosenbrock23()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
template<class ChemistryModel>
void Foam::OptRosenbrock23<ChemistryModel>::solve
(
    scalar& __restrict__ p,
    scalar& __restrict__ T,
    scalarField& y,
    const label li,
    scalar& __restrict__ deltaT,
    scalar& __restrict__ subDeltaT
) const
{

    double* __restrict__ y00    = this->YTpWork[0];
    double* __restrict__ y0     = this->YTpWork[1];

    // Map the composition, temperature and pressure into cTp
    for (int i=0; i<this->nSpecie(); i++)
    {
        y00[i] = max(0, y[i]);
    }
    y00[this->nSpecie()] = T;

    for (int i=0; i<n_; i++)
    {
        y0[i] = y00[i];
    }


    this->ODESolve
    (
        0,                  //start
        deltaT,             //end(time step) 
        li,                 //celli
        subDeltaT,          //sub time step
        y0                  //initial vector
    );


    for (int i=0; i<this->nSpecie(); i++)
    {
        y[i] = max(0.0, y0[i]);
    }
    T = y0[this->nSpecie()];
}

template<class ChemistryModel>
void Foam::OptRosenbrock23<ChemistryModel>::ODESolve
(
    const scalar xStart,
    const scalar xEnd,
    const label li,
    scalar& dxTry,
    double* __restrict__ y0
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
            x,              //xstart
            li,             //celli
            step.dxTry,     //sub time step
            y0              //initial vector
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
void Foam::OptRosenbrock23<ChemistryModel>::adaptiveSolve
(
    scalar& __restrict__ x,
    const label li,
    scalar& __restrict__ dxTry,
    double* __restrict__ y0
) const
{
    //scalarField& y0 = YTpWork_[1];
    //scalarField& y1 = YTpWork_[2];
    double* __restrict__ yTemp = this->YTpWork[2];

    scalar dx = dxTry;
    scalar err = 0.0;
    scalar invdx = 1.0/dx;

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error

    err = Rosenbrock23Solve(x, li, dx,invdx, y0,yTemp);
    while(err > 1)
    {

        scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
        dx *= scale;
        invdx = 1.0/dx;

        err = Rosenbrock23Solve(x, li, dx,invdx, y0,yTemp);
    } 


    // Update the state
    x += dx;
    for(label i = 0; i < n_;i++)
    {
        y0[i] = yTemp[i];
    }

    dxTry = (err>ratio)?
            min(max(safeScale_*pow(err, -alphaInc_), minScale_), maxScale_)*dx
            :safeMaxScale*dx;

}

template<class ChemistryModel>
Foam::scalar Foam::OptRosenbrock23<ChemistryModel>::Rosenbrock23Solve
(
    const scalar x0,
    const label li,
    const scalar dx,
    const scalar invdx,
    double* __restrict__ y0,
    double* __restrict__ yTemp
) const
{

    double* __restrict__ k1 = this->YTpWork[3];
    double* __restrict__ k2 = this->YTpWork[4];
    double* __restrict__ k3 = this->YTpWork[5];
    double* __restrict__ err = this->YTpWork[7];
    double* __restrict__ dydx = this->YTpWork[8];
    double* __restrict__ dfdx = this->YTpWork[9];
    double* __restrict__ a_ = this->YTpYTpWork[1];

    //odes_.jacobian(x0, y0, li, dfdx_, dfdy_);
    this->jacobian(x0, y0, li, dfdx, a_);

    {
        const unsigned int NN = n_*n_;
        unsigned int  remain = NN%16;
        for(unsigned int  i = 0 ; i<NN-remain;i=i+16)
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
        for(unsigned int  i = NN-remain; i < NN;i++)
        {
            a_[i] = -a_[i];
        }
    }
    for (label  i=0; i<n_; i++)
    {
        a_[i*n_+i] += invdx*Invgamma;
    }

    
    LU.Block4LUDecompose();
    

    for(label i = 0 ;i < n_;i++)
    {
        k1[i] = dfdx[i] + dx*d1*dfdx[i];
    }

    LU.xSolve(k1);
    

    // Calculate k2:
    for(label i = 0 ;i < n_;i++)
    {
        yTemp[i] = y0[i] + a21*k1[i];
    }

    this->derivatives(x0 + c2*dx, li, yTemp, dydx, k2, k3, err);

    for(label i = 0 ;i < n_;i++)
    {
        k2[i] = dydx[i] + dx*d2*dfdx[i] + c21*k1[i]/dx;
    }

    
    LU.xSolve(k2);
    
    // Calculate k3:
    for(label i = 0 ;i < n_;i++)
    {
        k3[i] = dydx[i] + dx*d3*dfdx[i] + (c31*k1[i] + c32*k2[i])/dx;
    }

    LU.xSolve(k3);
    
    // Calculate error and update state
    for(label i = 0 ;i < n_;i++)
    {
        yTemp[i] = y0[i] + b1*k1[i] + b2*k2[i] + b3*k3[i];
        err[i] = e1*k1[i] + e2*k2[i] + e3*k3[i];
    }

    return normaliseError(y0, yTemp, err);
}


template<class ChemistryModel>
Foam::scalar Foam::OptRosenbrock23<ChemistryModel>::normaliseError
(
    const double* __restrict__ y0,
    const double* __restrict__ y,
    const double* __restrict__ err
) const
{
    // Calculate the maximum error
    double maxErr = 0.0;

    for(label i = 0 ;i < n_;i++)
    {
        double tol = absTol_ + relTol_*std::max(std::fabs(y0[i]), std::fabs(y[i]));
        maxErr = std::max(maxErr, std::fabs(err[i])/tol);
    }
    return maxErr;
}


// ************************************************************************* //
