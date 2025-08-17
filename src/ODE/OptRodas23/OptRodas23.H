/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2019 OpenFOAM Foundation
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

Class
    Foam::Rodas23

Description
    L-stable, stiffly-accurate embedded Rosenbrock ODE solver of order (3)4.

    Reference:
    \verbatim
        Hairer, E., Nørsett, S. P., & Wanner, G. (1996).
        Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems, second edition",
        Springer-Verlag, Berlin.
    \endverbatim

SourceFiles
    Rodas23.C

\*---------------------------------------------------------------------------*/

#ifndef OptRodas23_H
#define OptRodas23_H

#include "chemistrySolver.H"
#include "simpleMatrix.H"
#include <iostream>
#include <vector>
#include <algorithm>

#include "LUsolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class OptRodas23 Declaration
\*---------------------------------------------------------------------------*/
template<class ChemistryModel>
class OptRodas23
:
    public chemistrySolver<ChemistryModel>
{
    // Private Data
    class stepState
    {
        public:


        scalar dxTry;
        bool last;

        stepState(const scalar dx)
        :
            dxTry(dx),
            last(false)
        {}
    };

        //- Coefficients dictionary
        dictionary coeffsDict_;

    // ODE solver

        //- Absolute convergence tolerance per step
        const scalar absTol_;

        //- Relative convergence tolerance per step
        const scalar relTol_;

        //- The maximum number of sub-steps allowed for the integration step
        const label maxSteps_;

        //- Size of the ODE system
        const unsigned int n_;

    // Adaptive Solver

        //- Step-size adjustment controls
        constexpr static const scalar safeScale_ =     0.9;
        constexpr static const scalar alphaInc_  =     0.2;
        constexpr static const scalar alphaDec_ =     0.25;
        constexpr static const scalar minScale_ =      0.2;
        constexpr static const scalar maxScale_ =      10;

        constexpr static const scalar ratio = std::pow(maxScale_/safeScale_, -1.0/alphaInc_);
        constexpr static const scalar safeMaxScale = safeScale_*maxScale_;

    // Rodas23
        constexpr static const scalar c3 = 1;
        constexpr static const scalar d1 = 1.0/2.0;
        constexpr static const scalar d2 = 3.0/2.0;
        constexpr static const scalar a31 = 2;
        constexpr static const scalar a41 = 2;
        constexpr static const scalar c21 = 4;
        constexpr static const scalar c31 = 1;
        constexpr static const scalar c32 = -1;
        constexpr static const scalar c41 = 1;
        constexpr static const scalar c42 = -1;
        constexpr static const scalar c43 = -8.0/3.0;
        constexpr static const scalar gamma = 1.0/2.0;
        constexpr static const scalar Invgamma= 1.0/gamma;

    // Adaptive Solver

        //- Field encapsulating the composition, temperature and pressure
        mutable scalarField cTp_;

    // Rodas23

        mutable labelList pivotIndices_;


        mutable LUsolver LU;

public:

    //- Runtime type information
    TypeName("OptRodas23");


    // Constructors

        //- Construct from thermo
        OptRodas23(const fluidReactionThermo& thermo);


    //- Destructor
    virtual ~OptRodas23();


    // Member Functions

        //- Update the concentrations and return the chemical time

        virtual void solve
        (
            scalar& __restrict__ p,
            scalar& __restrict__ T,
            scalarField& c,
            const label li,
            scalar& __restrict__ deltaT,
            scalar& __restrict__ subDeltaT
        ) const;

    // ODE solver

        void ODESolve
        (
            const scalar xStart,
            const scalar xEnd,
            const label li,
            scalar& dxTry,
            double* __restrict__ y
        )const;

    // adaptive solver

        void adaptiveSolve
        (
            scalar& __restrict__ x,
            const label li,
            scalar& __restrict__ dxTry,
            double* __restrict__ y
        )const;

        scalar Rodas23Solve
        (
            const scalar x0,
            const label li,
            const scalar dx,
            const scalar invdx,
            double* __restrict__ y0,
            double* __restrict__ y
        )const;

        //- Return the normalised scalar error
        scalar normaliseError
        (
            const double* __restrict__ y0,
            const double* __restrict__ y,
            const double* __restrict__ err
        )const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#ifdef NoRepository
    #include "OptRodas23.C"
#endif

#endif

// ************************************************************************* //
