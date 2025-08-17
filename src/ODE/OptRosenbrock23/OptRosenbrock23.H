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

Class
    Foam::OptRosenbrock23

Description
    L-stable embedded Rosenbrock ODE solver of order (2)3.

    References:
    \verbatim
        Sandu, A., Verwer, J. G., Blom, J. G., Spee, E. J., Carmichael, G. R.,
        & Potra, F. A. (1997).
        Benchmarking stiff ODE solvers for atmospheric chemistry problems II:
        Rosenbrock solvers.
        Atmospheric environment, 31(20), 3459-3472.
    \endverbatim

    
SourceFiles
    OptRosenbrock23.C

\*---------------------------------------------------------------------------*/

#ifndef OptRosenbrock23_H
#define OptRosenbrock23_H

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
                        Class OptRosenbrock23 Declaration
\*---------------------------------------------------------------------------*/

template<class ChemistryModel>
class OptRosenbrock23
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
        const label n_;

    // Adaptive Solver

        //- Step-size adjustment controls
        constexpr static const scalar safeScale_ =     0.9;
        constexpr static const scalar alphaInc_  =     0.2;
        constexpr static const scalar alphaDec_ =     0.25;
        constexpr static const scalar minScale_ =      0.2;
        constexpr static const scalar maxScale_ =      10;

        constexpr static const scalar ratio = std::pow(maxScale_/safeScale_, -1.0/alphaInc_);
        constexpr static const scalar safeMaxScale = safeScale_*maxScale_;

    // Rosenbrock

        constexpr static const scalar a21 = 1;
        constexpr static const scalar a31 = 1;
        constexpr static const scalar a32 = 0;

        constexpr static const scalar c21 = -1.0156171083877702091975600115545;
        constexpr static const scalar c31 = 4.0759956452537699824805835358067;
        constexpr static const scalar c32 = 9.2076794298330791242156818474003;

        constexpr static const scalar b1 = 1;
        constexpr static const scalar b2 = 6.1697947043828245592553615689730;
        constexpr static const scalar b3 = -0.4277225654321857332623837380651;

        constexpr static const scalar e1 = 0.5;
        constexpr static const scalar e2 = -2.9079558716805469821718236208017;
        constexpr static const scalar e3 = 0.2235406989781156962736090927619;

        constexpr static const scalar gamma = 0.43586652150845899941601945119356;
        constexpr static const scalar Invgamma = 1.0/gamma;
        constexpr static const scalar c2 = 0.43586652150845899941601945119356;

        constexpr static const scalar d1 = 0.43586652150845899941601945119356;
        constexpr static const scalar d2 = 0.24291996454816804366592249683314;
        constexpr static const scalar d3 = 2.1851380027664058511513169485832;



    // Adaptive Solver

        //- Field encapsulating the composition, temperature and pressure
        mutable scalarField cTp_;

        mutable labelList pivotIndices_;    //-  pivotIndices for LUDecompose

        mutable LUsolver LU;


public:

    //- Runtime type information
    TypeName("OptRosenbrock23");


    // Constructors

        //- Construct from thermo
        OptRosenbrock23(const fluidReactionThermo& thermo);


    //- Destructor
    virtual ~OptRosenbrock23();


    // Member Functions

        //- Update the concentrations and return the chemical time
        // 
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

        scalar Rosenbrock23Solve
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
    #include "OptRosenbrock23.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
