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

Class
    Foam::FastChemistryModel

Description

SourceFiles
    FastChemistryModelI.H
    FastChemistryModel.C

\*---------------------------------------------------------------------------*/

#ifndef FastChemistryModel_H
#define FastChemistryModel_H


#include "basicChemistryModel.H"
#include "ReactionList.H"
#include "ODESystem.H"
#include "volFields.H"
#include "multiComponentMixture.H"
#include "DynamicField.H"
#include <vector>
#include "dataBlock.H"
#include "simpleDataBlock.H"
#include "OptReaction.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class FastChemistryModel Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class FastChemistryModel
:
    public basicChemistryModel
{
    // Private classes

        //- Class to define scope of reaction evaluation. Runs pre-evaluate
        //  hook on all reactions on construction and post-evaluate on
        //  destruction.
        class reactionEvaluationScope
        {
            const FastChemistryModel<ThermoType>& chemistry_;

        public:

            reactionEvaluationScope
            (
                const FastChemistryModel<ThermoType>& chemistry
            )
            :
                chemistry_(chemistry)
            {
                /*forAll(chemistry_.reactions_, i)
                {
                    chemistry_.reactions_[i].preEvaluate();
                }*/
            }

            ~reactionEvaluationScope()
            {
                /*forAll(chemistry_.reactions_, i)
                {
                    chemistry_.reactions_[i].postEvaluate();
                }*/
            }
        };



    // Private data


        //- Reference to the field of specie mass fractions
        const PtrList<volScalarField>& Yvf_;

        //- Number of species
        label nSpecie_;

        //- Index of default species
        label defaultIndex;

        //- Size of the ODE system
        label n_;

        //- Type of the Jacobian to be calculated
        const jacobianType jacobianType_;

        //- Reference to the multi component mixture
        const multiComponentMixture<ThermoType>& mixture_;

        //- Thermodynamic data of the species
        const PtrList<ThermoType>& specieThermos_;

        //- Reactions
        //const ReactionList<ThermoType> reactions_;

        //- Reactions
        mutable OptReaction    reaction;

        //- List of reaction rate per specie [kg/m^3/s]
        PtrList<volScalarField> RR_;

        //- Temporary mass fraction field
        mutable scalarField Y_;

        //- Temporary simplified mechanism mass fraction field
        //DynamicField<scalar> sY_;

        //- Temporary concentration field
        mutable scalarField c_;


public:
        //- Data buffer
        mutable double*     buffer;             
        mutable double*     YTpWork[12];            
        mutable double*     YTpYTpWork[3];             

private:

        //- Minimum reaction temperature
        const scalar Treact;

            //****************************MPI Communication*****************************//

            //- CPU time larger tave*DLBthreshold is busy process, and will send data to other process.
            scalar DLBthreshold = 1.0;

            //- Updte load balancing per Iter
            label MaxIter = 1;
            label CurrentIter = 0;

            //- CPU time for data transfer
            List<List<int64_t>> cpuLoadTransferTable;

            //- CPU time to solve chemistry for each cell
            std::vector<std::pair<int64_t,label>> CPUtimeField;

            //- CPU time so solve chemistry for each process
            List<int64_t> chemistryIntegrationTime;

            //- Size of send buffer
            List<std::streamsize> sendBufferSize_;

            //- Size of receive buffer
            List<std::streamsize> recvBufferSize_;

            //- Send buffer, containing the binary data block
            List<DynamicList<char>> sendBuffer_;

            //- Receive buffer, containing the binary data block
            List<DynamicList<char>> recvBuffer_;

            //- The receive position of receive buffer
            List<label> recvBufPos_;

            //- firt time to be executed
            bool firstTime = true;

            //- Skip this cell or solve?
            List<bool> skip;

            //- Determine whether this process is busy
            List<bool> IamBusyProcess;

            //- Whether to switch on the load balancing
            bool Balance;

            void exchange
            (
                const UList<DynamicList<char>>& sendBufs,
                const List<std::streamsize>& recvSizes,
                List<DynamicList<char>>& recvBufs,
                const int tag,
                const label comm,
                const bool block
            );

            void exchangeSizes
            (
                const UList<DynamicList<char>>& sendBufs,
                labelList& recvSizes,
                const label comm
            );
            //****************************MPI Communication*****************************//

    // Private Member Functions

        //- Write access to chemical source terms
        //  (e.g. for multi-chemistry model)
        inline PtrList<volScalarField>& RR();

        //- Solve the reaction system for the given time step
        //  of given type and return the characteristic time
        //  Variable number of species added
        template<class DeltaTType>
        scalar solve(const DeltaTType& deltaT){return 0;}

public:

    //- Runtime type information
    TypeName("FastChemistryModel");


    // Constructors

        //- Construct from thermo
        FastChemistryModel(const fluidReactionThermo& thermo);

        //- Disallow default bitwise copy construction
        FastChemistryModel(const FastChemistryModel&) = delete;


    //- Destructor
    virtual ~FastChemistryModel();


    // Member Functions

        //- Return reference to the mixture
        inline const multiComponentMixture<ThermoType>& mixture() const;

        //- The reactions
        inline const PtrList<Reaction<ThermoType>>& reactions() const;

        //- Thermodynamic data of the species
        inline const PtrList<ThermoType>& specieThermos() const;

        //- The number of reactions
        virtual inline label nReaction() const;

        //- Calculates the reaction rates
        virtual void calculate();


        // Chemistry model functions (overriding abstract functions in
        // basicChemistryModel.H)

            //- The number of species
            inline virtual label nSpecie() const;


            //- Return const access to the chemical source terms for specie, i
            inline const volScalarField::Internal& RR
            (
                const label i
            ) const;

            //- Return non const access to chemical source terms [kg/m^3/s]
            virtual volScalarField::Internal& RR
            (
                const label i
            );

            //- Return reaction rate of the speciei in reactionI
            virtual tmp<volScalarField::Internal> calculateRR
            (
                const label reactionI,
                const label speciei
            ) const;

            //- Solve the reaction system for the given time step
            //  and return the characteristic time
            virtual scalar solve(const scalar deltaT) ;

            //- Solve the reaction system for the given time step
            //  and return the characteristic time
            virtual scalar solve(const scalarField& deltaT) ;

            //- Return the chemical time scale
            virtual tmp<volScalarField> tc() const;

            //- Return the heat release rate [kg/m/s^3]
            virtual tmp<volScalarField> Qdot() const;


        // ODE functions (overriding abstract functions in ODE.H)
            virtual void derivatives
            (
                const scalar t,
                const scalarField& YTp,
                const label li,
                scalarField& dYTpdt
            ) const;

            virtual void derivatives
            (
                const scalar t,
                const label li,
                double* __restrict__ YTp,
                double* __restrict__ dYTpdt,
                double* __restrict__ Cp,
                double* __restrict__ Ha,
                double* __restrict__ ExpGbyRT
            ) const ;

            virtual void derivativesC
            (
                const scalar t,
                const scalarField& __restrict__ YTp,
                const label li,
                scalarField& __restrict__ dCTpdt,
                scalarField& __restrict__ Cp,
                scalarField& __restrict__ Ha,
                scalarField& __restrict__ ExpGbyRT
            ) const ;            
        
            virtual void derivatives
            (
                const scalar t,
                const label li,
                double* __restrict__ YTp,                
                double* __restrict__ dYTpdt,
                double* __restrict__ Cp,
                double* __restrict__ Ha
            ) const ;

            virtual void derivativesC
            (
                const scalar t,
                const scalarField& __restrict__ YTp,
                const label li,
                scalarField& __restrict__ dCTpdt,
                scalarField& __restrict__ Cp,
                scalarField& __restrict__ Ha
            ) const ;

            virtual void jacobian
            (
                const scalar t,
                double* __restrict__ YTp,
                const label li,
                double* __restrict__ dYTpdt,
                double* __restrict__ Jy
            ) const ;

            virtual void jacobian
            (
                const scalar t,
                const scalarField& __restrict__ YTp,
                const label li,
                scalarField& __restrict__ dYTpdt,
                scalarSquareMatrix& J,
                int Placeholder
            ) const ;

            virtual void solve
            (
                scalar& p,
                scalar& T,
                scalarField& Y,
                const label li,
                scalar& deltaT,
                scalar& subDeltaT
            ) const = 0;

        // Mechanism reduction access functions

            //- Return true if specie i is active
            inline bool active(const label i) const;

            //- Set specie i active
            inline void setActive(const label i);

            inline OptReaction&  getReaction() const{return reaction;}

    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const FastChemistryModel&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "FastChemistryModelI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "FastChemistryModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
