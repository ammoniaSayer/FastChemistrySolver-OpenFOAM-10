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
    Foam::simpleDataBlock

Description
    A data block to store the chemistry variable of a cell. 
    it doesn't stores time step and RR vector

SourceFiles
    simpleDataBlock.C

\*---------------------------------------------------------------------------*/

#ifndef simpleDataBlock_H
#define simpleDataBlock_H

#include "fvCFD.H"
#include "Ostream.H"
#include "Istream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

class simpleDataBlock;

bool operator==(const simpleDataBlock&, const simpleDataBlock&);
bool operator!=(const simpleDataBlock&, const simpleDataBlock&);

Istream& operator >> (Istream& is, simpleDataBlock& data);
Ostream& operator << (Ostream& os, const simpleDataBlock& data);

/*---------------------------------------------------------------------------*\
                       Class simpleDataBlock Declaration
\*---------------------------------------------------------------------------*/

class simpleDataBlock
{

public:

    // Public Data
        
        //- a list of species mass fraction for chemistry reaction
        List<scalar> Y_;

        //- temperature [K]
        scalar T;

        //- absolute pressure [Pa]
        scalar p;

        //- CPU time for chemistry integration
        int64_t CPUtime;

        //- cell id
        label celli;

        //- the chemical integration step in last computation
        scalar deltaTChem_;

public:

    // Constructors

        //- Default constructor
        simpleDataBlock() = default;

        //- Construct from total number of species
        simpleDataBlock(label& nSpecies_);

        //- Construct from Istream
        simpleDataBlock(Istream& is);


    // Friend Operators

        inline friend bool operator==(const simpleDataBlock&, const simpleDataBlock&);
        inline friend bool operator!=(const simpleDataBlock&, const simpleDataBlock&);


    // IOstream operator

        friend Istream& operator>>(Istream&, simpleDataBlock&);
        friend Ostream& operator<<(Ostream&, const simpleDataBlock&);

};

//- inline operator
inline bool operator==(const simpleDataBlock& data1, const simpleDataBlock& data2)
{
    return
        data1.Y_ == data2.Y_ &&
        data1.T == data2.T &&
        data1.p == data2.p &&
        data1.CPUtime == data2.CPUtime &&
        data1.celli == data2.celli &&
        data1.deltaTChem_ == data2.deltaTChem_;
}

inline bool operator!=(const simpleDataBlock& data1, const simpleDataBlock& data2)
{
    return !(data1==data2);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
