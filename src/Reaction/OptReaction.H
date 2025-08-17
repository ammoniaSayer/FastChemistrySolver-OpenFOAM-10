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
    self Reaction

Description

SourceFiles
    Reaction.C

\*---------------------------------------------------------------------------*/

#ifndef OptReaction_H
#define OptReaction_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <math.h>
#include <unordered_map>
#include <iomanip>

#include "PtrList.H"
#include "scalarField.H"
#include "fvCFD.H"
#include "HashTable.H"
#include "Macro.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #if USE_LOCALFILE_

        extern "C" __m256d _ZGVdN4v_gnuexp_avx2(__m256d x);
        extern "C" __m256d _ZGVdN4vv_gnupow_avx2(__m256d x, __m256d y);
        extern "C" __m256d _ZGVdN4v_gnulog_avx2(__m256d x);

        inline __m256d vec256_expd(__m256d x) 
        {
            return _ZGVdN4v_gnuexp_avx2(x);
        }
        inline __m256d vec256_powd(__m256d x, __m256d y) 
        {
            return _ZGVdN4vv_gnupow_avx2(x,y);
        }
        inline __m256d vec256_logd(__m256d x) 
        {
            return _ZGVdN4v_gnulog_avx2(x);
        }

    #else

        extern __m256d vec256_logd(__m256d) asm("_ZGVdN4v_log");
        extern __m256d vec256_expd(__m256d) asm("_ZGVdN4v_exp");
        extern __m256d vec256_powd(__m256d, __m256d) asm("_ZGVdN4vv_pow");
        //extern __m512d vec512_expd(__m512d) asm("_ZGVdN8v_exp");
    #endif


/*---------------------------------------------------------------------------*\
                        Class Reaction Declaration
\*---------------------------------------------------------------------------*/


class OptReaction
{

public:

    // Constructors

        // construct from using OpenFOAM
        OptReaction(const dictionary& chemistryDict,const dictionary& physicalDict);

        // default constructor
        OptReaction(bool includePressure);

        // disable copy construction
        OptReaction(const OptReaction& reactions) = delete;

    // Destructor

        ~OptReaction();

    // Member function

        // Read information from OpenFOAM
        void readInfo
        (
            const dictionary& chemistryDict,
            const dictionary& physicalDict  
        );

        // Read reaction information from OpenFOAM
        void readReactionInfo
        (
            std::vector<unsigned int>& lhsIndex,
            std::vector<unsigned int>& lhsstoichCoeff,
            std::vector<unsigned int>& rhsIndex,
            std::vector<unsigned int>& rhsstoichCoeff,
            const dictionary& nthreaction,
            const hashedWordList& speciesTable
        );

        // Find the global reaction
        void findGlobalReaction
        (
            bool& isGlobal,
            const dictionary& nthreaction,
            const hashedWordList& speciesTable
        );

        // Find the two-two reaction
        bool findTwoTwoReaction
        (
            const dictionary& nthreaction,
            const hashedWordList& speciesTable
        );

    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OptReaction&) = delete;


    // Public data

        // Accelerating options

            // Whether the pressure range is exactly the same for
            // all Plog reaction
            bool uniformPressureRange = false;

        // Save the basic information of reactions and species.

            // Contain the species name, e.g. "CO", "H2O".
            mutable std::vector<std::string> speciesTable_;

            // Contain the reaction type name, e.g. "irreversibleArrhenius", "reversibleArrhenius".
            mutable std::vector<std::string> reactionType_;

            // Contain the reaction name, e.g. "un-named-reaction-1", "un-named-reaction-2".
            mutable std::vector<std::string> reactionName_;

            // Contain the chemical reaction equation, e.g. "O + H2 = H + OH","O + H2O2 = OH + HO2".
            mutable std::vector<std::string> reactionTable_;  
            
            // Contain the global chemical reaction equation, e.g. "OC12H23OOH = 3C2H4 + C2H5 + 2CH2CHO + OH".            
            mutable std::vector<std::string> GlobalReactionTable_;

        // Data to store the numbers of reactions and species.

            // The number of all reactions.
            mutable unsigned int nReactions=0;

            // The number of chemical species, including the default species.
            mutable unsigned int nSpecies=0;

            // The number of irreversible reactions.
            mutable unsigned int n_Irreversible_Reaction=0;

            // The number of reversible reactions.
            mutable unsigned int n_Reversible_Reaction=0;

            // The number of reversible reactions.
            mutable unsigned int n_Arrhenius=0;      

            mutable unsigned int n_NonEquilibriumReversibleArrhenius=0;       

            mutable unsigned int n_NonEquilibriumThirdBodyReaction=0;

            // The number of third body reactions.
            mutable unsigned int n_ThirdBodyReaction=0;     

            // The number of fall-off reactions.
            mutable unsigned int n_Fall_Off_Reaction=0;

            // The number of chemically activated reactions.
            mutable unsigned int n_ChemicallyActivated_Reaction=0;

            // The number of Plog reaction;
            mutable unsigned int n_PlogReaction=0;

            // The number of global reactions. The reaction involves more than 3
            // reactants or more than 3 products
            mutable unsigned int n_Global_Reaction=0;

            // The number of two two reactions. The reaction involves more than 3            
            mutable unsigned int nTwoTwo = 0;

            // The number of temperature independent reactions in the type of
            // irreversible reaction and reversible reaction, it doesn't include 
            // the third body reaction and pressure reaction
            mutable unsigned int n_Temperature_Independent_Reaction=0;

            // Whether this reaction is reversible
            mutable std::vector<char> isIrreversible;

            // Whether this reaction involves same species
            mutable std::vector<char> sameSpecies;            

            // offset of k inf
            mutable unsigned int offset_kinf=0;

            // the size of ODE systems
            mutable unsigned int n_=0;

            mutable std::array<unsigned int,6> Itbr;

            mutable std::array<unsigned int,12> Ikf;

        // Data to save the index of falloff and chemicallyactivated reactions.

            // Store the index of the Lindemann form.
            mutable std::vector<unsigned int> Lindemann;

            // Store the index of the Troe form.        
            mutable std::vector<unsigned int> Troe;

            // Store the index of the SRI form.        
            mutable std::vector<unsigned int> SRI;

        // The data to save the information of chemical reaction equations.

            // Data to save the index of reactants for all equations.
            mutable std::vector<std::vector<unsigned int>>     lhsIndex;

            // Data to save the reaction order of reactants for all equations.
            mutable std::vector<std::vector<unsigned int>>  lhsstoichCoeff;

            // Data to save the index of products for all equations.
            mutable std::vector<std::vector<unsigned int>>     rhsIndex;

            // Data to save the reaction order of products for all equations.
            mutable std::vector<std::vector<unsigned int>>  rhsstoichCoeff;

            // 1D Data to save the index of reactants for all equations.
            mutable std::vector<unsigned int> lhsSpeciesIndex;

            mutable std::vector<unsigned int> lhsOffset;

            mutable std::vector<unsigned int> rhsSpeciesIndex;

            mutable std::vector<unsigned int> rhsOffset;           

        // Store the Arrhenius parameters for all reactions. For pressure depedent reaction,
        // these contains the parameters of low pressure limits and high pressure limits.

            // Arrhenius pre-exp factor.
            mutable std::vector<double>  A;

            // Arrhenius temperature corrector.        
            mutable std::vector<double>  beta;

            // Arrhenius activation temperature.        
            mutable std::vector<double>  Ta;
 
        // The data to save the temporary results

            // Data to save the Arrhenius forward rate constants for all reactions.
            mutable std::vector<double> Kf_;

            // Data to save the derivatives of forward rate constants w.r.t 
            // temperature for all reactions.
            mutable std::vector<double> dKfdT_;

            // Data to save the additional Exp computation, including 
            // species and pressure reaction
            mutable std::vector<double> tmp_Exp;

            // Data to save the derivatives of forward rate constants w.r.t 
            // species concentration for third body reactions and pressure 
            // dependent reactions.
            mutable std::vector<double> dKfdC_;    

            // Data to save the third body efficiency.
            mutable std::vector<double> tmp_M;

        // Store the third dody efficiency for third body reactions and 
        // falloff and chemicallyactivated reactions

            mutable std::vector<std::vector<double>> ThirdBodyFactor; 

            mutable std::vector<double> ThirdBodyFactor1D; 

        // Store the parameters of Troe form for falloff and chemicallyactivated reaction. 

            mutable std::vector<double> alpha_;

            mutable std::vector<double> Ts_;

            mutable std::vector<double> invTs_;

            mutable std::vector<double> Tss_;

            mutable std::vector<double> Tsss_;

            mutable std::vector<double> invTsss_;

        // Store the parameters of SRI form for falloff and chemicallyactivated reaction. 

            mutable std::vector<double> a_;

            mutable std::vector<double> b_;

            mutable std::vector<double> c_;

            mutable std::vector<double> invc_;

            mutable std::vector<double> d_;

            mutable std::vector<double> e_;

        // General pressure dependence using logarithmic interpolation

            mutable std::vector<double> Kf_Plog;

            mutable std::vector<double> dKfdT_Plog;

            mutable std::vector<std::vector<double>> APlog;

            mutable std::vector<std::vector<double>> logAPlog;

            mutable std::vector<std::vector<double>> betaPlog;

            mutable std::vector<std::vector<double>> TaPlog;

            mutable std::vector<std::vector<double>> Prange; 
            
            mutable std::vector<std::vector<double>> rDeltaP_;

            mutable std::vector<std::vector<double>> logPi;

            mutable std::vector<int> Pindex;

        // Data to store the parameters of Janf polynomial Fits.

            // Store the parameters of polynomial Fits at high temperature range
            // for all species.
            mutable std::vector<std::array<double,7>> HCoeffs;

            // Store the parameters of polynomial Fits at low temperature range
            // for all species.            
            mutable std::vector<std::array<double,7>> LCoeffs;

            // Store the low temperature limits of all species.
            mutable std::vector<double> Tlow;

            // Store the high temperature limits of all species.
            mutable std::vector<double> Thigh;

            // Store the common temperature limits of all species.           
            mutable std::vector<double> Tcommon;

            // Store the minimum value of Tcommon of all species.
            mutable double TcommonMin;

            // Store the maximum value of Tcommon of all species.
            mutable double TcommonMax;

            mutable double TlowMin;
            
            mutable double ThighMax;

            // Pointer to point the coefficients of all species.
            mutable std::vector<std::array<double,7>*> PtrCoeffs;

        // Thermodynamic parameter

            // Moleweight of species. [kg/kmol]
            mutable std::vector<double> W;

            // The reciprocal of moleweight of species.
            mutable std::vector<double> invW;

            // universal gas constant. [J/(kmol·K)]
            //const static constexpr double Ru = 8314.462618; 
            const static constexpr double Ru = 8314.470066505449722171761095523834228515625; 
            // inverse of universal gas constant. [J/(kmol·K)]
            const static constexpr double invRu = 1.0/Ru;

            // standard pressure. [Pa]
            const static constexpr double Pstd = 100000;
            //const static constexpr double Pstd = 101325;
        // Chemical equilibrium calculation

            // Table to save the temporary result for chemical equilibrium calculation.
            mutable std::unordered_map<int,double> Pow_pByRT_SumVki_;

            // Table to save the temporary result for chemical equilibrium calculation.
            // 5 element version of (Pstd/(Ru*T))^sumVki;
            mutable std::array<double,5> Pow_pByRT_SumVki;

        // Thermophysical variable

            // log(T).
            mutable double logT;

            // T
            mutable double T;

            // 1/T.
            mutable double invT;

            // T*T.
            mutable double sqrT;

            // mixture density
            mutable double rhoM;
      
        // Array
            //mutable std::vector<std::array<double,4>>Arr;

            mutable std::vector<double*> ArrPtr;

            mutable double* buffer; 

    // Member function

        // Thermodynamic function

            // Pointing the janaf coefficient using the given temperature.
            inline void setPtrCoeffs (const double T) const noexcept;

            // Compute the thermodynamic variable for Jacobian matrix
                // YTp:             Array for species mass fraction, temperature and pressure.
                // c_:              Species concentration.                  [kmol/m^3]
                // dNdtByV:         Instantaneous reaction rate.            [kmol/m^3/s]
                // ExpNegGstdByRT:  exp(-Gstd/(Ru*T)).                      [dimLess]
                // Gstd:            Standard Gibbs free energy.             [J/kmol]
                // dBdT:            ddT(-Gstd)/Ru, Partial derivative Gstd 
                //                  w.r.t temperature divided by Ru.        [dimLess]
                // dCpdT:           Partial derivative of Specific heat 
                //                  capacity w.r.t temperature.             [dimLess]
                // Cp:              Specific heat capacity at constant 
                //                  pressure.                               [J/kg/k]
                // Ha:              Absolute enthalpy.                      [J/kg]
                // rhoMvj:          Mixture density multiplied by specific  
                //                  volume                                  [dimLess].
                // WiByrhoM:        Molecular weight of i-th species divided
                //                  by mixture density                      [m^3/kmol].
                // ddNdtByVdcTp     Jacobian matrix based on species concentration.
            inline void JacobianThermo
            (
                double p,
                double T,
                double* __restrict__ YTp,
                double* __restrict__ c_,
                double* __restrict__ dNdtByV,
                double* __restrict__ ExpNegGstdByRT,
                double* __restrict__ dBdT,
                double* __restrict__ dCpdT,
                double* __restrict__ Cp,
                double* __restrict__ Ha,
                double* __restrict__ rhoMvj,
                double* __restrict__ WiByrhoM,
                double* __restrict__ ddNdtByVdcTp
            ) const noexcept;


            inline void JacobianThermo
            (
                const double p,
                const double T_,
                double* __restrict__ cTp,
                double* __restrict__ ExpNegGstdByRT,
                double* __restrict__ dBdT,
                double* __restrict__ dCpdT,
                double* __restrict__ Cp,
                double* __restrict__ Ha
            ) const noexcept;

            // Compute the thermodynamic variable for reaction rate.
            inline void CpHaExpNegGstdByRT
            (
                const double T,
                double* __restrict__ Cp,
                double* __restrict__ Ha,
                double* __restrict__ ExpNegGstdByRT
            )const noexcept;

            inline void cphaExpNegGstdByRT
            (
                const double T,
                double* __restrict__ Cp,
                double* __restrict__ Ha,
                double* __restrict__ ExpNegGstdByRT
            )const noexcept;

            // Compute the exp(-Gstd/(Ru*T)).           
            inline void ExpNegGstdByRT (const double T, double* ExpNegGstdByRT) const;

            // Compute the specific heat capacity and absolute enthalpy.
            inline void CpHa(const double T, double* __restrict__ Cp, double* __restrict__ Ha) const;

            // Compute the specific heat capacity and absolute enthalpy.
            inline void cpha(const double T, double* __restrict__ Cp, double* __restrict__ Ha) const;

            // Compute the mass density of species                          [kg/m^3]     
            inline double rho(const double P, const double T, const unsigned i)const;

        // Chemical equilibrium function

            // Update the variable (Pstd/(Ru*T))^sumVki for global reactions.
            inline void update_Pow_pByRT_SumVki(const double T) const;

            // Update the variable (Pstd/(Ru*T))^sumVki for element reactions.
            inline void update_Pow_pByRT_SumVki2(const double T) const;

        // Third body reaction function

            // Return the derivatives of third body efficiency w.r.t species concentration.
            inline double dMdC(const unsigned int ri, const unsigned int cj) const noexcept;

            // Compute the third body efficiency.
            //inline void M(const double* __restrict__ C, double* __restrict__ M) const;

        // Pressure depedent reaction function

            // Compute the F using SRI form.
            inline double SRI_F
            (
                const double T,
                const double Pr,
                const unsigned i
            ) const;

            // Compute the F using SRI form, vectorized version.
            inline void SRI_F
            (
                const double T, 
                const double* __restrict__ Pr, 
                double* __restrict__ F,
                const unsigned int i
            ) const;

            // Compute the F, the derivative of F w.r.t temperature
            // and the derivative of F w.r.t Pr.
            // Pr: reduced pressure.
            inline void SRI_F_dFdT_dFdPr
            (
                const double T,
                const double Pr,
                const unsigned int i,
                double& __restrict__ F,
                double& __restrict__ dFdT,
                double& __restrict__ dFdPr
            ) const;

            // Vectorized
            inline void SRI_F_dFdT_dFdPr
            (
                const double T,
                __m256d& __restrict__ Pr,
                __m256d& __restrict__ F,
                __m256d& __restrict__ dFdT,
                __m256d& __restrict__ dFdPr,
                const unsigned int i
            ) const;

            // Compute the F using Troe form.
            inline double Troe_F
            (
                const double T,
                const double Pr, 
                const unsigned int i 
            ) const noexcept;

            // Compute the F using Troe form, vectorized version.
            inline void Troe_F
            (
                const double T, 
                const double* __restrict__ Pr, 
                double* __restrict__ F,
                const unsigned int i
            ) const noexcept;

            void Troe_F_1() const noexcept;

            void Troe_F_2() const noexcept;         
            
            void Troe_F_3() const noexcept;   

            // Compute the F using Troe form, vectorized version.
            inline void Troe_F
            (
                const double T, 
                __m256d& __restrict__ Pr_, 
                __m256d& __restrict__ F_,
                const unsigned int i
            ) const noexcept;

            // Compute the F, the derivative of F w.r.t temperature
            // and the derivative of F w.r.t Pr.
            // Pr: reduced pressure.
            inline void Troe_F_dFdT_dFdPr
            (
                const double T,
                const double Pr,
                const size_t i,
                double& __restrict__ F,
                double& __restrict__ dFdT,
                double& __restrict__ dFdPr
            ) const noexcept;

            void Troe_Jac_1() const noexcept;

            void Troe_Jac_2() const noexcept;

            void Troe_Jac_3() const noexcept;

            // Vectorized
            inline void Troe_F_dFdT_dFdPr
            (
                const double T,
                __m256d& __restrict__ Pr,
                __m256d& __restrict__ F,
                __m256d& __restrict__ dFdT,
                __m256d& __restrict__ dFdPr,
                const size_t i
            ) const noexcept;

        // Jacobian matrix function

            // Compute the Jacobian matrix and reaction rate
            void ddNdtByVdcTp    
            (
                const double p,
                const double T,
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
            ) const noexcept;

            // Convert the concentration based Jacobian matrix
            // to the mass fraction based Jacobian matrix

                void ddYdtdY_Vec
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY_Vec1
                (
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY_Vec1_0
                (
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY_Vec1_1
                (
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY_Vec1_2
                (
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY_Vec1_3
                (
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp,
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY2
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdY3
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                // Neglecting the non-diagonal element in dC/dY
                void FastddYdtdY
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                void FastddYdtdY_Vec
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,  
                    const double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdTP_Vec
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,
                    const double* __restrict__ C,  
                    double* __restrict__ dYTpdt,
                    const double* __restrict__ YTp,  
                    double* __restrict__ J
                ) const noexcept;

                void ddYdtdTP_Vec_0
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,
                    const double* __restrict__ C,  
                    double* __restrict__ dYTpdt,
                    const double* __restrict__ YTp,  
                    double* __restrict__ J
                ) const noexcept;  

                void ddYdtdTP_Vec_1
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,
                    const double* __restrict__ C,  
                    double* __restrict__ dYTpdt,
                    const double* __restrict__ YTp,  
                    double* __restrict__ J
                ) const noexcept;  

                void ddYdtdTP_Vec_2
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,
                    const double* __restrict__ C,  
                    double* __restrict__ dYTpdt,
                    const double* __restrict__ YTp,  
                    double* __restrict__ J
                ) const noexcept;   

                void ddYdtdTP_Vec_3
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM,
                    const double* __restrict__ C,  
                    double* __restrict__ dYTpdt,
                    const double* __restrict__ YTp,  
                    double* __restrict__ J
                ) const noexcept;                

                void ddYdtdTP
                (
                    const double* __restrict__ ExpNegGstdByRT,
                    const double* __restrict__ ddNdtByVdcTp,
                    const double* __restrict__ rhoMvj,
                    const double* __restrict__ WiByrhoM, 
                    const double* __restrict__ C, 
                    double* __restrict__ dYTpdt,  
                    const double* __restrict__ YTp, 
                    double* __restrict__ J
                ) const noexcept;

                void ddTdtdYT
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
                ) const noexcept;

                void ddTdtdYT_Vec
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
                ) const noexcept;

                void ddTdtdYT_Vec_0
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
                ) const noexcept;

                void ddTdtdYT_Vec_1
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
                ) const noexcept;

                void ddTdtdYT_Vec_2
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
                ) const noexcept;          
                
                void ddTdtdYT_Vec_3
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
                ) const noexcept;                  

            // for one-one reactions. e.g. A<=>B.
            void inline JF11
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;

            void inline JF11
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;

            // for one-two reactions. e.g. A<=>B+B, A<=>B+C.
            void inline JF12
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;

            void inline JF12
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;

            // for one-three reactions. e.g. A<=>B+B+C, A<=>B+C+D.
            void inline JF13
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;


            void inline JF13
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;

            // for two-one reactions. e.g. A+A<=>B, A+B<=>C.
            void inline JF21
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;   
            
            
            void inline JF21
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;        

            // for two-two reactions. e.g. A+A<=>B+C, A+B<=>C+D.
            void inline JF22
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;   

            void inline JF22
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;   

            // for two-three reactions. e.g. A+A<=>B+C+D, A+B<=>C+D+D.
            void inline JF23
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept; 


            void inline JF23
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept; 

            // for three-one reactions. e.g. A+A+A<=>B, A+B+C<=>D.
            void inline JF31
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;  


            void inline JF31
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;  

            // for three-two reactions. e.g. A+A+A<=>B+C, A+B+C<=>D+E.
            void inline JF32
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;  

            void inline JF32
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;  

            // for three-three reactions. e.g. A+A+A<=>B+C+D, A+B+C<=>D+E+F.
            void inline JF33
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;  

            void inline JF33
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;    

            // for general reactions. e.g. A+A+A+A<=>B+C+D, A+B+C<=>D+E+F.
            void inline JFG
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp,
                const double* __restrict__ ExpNegGbyRT,
                const double* __restrict__ dBdT
            )const noexcept;  

            void inline JFG
            (
                unsigned int i,
                double Kf,
                double dKfdT,
                const double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ddNdtByVdcTp
            )const noexcept;    

        // Reaction rate function

            // Chemistry time scale(OpenFOAM-10)
            void Tc
            (
                int celli,
                double p,
                double T,
                double* C,
                double& sumW,
                double& sumWRateByCTot
            ) const noexcept;

            // Compute the reaction rate
            void dNdtByV
            (
                double p,
                double T,
                double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ExpNegGstdbyRT
            ) const noexcept;

            // Compute the reaction rate
            void dNdtByV
            (
                double p,
                double T,
                double* __restrict__ C,
                double* __restrict__ dNdtByV,
                double* __restrict__ ExpNegGstdbyRT,
                double* __restrict__ Cp,
                double* __restrict__ Ha
            ) const noexcept;


            // for one-one reactions. e.g. A<=>B.
            void inline RF11
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update11
            (
                double q,
                double*  __restrict__ sl0,
                double*  __restrict__ sr0
            )const noexcept;

            // for one-two reactions. e.g. A<=>B+B, A<=>B+C.
            void inline RF12
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update12
            (
                double q,
                double*  __restrict__ sl0,
                double*  __restrict__ sr0,
                double*  __restrict__ sr1
            )const noexcept;            

            // for one-three reactions. e.g. A<=>B+B+C, A<=>B+C+D.
            void inline RF13
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update13
            (
                double q,
                double*  __restrict__ sl0,
                double*  __restrict__ sr0,
                double*  __restrict__ sr1,
                double*  __restrict__ sr2
            )const noexcept;  

            // for two-one reactions. e.g. A+A<=>B, A+B<=>C.
            void inline RF21
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update21
            (
                double q,
                double*  __restrict__ sl0,
                double*  __restrict__ sl1,
                double*  __restrict__ sr0
            )const noexcept; 

            // for two-two reactions. e.g. A+A<=>B+C, A+B<=>C+D.
            void inline RF22
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update22
            (
                double q,
                double*  __restrict__ sl1,
                double*  __restrict__ sl2,
                double*  __restrict__ sr1,
                double*  __restrict__ sr2
            )const noexcept; 

            // for two-three reactions. e.g. A+A<=>B+C+D, A+B<=>C+D+D.
            void inline RF23
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update23
            (
                double q,
                double*  __restrict__ sl0,
                double*  __restrict__ sl1,
                double*  __restrict__ sr0,
                double*  __restrict__ sr1,
                double*  __restrict__ sr2
            )const noexcept; 

            // for three-one reactions. e.g. A+A+A<=>B, A+B+C<=>D.
            void inline RF31
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update31
            (
                double q,
                double*  __restrict__ sl1,
                double*  __restrict__ sl2,
                double*  __restrict__ sl3,
                double*  __restrict__ sr1
            )const noexcept; 

            // for three-two reactions. e.g. A+A+A<=>B+C, A+B+C<=>D+E.
            void inline RF32
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update32
            (
                double q,
                double*  __restrict__ sl1,
                double*  __restrict__ sl2,
                double*  __restrict__ sl3,
                double*  __restrict__ sr1,
                double*  __restrict__ sr2
            )const noexcept; 


            // for three-three reactions. e.g. A+A+A<=>B+C+D, A+B+C<=>D+E+F.
            void inline RF33
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;

            void inline update33
            (
                double q,
                double*  __restrict__ sl1,
                double*  __restrict__ sl2,
                double*  __restrict__ sl3,
                double*  __restrict__ sr1,
                double*  __restrict__ sr2,
                double*  __restrict__ sr3
            )const noexcept; 


             // for reactions with reactants' or products' number larger than 4 
             // e.g. A+A+A+B<=>B+C+D, A+B+C<=>D+D+E+F.
            void inline RFG
            (
                unsigned int i,
                double Kf,
                const double* __restrict__ C,
                double*  __restrict__ dNdtByV,
                const double*  __restrict__ ExpNegGbyRT
            )const noexcept;
         
            using RFHandler = void (OptReaction::*)(unsigned int, double, const double*, double*, const double*) const;
            RFHandler RFTable[4][4] = {};

        // Math function

            inline double fastPow (const double x, const int order) const 
            {
                double xx = order < 0 ? 1/x : x ;

                double y = 1;

                for (int i = order; i != 0; i /= 2)
                {
                    if (i % 2)
                    {
                        y *= xx;
                    }

                    xx *= xx;
                }
                return y;
            }

            // s = v0 + v1 + v2 + v3
            inline double hsum4(__m256d v) const
            {
                __m128d lo = _mm256_castpd256_pd128(v);
                __m128d hi = _mm256_extractf128_pd(v,1);
                lo = _mm_add_pd(lo,hi);               // [v0+v2, v1+v3]
                __m128d sh = _mm_unpackhi_pd(lo,lo);  // [v1+v3, v1+v3]
                __m128d s = _mm_add_sd(lo, sh);       // [sum, …]
                return _mm_cvtsd_f64(s);
            }

        
            inline double get_elem0(__m256d vec) const
            {
                return _mm256_cvtsd_f64(vec); 
            }

            inline double get_elem1(__m256d vec) const
            {
                __m128d low = _mm256_castpd256_pd128(vec); 
                return _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
            }

            inline double get_elem2(__m256d vec) const
            {
                __m128d high = _mm256_extractf128_pd(vec, 1); 
                return _mm_cvtsd_f64(high);
            }

            inline double get_elem3(__m256d vec) const
            {
                __m128d high = _mm256_extractf128_pd(vec, 1); 
                return _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
            }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

 // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "ThermoFunctionI.H"
#include "ChemicalEquilibriumFunctionI.H"
#include "thirdBodyReactionI.H"
#include "pressureReactionFunctionI.H"
#include "JacobianFunctionI.H"
#include "ReactionFunctionI.H"

#ifdef NoRepository
    #include "OptReaction.C"
    #include "Jacobian.C"
    #include "ReactionRate.C"
    #include "Tc.C"    
#endif




// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
