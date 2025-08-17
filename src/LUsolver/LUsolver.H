


#ifndef LUsolver_H
#define LUsolver_H
#include <vector>
#include <immintrin.h>
class LUsolver
{
    // Private data

        //- Matrix size
        unsigned int N;

        //- Remain
        unsigned int Remain;

        //- Data pointer
        double* __restrict__ v_;

        //- Store the global index of target row.
        unsigned int* pivotIndice_;

        //- Store the row address
        std::vector<double*> rowPtr;

        //- Save the reciprocal of diagonal elements
        std::vector<double> invD;

        //- Whether the data is own
        bool owner;

public:

    // Constructor.

        // Constructor from given size.
        LUsolver(int n); 

        // Constructor from external data.
        LUsolver(double* externalData, int size);

        // Disable default constructor 
        LUsolver()=delete;  

    // Destructor

        ~LUsolver();

    // Member function

        // Access

            //- (i, j) element access.
            inline double& operator()(const unsigned int i,const unsigned int j);

            //- (i, j) element access. 
            inline double operator() (const unsigned int i,const unsigned int j)const;
 
            //- pivotIndice access.
            inline unsigned int& pivotIndice(const unsigned int i);

            //- pivotIndice access.
            inline unsigned int pivotIndice(const unsigned int i) const;

        // Print the variable

            void printMatrix(double*A,int mRows, int nCol);

            void printMatrix();

            void printPivotIndice();

        // Block LU decomposition algorithm

            // using 4*4 block.
            void Block4LUDecompose();

            // Solve Ax=b.
            void xSolve_Serial(double* __restrict__ b);
            void xSolve_Vec_0(double* __restrict__ b);
            void xSolve_Vec_1(double* __restrict__ b);
            void xSolve_Vec_2(double* __restrict__ b);
            void xSolve_Vec_3(double* __restrict__ b);
            void xSolve(double* __restrict__ b);            

            // Sub functions

                // LU decomposition of 4*4 block.
                void LUDecompose4(unsigned int k0);
                void LUDecompose4Old(unsigned int k0);
                void LUDecompose4();
                void LUDecompose4_2();           
                void LUDecompose4_3();        
                
                void forwardSitituate4(unsigned int k0,unsigned int k1);
                void forwardSitituate4_0(unsigned int k0,unsigned int k1);
                void forwardSitituate4_1(unsigned int k0,unsigned int k1);
                void forwardSitituate4_2(unsigned int k0,unsigned int k1);
                void forwardSitituate4_3(unsigned int k0,unsigned int k1);

                void backSitituate4(unsigned int k0,unsigned int k1);
                void backSitituate4_0(unsigned int k0,unsigned int k1);
                void backSitituate4_1(unsigned int k0,unsigned int k1);
                void backSitituate4_2(unsigned int k0,unsigned int k1);
                void backSitituate4_3(unsigned int k0,unsigned int k1);                

                void permutation0(unsigned int k0,unsigned int k1);
                void permutation1(unsigned int k0,unsigned int k1);
                void permutation2(unsigned int k0,unsigned int k1);
                void permutation3(unsigned int k0,unsigned int k1);
                void permutationold(unsigned int k0,unsigned int k1);
                void permutation();
                void permutation_2();
                void permutation_3();

                void UpdateL22U22(unsigned int k0,unsigned int k1);
                void UpdateL22U22_Vec(unsigned int k0,unsigned int k1);
                void UpdateL22U22_Vec2(unsigned int k0,unsigned int k1);   
                void UpdateL22U22_Vec2_0(unsigned int k0,unsigned int k1);   
                void UpdateL22U22_Vec2_1(unsigned int k0,unsigned int k1);  
                void UpdateL22U22_Vec2_2(unsigned int k0,unsigned int k1);  
                void UpdateL22U22_Vec2_3(unsigned int k0,unsigned int k1);   
                
        void LUDecompose_s();

        // Read square matrix from text.
        void ReadTxt(std::string fileName);

        // Reassign a square matrix
        void ReAssign(double* externalData);

        inline double hsum4(__m256d v);
};

#include "LUsolverI.H"

#ifdef NoRepository
    #include "LUsolver.C" 
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
