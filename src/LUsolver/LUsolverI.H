
inline double& LUsolver::operator()
(
    const unsigned int i,
    const unsigned int j
)
{
    return v_[i*N + j];
}


inline double LUsolver::operator() 
(
    const unsigned int i,
    const unsigned int j
)const
{
    return v_[i*N + j];
}


inline unsigned int& LUsolver::pivotIndice
(
    const unsigned int i
)
{
    return this->pivotIndice_[i];
}


inline unsigned int LUsolver::pivotIndice
(
    const unsigned int i
) const
{
    return this->pivotIndice_[i];
}
inline double LUsolver::hsum4(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v,1);
    lo = _mm_add_pd(lo,hi);               
    __m128d sh = _mm_unpackhi_pd(lo,lo);  
    __m128d s = _mm_add_sd(lo, sh);       
    return _mm_cvtsd_f64(s);
}
