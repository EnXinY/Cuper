#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#include "mmio_b.h"

// read matrix infomation from mtx file
int mmio_info(int *m, int *n, int64_t *nnz, int *isSymmetric, char *filename)
{
    int m_tmp, n_tmp;
    int64_t nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int64_t nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
    {
        INFO_LOG("regular file %s don't exit\n", filename);
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    INFO_LOG("%d %d %ld\n", m_tmp, n_tmp, nnz_mtx_report);

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int64_t *csrRowPtr_counter = (int64_t *)malloc((m_tmp + 1) * sizeof(int64_t));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int64_t));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int idxi, idxj;
    double fval, fval_im;
    int ival;
    int returnvalue;
    for (int64_t i = 0; i < nnz_mtx_report; i++)
    {
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int64_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    int64_t old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int64_t i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

// read matrix infomation from mtx file
int mmio_data(int64_t *csrRowPtr, int *csrColIdx, double *csrVal, char *filename)
{
    int m_tmp, n_tmp;
    int64_t nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int64_t nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else
    {
        // printf("input matrix is symmetric = false\n");
    }

    int64_t *csrRowPtr_counter = (int64_t *)malloc((m_tmp + 1) * sizeof(int64_t));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int64_t));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int idxi, idxj;
    double fval, fval_im;
    int ival;
    int returnvalue;
    for (int64_t i = 0; i < nnz_mtx_report; i++)
    {
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int64_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    int64_t old_val, new_val;

    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int64_t i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp + 1) * sizeof(int64_t));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(int64_t));

    if (isSymmetric_tmp)
    {
        for (int64_t i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int64_t offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int64_t offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int64_t i = 0; i < nnz_mtx_report; i++)
        {
            int64_t offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}

#endif
