#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __riscv
#include <riscv_vector.h>
#endif

void intialize_matrix(int** matrix, int shape0, int shape1, int value) {
    for (int i = 0; i < shape0; i++) {
        for (int j = 0; j < shape1; j++) {
            matrix[i][j] = value++;
        }
    }
}

void intialize_to_zero(int** matrix, int shape0, int shape1) {
    for (int i = 0; i < shape0; i++) {
        for (int j = 0; j < shape1; j++) {
            matrix[i][j] = 0;
        }
    }
}

void pack(int** matrix, int* array, int shape0, int shape1, int inner_parallel, int inner_reduction) {
    int index = 0;
    for (int i = 0; i < shape0; i+=inner_parallel) {
        for (int k = 0; k < shape1; k+=inner_reduction) {
            for (int i0 = 0; i0 < inner_parallel; ++i0) {
                for (int k0 = 0; k0 < inner_reduction; ++k0) {
                    array[index++] = matrix[i + i0][k + k0];
                }
            }
        }
    }
}

void transpose(int** matrix, int** tranposed, int shape0, int shape1) {
    for (int i = 0; i < shape0; i++) {
        for (int j = 0; j < shape1; j++) {
            tranposed[j][i] = matrix[i][j];
        }
    }
}

void unpack(int *array, int** matrix, int shape0, int shape1, int inner_parallel, int inner_reduction) {
    int index = 0;
    for (int i = 0; i < shape0; i+=inner_parallel) {
        for (int k = 0; k < shape1; k+=inner_reduction) {
            for (int i0 = 0; i0 < inner_parallel; ++i0) {
                for (int k0 = 0; k0 < inner_reduction; ++k0) {
                    matrix[i + i0][k + k0] = array[index++];
                }
            }
        }
    }
}

#ifdef __riscv
void mmt4d_rvv(int* lhs_packed, int* rhs_packed, int* res_packed, int M, int N, int K, int M0, int N0, int K0) {
    for (int i = 0; i < M; i++) {
        int* lhs_panel = &lhs_packed[i * K * M0 * K0];
        for (int j = 0; j < N; j++) {
            int* rhs_panel = &rhs_packed[j * K * N0 * K0];
            int* out_panel = &res_packed[i * N * M0 * N0 + j * M0 * N0];

            vint32m2_t  acc0, acc1, acc2, acc3;
            size_t vl = N0;

            acc0 = __riscv_vle32_v_i32m2(out_panel, vl);
            acc1 = __riscv_vle32_v_i32m2(out_panel + N0, vl);
            acc2 = __riscv_vle32_v_i32m2(out_panel + N0 * 2, vl);
            acc3 = __riscv_vle32_v_i32m2(out_panel + N0 * 3, vl);

            int *lhs_ptr = lhs_panel;

            for (int k = 0; k < K; ++k) {
                vint32m2_t rhs = __riscv_vle32_v_i32m2(rhs_panel, vl);
                rhs_panel += N0;

                acc0 = __riscv_vmacc_vx_i32m2(acc0, *lhs_ptr++, rhs, vl);
                acc1 = __riscv_vmacc_vx_i32m2(acc1, *lhs_ptr++, rhs, vl);
                acc2 = __riscv_vmacc_vx_i32m2(acc2, *lhs_ptr++, rhs, vl);
                acc3 = __riscv_vmacc_vx_i32m2(acc3, *lhs_ptr++, rhs, vl);
            }
            __riscv_vse32_v_i32m2(out_panel, acc0, vl);
            __riscv_vse32_v_i32m2(out_panel + N0, acc1, vl);
            __riscv_vse32_v_i32m2(out_panel + N0 * 2, acc2, vl);
            __riscv_vse32_v_i32m2(out_panel + N0 * 3, acc3, vl);
        }
    }
}
#endif

void mmt4d(int* lhs_packed, int* rhs_packed, int* res_packed, int lhs_strides[3], int rhs_strides[3], int res_strides[3], int M1, int N1, int K1, int M0, int N0, int K0) {
    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < N1; j++) {
            for (int k = 0; k < K1; k++) {
                for (int i0 = 0; i0 < M0; ++i0) {
                    for (int j0 = 0; j0 < N0; ++j0) {
                        for (int k0 = 0; k0 < K0; ++k0) {
                            res_packed[i * res_strides[0] + j * res_strides[1] + i0 * res_strides[2] + j0] += lhs_packed[i * lhs_strides[0] + k * lhs_strides[1] + i0 * lhs_strides[2] + k0] * rhs_packed[j * rhs_strides[0] + k * rhs_strides[1] + j0 * rhs_strides[2] + k0];
                        }
                    }
                }
            }
        }
    }
}

void matmul(int** lhs, int** rhs, int** res, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                res[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
}

void print_packed_matrix(const int *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%d ", matrix[i]);
    }
    printf("\n");
}

void print_matrix(int** matrix, int shape0, int shape1) {
    for (int i = 0; i < shape0; ++i) {
        for (int j = 0; j < shape1; ++j) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void compare(int **matrix1, int **matrix2, int shape0, int shape1) {
    for (int i = 0; i < shape0; ++i) {
        for (int j = 0; j < shape1; ++j) {
            if (matrix1[i][j] != matrix2[i][j]) {
                printf("Error at [%d], [%d] ---> %d, %d\n", i, j, matrix1[i][j], matrix2[i][j]);
                return;
            }
        }
    }
}

void calculateStrides(int *lhs_strides, int* rhs_strides, int*res_strides, int M1, int N1, int K1, int M0, int N0, int K0) {
    lhs_strides[2] = K0;
    lhs_strides[1] = M0 * lhs_strides[2];
    lhs_strides[0] = K1 * lhs_strides[1];
    rhs_strides[2] = K0;
    rhs_strides[1] = N0 * rhs_strides[2];
    rhs_strides[0] = K1 * rhs_strides[1];
    res_strides[2] = N0;
    res_strides[1] = M0 * res_strides[2];
    res_strides[0] = N1 * res_strides[1];
}

void free2D(int** matrix, int shape0) {
    for (int i = 0; i < shape0; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int agrc, char* argv[]) {
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int M0 = atoi(argv[4]);
    int N0 = atoi(argv[5]);
    int K0 = atoi(argv[6]);

    int M1 = M / M0;
    int N1 = N / N0;
    int K1 = K / K0;

    int** lhs = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        lhs[i] = (int*)malloc(K * sizeof(int));
    }
    int** rhs = (int**)malloc(K * sizeof(int*));
    for (int i = 0; i < K; i++) {
        rhs[i] = (int*)malloc(N * sizeof(int));
    }
    int** rhs_t = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        rhs_t[i] = (int*)malloc(K * sizeof(int));
    }
    int** res = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        res[i] = (int*)malloc(N * sizeof(int));
    }
    int** check = (int**)malloc(M * sizeof(int*));
    for (int i = 0; i < M; i++) {
        check[i] = (int*)malloc(N * sizeof(int));
    }

    int* lhs_packed = (int*)malloc(M * K * sizeof(int));
    int* rhs_t_packed = (int*)malloc(N * K * sizeof(int));
    int* res_packed = (int*)malloc(M * N * sizeof(int));

    int lhs_strides[3];
    int rhs_strides[3];
    int res_strides[3];

    calculateStrides(lhs_strides, rhs_strides, res_strides, M1, N1, K1, M0, N0, K0);
    printf("LHS strides: %d, %d, %d\n", lhs_strides[0], lhs_strides[1], lhs_strides[2]);
    printf("RHS strides: %d, %d, %d\n", rhs_strides[0], rhs_strides[1], rhs_strides[2]);
    printf("RES strides: %d, %d, %d\n", res_strides[0], res_strides[1], res_strides[2]);

    intialize_matrix(lhs, M, K, 1);
    intialize_matrix(rhs, K, N, 65);
    intialize_to_zero(res, M, N);
    intialize_to_zero(check, M, N);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul(lhs, rhs, check, M, N, K);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double matmul_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time for matmul: %f seconds\n", matmul_time);

    clock_gettime(CLOCK_MONOTONIC, &start);
    pack(lhs, lhs_packed, M, K, M0, K0);
    transpose(rhs, rhs_t, K, N);
    pack(rhs_t, rhs_t_packed, N, K, N0, K0);
    pack(res, res_packed, M, N, M0, N0);

    #ifdef __riscv
        mmt4d_rvv(lhs_packed, rhs_t_packed, res_packed, M1, N1, K1, M0, N0, K0);
    #else
        mmt4d(lhs_packed, rhs_t_packed, res_packed, lhs_strides, rhs_strides, res_strides, M1, N1, K1, M0, N0, K0);
    #endif

    unpack(res_packed, res, M, N, M0, N0);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double mmt4d_rvv_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time for mmt4d: %f seconds\n", mmt4d_rvv_time);
    compare(res, check, M, N);

    free2D(lhs, M);
    free2D(rhs, K);
    free2D(res, M);
    free2D(rhs_t, N);
    free2D(check, M);
    free(lhs_packed);
    free(rhs_t_packed);
    free(res_packed);
    return 0;
}
