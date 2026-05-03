#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Alineación en bytes
#define ALIGN 32
// Redondear el tamaño al siguiente múltiplo de 32 con bitwise
#define ALIGN_UP(size) (((size) + ALIGN - 1) & ~(size_t)(ALIGN - 1))


#ifndef USE_REDUCTION
#define USE_REDUCTION 1     // 1 -> reduction, 0 -> critical
#endif
#ifndef SCHED_KIND
#define SCHED_KIND static   // -> static, dynamic, guided
#endif                      // no usar: auto o runtime


static void free_matrix(double** A, int rows);
static int jacobi(double** A, double* b, double* x, int n, double tol, int max_iter, int num_threads);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <n> [c]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "El tamaño 'n' debe ser positivo.\n");
        return 1;
    }

    int c = atoi(argv[2]); 
    if(c <= 0) {
        fprintf(stderr, "El número de hilos 'c' debe ser positivo.\n");
        return 1;
    }

    // Reservamos el array de punteros para las filas
    double** A = (double**)aligned_alloc(ALIGN, ALIGN_UP(n * sizeof(double*)));
    if (A == NULL) {
        fprintf(stderr, "Error reservando memoria para A.\n");
        return 1;
    }

    // Reservamos cada fila de la matriz
    for (int i = 0; i < n; i++) {
        A[i] = (double*)aligned_alloc(ALIGN, ALIGN_UP(n * sizeof(double)));
        if (A[i] == NULL) {
            fprintf(stderr, "Error reservando la fila %d.\n", i);
            free_matrix(A, i);
            return 1;
        }
    }

    // Vectores de términos independientes (b) y solución (x)
    double* b = (double*)aligned_alloc(ALIGN, ALIGN_UP(n * sizeof(double)));
    double* x = (double*)aligned_alloc(ALIGN, ALIGN_UP(n * sizeof(double)));
    
    if (b == NULL || x == NULL) {
        fprintf(stderr, "Error reservando memoria para los vectores.\n");
        free(b);
        free(x);
        free_matrix(A, n);
        return 1;
    }

    // Inicialización de datos
    // Semilla n para que siempre calcule lo mismo
    srand(n);

    // Matriz diagonalmente dominante
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                // Rellenamos números entre 1 y 10
                A[i][j] = rand() % 10 + 1; 
                sum += fabs(A[i][j]);
            }
        }
        // Forzamos la diagonal para que sea mayor que la suma del resto de su fila
        A[i][i] = sum + (rand() % 10 + 1);
        
        b[i] = rand() % 10 + 1;  // b con valores aleatorios
        x[i] = 0.0;     // x inicializado a 0
    }

    // Condiciones de parada
    double tol = 1e-5;
    int max_iter = 15000;

    // Ejecutar Jacobi
    if (jacobi(A, b, x, n, tol, max_iter, c) != 0) {
        free_matrix(A, n);
        free(b);
        free(x);
        return 1;
    }

    // Liberar memoria
    free_matrix(A, n);
    free(b);
    free(x);

    return 0;
}

// Función auxiliar para liberar memoria de matrices
static void free_matrix(double** A, int rows) {
    if (A == NULL) return;
    for (int i = 0; i < rows; i++) {
        free(A[i]);
    }
    free(A);
}

// Algoritmo de Jacobi paralelizado con OpenMP
static int jacobi(double** A, double* b, double* x, int n, double tol, int max_iter, int num_threads) {
    
    double* x_new = (double*)aligned_alloc(ALIGN, ALIGN_UP(n * sizeof(double)));
    if (x_new == NULL) {
        fprintf(stderr, "Error de memoria en x_new.\n");
        return 1;
    }

    int done = 0; // flag para indicar si se ha alcanzado la convergencia
    double final_norm2 = 0.0; // variable para almacenar la norma final
    double norm2 = 0.0;

    /* ##################################################### */
    // Parte de cómputo, medimos
    start_counter();
    
    /*  #pragma omp parallel --> crea una región paralela y forma un equipo de hilos
        num_threads(num_threads) --> fija el núm. de hilos de la región paralela
        shared(...) --> indica que variables son comunes a todos los hilos
        |__ private(...) --> variables que cada hilo tiene su propia copia
    */
    #pragma omp parallel num_threads(num_threads) shared(A,b,x,x_new,n,tol,max_iter,done, final_norm2)
    {
        for (int iter = 0; iter < max_iter; iter++) {            
            
            /* single --> hace que un bloque lo ejecute solo 1 hilo (no necesariamente el maestro)
                   presenta barrara implícita al final del bloque, por lo que ningún hilo continúa hasta que todos hayan terminado el bloque
            */
            #pragma omp single
            {
                norm2 = 0.0;
            }
            
            /*  #pragma omp for --> reparte las iteraciones del bucle entre los hilos
                schedule(...) --> distribuye las iteraciones dependiendo del modo
                    |__ static --> en bloques fijos asignados antes de empezar
                    |__ dynamic --> los hilos van pidiendo trabajo a medida que terminan el suyo
                    |__ guided --> parecido a dynamic, pero con bloques grandes al principio y más pequeños al final
                    |__ auto, runtime
                reduction(+:norm2) --> cada hilo acumula norm2 y al final de la ejecucion suman sus resultados
                |_ #pragma omp critical --> hace que el bloque asociado lo ejecute solo 1 hilo de cada vez
            */
#if USE_REDUCTION

            #pragma omp for reduction(+:norm2) schedule(SCHED_KIND)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;

                for (int j = 0; j < i; j++) {
                    sigma += A[i][j] * x[j];
                }

                for (int j = i + 1; j < n; j++) {
                    sigma += A[i][j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / A[i][i];
                norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
            }

#else
            // Sin reduction, usamos critical para actualizar norm2 de forma segura
            double local_norm2 = 0.0;

            #pragma omp for schedule(SCHED_KIND)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;

                for (int j = 0; j < i; j++) {
                    sigma += A[i][j] * x[j];
                }

                for (int j = i + 1; j < n; j++) {
                    sigma += A[i][j] * x[j];
                }

                x_new[i] = (b[i] - sigma) / A[i][i];
                local_norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
            }

            #pragma omp critical
            {
                norm2 += local_norm2;
            }

#endif

            // Un solo hilo copia x_new a x, comprueba la convergencia y actualiza final_norm2
            #pragma omp single
            {
                for (int k = 0; k < n; k++) {
                    x[k] = x_new[k];
                }

                done = (sqrt(norm2) < tol);
                final_norm2 = norm2;
            }

            /* --> barrera: ningún hilo continúa hasta que todos lleguen a ella
                   en sí, no es necesario ponerla explícitamente porque single ya la incluye implícitamente
                   
                   es redundante en este caso, pero se queda por ahora. se puede usar si a single se le indica nowait
            */
            #pragma omp barrier
            if (done) break;
        }
    }
    
    // Parar reloj
    double ciclos = get_counter();
    /* ##################################################### */

    // Imprime la norma y los ciclos
    printf("%.12e,%.0f\n", final_norm2, ciclos);

    free(x_new);
    return 0;
}