#!/bin/bash
#!/bin/bash
# Script de experimentos para v1 - Método de Jacobi (secuencial base)
# Ejecuta 10 repeticiones por cada combinación (n, cflags)

#SBATCH -n 1 -c 64 -t 06:00:00 --mem=200
#SBATCH --job-name p2acg14
#SBATCH -o salida_v3_guided_O3_%j.out

REPETICIONES=10
VALORES_N="1250 2000 3200"
VALORES_CFLAGS="-O3"
VALORES_HILOS="1 4 16 32"

RESULTADOS="resultados_v3_guided_O3.csv"
echo "version,n,cflags,hilos,schedule,acumulacion,repeticion,norm2,ciclos" > "$RESULTADOS"

for cflags in $VALORES_CFLAGS; do
    for acumulacion in reduction critical; do
        if [ "$acumulacion" = "reduction" ]; then
            use_reduction=1
        else
            use_reduction=0
        fi

        echo "=== Compilando v3 con CFLAGS=${cflags}, schedule=guided, acumulacion=${acumulacion} ==="
        make v3 -B CFLAGS="${cflags}" CPPFLAGS="-DUSE_REDUCTION=${use_reduction} -DSCHED_KIND=guided" 2>&1

        for n in $VALORES_N; do
            for hilos in $VALORES_HILOS; do
                for rep in $(seq 1 $REPETICIONES); do
                    output=$(./v3 $n $hilos)
                    echo "v3,${n},${cflags},${hilos},guided,${acumulacion},${rep},${output}" >> "$RESULTADOS"
                done
                echo "  v3 | n=${n} | hilos=${hilos} | cflags=${cflags} | schedule=guided | acumulacion=${acumulacion} | completado"
            done
        done
    done
done

echo ""
echo "=== Test v3 finalizado ==="
echo "Resultados guardados en: ${RESULTADOS}"
