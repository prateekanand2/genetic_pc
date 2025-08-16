#!/bin/bash

for b in "10K_real_8020" "10K_hclt_8020" "10K_rbm_8020" "10K_wgan_8020"; do
    # method=${b%%_*}  # Extract substring before the first underscore
    method=$(echo "$b" | cut -d'_' -f2)
    mapfile_path="rerun/bootstrap/impute_${b}_correct_map.txt"
    
    # Create the mapfile with lines 1-10000 if it doesn't exist
    if [ ! -f "$mapfile_path" ]; then
        echo "Mapfile $mapfile_path not found. Creating with 1 to 10000..."
        mkdir -p "$(dirname "$mapfile_path")"
        seq 1 10000 > "$mapfile_path"
    fi

    num_lines=$(wc -l < "$mapfile_path")
    
    # Skip if num_lines is 0
    if [ "$num_lines" -eq 0 ]; then
        echo "Mapfile $mapfile_path is empty. Skipping submission."
        continue
    fi

    qsub -v METHOD=${method},METHOD_FULL=${b},MAPFILE=${mapfile_path} -t 1-${num_lines}:100 impute5.sh
done