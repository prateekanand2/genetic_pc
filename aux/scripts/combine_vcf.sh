#!/bin/bash

# Check if the required number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <vcf_file1> <vcf_file2> <output_vcf>"
    exit 1
fi

# Input arguments
VCF1=$1
VCF2=$2
OUTPUT_VCF=$3

# Check if input files exist
if [ ! -f "$VCF1" ]; then
    echo "Error: File '$VCF1' not found!"
    exit 1
fi

if [ ! -f "$VCF2" ]; then
    echo "Error: File '$VCF2' not found!"
    exit 1
fi

# Merge the VCF files using bcftools
/scratch2/prateek/bcftools/bcftools merge "$VCF1" "$VCF2" -o "$OUTPUT_VCF" --output-type z

# Check if the merge was successful
if [ $? -eq 0 ]; then
    echo "VCF files successfully combined into '$OUTPUT_VCF'."
else
    echo "Error: Failed to combine VCF files."
    exit 1
fi
