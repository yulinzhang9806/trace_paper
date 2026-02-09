
import sys
import argparse
import numpy as np
from cyvcf2 import VCF, Writer
from pyfaidx import Fasta
import pysam
import gzip

def process_vcf(vcf_path, fasta_path, output_path, unpolarized_path, flipped_path):
    fasta = Fasta(fasta_path)
    ancestral_sequence = fasta[0]  # Assuming the first sequence in the FASTA is the ancestral

    count = 0
    flipped_sites = []
    unpolarized_sites = []

    with gzip.open(vcf_path, 'rt') as vcf_in, open(output_path, 'w') as vcf_out:
        for line in vcf_in:
            if line.startswith('##'):
                vcf_out.write(line)
                continue

            if line.startswith('#CHROM'):
                # Add custom INFO fields for ancestral allele, flipped status, and AC
                vcf_out.write('##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele">\n')
                vcf_out.write('##INFO=<ID=Flipped,Number=0,Type=Flag,Description="Indicates the variant was flipped to match the ancestral allele">\n')
                vcf_out.write('##INFO=<ID=AC,Number=1,Type=Integer,Description="Allele count in genotypes">\n')
                vcf_out.write(line)
                continue

            count += 1
            parts = line.strip().split('\t')
            chrom, pos, _, ref, alts, _, _, info, format, *samples = parts
            pos = int(pos)
            ancestral_allele = str(ancestral_sequence[pos - 1]).upper()

            info_updates = f"AA={ancestral_allele};"
            ac_count = 0  # Initialize allele count

            # Check if the ancestral allele matches REF or any ALT
            if ancestral_allele == ref.upper():
                # The REF allele is ancestral, no need to flip
                # Directly count AC from the genotypes
                for sample in samples:
                    genotype, *rest = sample.split(':')
                    ac_count += genotype.count('1')
            elif ancestral_allele in alts.split(','):
                # The ancestral allele matches an ALT allele; need to flip
                flipped_sites.append(str(pos))
                info_updates += "Flipped;"

                alts_list = alts.split(',')
                alt_index = alts_list.index(ancestral_allele)
                alts_list[alt_index], ref = ref, ancestral_allele  # Swap REF and ALT
                alts = ','.join(alts_list)

                # Flip genotypes and count AC
                new_samples = []
                for sample in samples:
                    genotype, *rest = sample.split(':')
                    flipped_genotype = '|'.join(['1' if g == '0' else '0' for g in genotype.split('|')])
                    ac_count += flipped_genotype.count('1')  # Count '1's for AC
                    new_samples.append(':'.join([flipped_genotype] + rest))
                samples = new_samples  # Update the sample list to flipped genotypes

            else:
                # If neither REF nor ALT matches the ancestral allele, skip this variant
                unpolarized_sites.append(str(pos))
                continue

            info_updates += f"AC={ac_count}"  # Append AC count to the INFO field
            parts = [chrom, str(pos), '.', ref, alts, '.', '.', info_updates, format] + samples
            vcf_out.write('\t'.join(parts) + '\n')

    # Write positions of flipped and unpolarized sites to separate files
    with open(flipped_path, 'w') as f:
        f.write('\n'.join(flipped_sites) + '\n')

    with open(unpolarized_path, 'w') as f:
        f.write('\n'.join(unpolarized_sites) + '\n')

    print(f"Total Variants Processed: {count}")
    print(f"Variants Not Matched to Ancestral Sequence: {len(unpolarized_sites)}")
    print(f"Variants with Flipped Genotypes and Alleles: {len(flipped_sites)}")
    print(f"Note: AC values are included in the output VCF INFO field for each variant.")



def main():
    parser = argparse.ArgumentParser(description='Process a VCF file with an ancestral genome.')

    parser.add_argument('-vcf', type=str, required=True, help='Path to the input VCF file.')
    parser.add_argument('-fasta', type=str, required=True, help='Path to the ancestral genome FASTA file.')
    parser.add_argument('-output', type=str, required=True, help='Path to the output VCF file.')
    parser.add_argument('-unpolarized', type=str, required=True, help='Path to the file of unpolarized sites.')   
    parser.add_argument('-flipped', type=str, required=True, help='Path to the file of flipped sites.') 

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    print(f"Input VCF File: {args.vcf}")
    print(f"Input Ancestral Sequence Fasta File: {args.fasta}")
    print(f"Output VCF File: {args.output}")
    print(f"Unpolarized Sites File: {args.unpolarized}")
    print(f"Flipped Sites File: {args.flipped}")

    process_vcf(args.vcf, args.fasta, args.output, args.unpolarized, args.flipped)

if __name__ == "__main__":
    main()
