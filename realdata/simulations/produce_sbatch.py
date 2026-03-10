import os
import sys
import argparse
import numpy as np

exclude_regions = []#[122000000, 142000000]]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate sbatch job files for singer submission. Please modify exclude regions in the code.")
    
    # Add arguments with prefixes
    parser.add_argument("--chunk-size", required=True, help="size of each chunk")
    parser.add_argument("--seq-end", required=True, help="end coordinate of the sequence")
    parser.add_argument("--sbatch-base", required=True, help="base sbatch file")
    parser.add_argument("--nposterior", required=True, help="number of posterior samples")
    parser.add_argument("--singer-outpref", required=True, help="singer output prefix")
    parser.add_argument("--output", required=True, help="prefix for output files")
    parser.add_argument("--chrom", required=True, help="chromosome name (int)")      
 
    args = parser.parse_args()
    chunk_size = int(args.chunk_size)
    seq_end = int(args.seq_end)
    sbatch_base = args.sbatch_base
    nposterior = int(args.nposterior)
    singer_outpref = args.singer_outpref
    output_prefix = args.output
    chrom = int(args.chrom)

    with open(sbatch_base, 'r') as f:
        sbatch_content = f.read()
    # Generate sbatch files
    start_pos = []
    jobid = 0
    for i in range(0, seq_end, chunk_size):
        start = i
        end = min(i + chunk_size, seq_end)
        skip = False
        for exclude_region in exclude_regions:
            if end > exclude_region[0] and end <= exclude_region[1]:
                print(f"Skip {start}-{end} because it is in the exclude region {exclude_region}")
                skip = True
            elif start < exclude_region[0] and end > exclude_region[1]:
                print(f"Skip {start}-{end} because it is in the exclude region {exclude_region}")
                skip = True
            elif start >= exclude_region[0] and start < exclude_region[1]:
                print(f"Skip {start}-{end} because it is in the exclude region {exclude_region}")
                skip = True
        if skip:
            continue
        sbatch_content_new = sbatch_content.replace("START", str(start)).replace("END", str(end))
        sbatch_content_new = sbatch_content_new.replace("JOBNAME", str(jobid))
        sbatch_content_new = sbatch_content_new.replace("CHROM", str(chrom))
        output_filename = f"{output_prefix}_{jobid}.sbatch"
        start_pos.append((start, end))
        jobid += 1
        # create output directory recursively
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write(sbatch_content_new)
        print(f"Generated {output_filename}")
    
    for i in range(nposterior):
        singer_merge_posterior = ""
        for j in range(len(start_pos)):
            start, _ = start_pos[j]
            singer_merge_posterior += f"{singer_outpref}_{j}_nodes_{i}.txt\t{singer_outpref}_{j}_branches_{i}.txt\t{singer_outpref}_{j}_muts_{i}.txt\t{start}\n"
        with open(f"{output_prefix}_posterior_{i}.txt", 'w') as f:
            f.write(singer_merge_posterior)
        print(f"Generated {output_prefix}_posterior_{i}.txt")

    singer_index = ""
    for i in range(len(start_pos)):
        start, _ = start_pos[i]
        singer_index += f"{start}\t{i}\n"
    with open(f"{output_prefix}_index.txt", 'w') as f:
        f.write(singer_index)


if __name__  == "__main__":
    main()
