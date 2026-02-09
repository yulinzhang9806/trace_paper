import polars as pl
from intervaltree import IntervalTree, Interval
from tqdm import tqdm
import numpy as np


def determine_peaks(
    segments_df, chrom_size_df=None, window_size=1000, autosome_size=2875001522, sigma=2
):
    """Estimate peak regions based on a chromosome-wide scan."""
    # Determine the average frequency per-individual ...
    for c in ["chromosome", "start", "end", "samplename", "length(bp)"]:
        assert c in segments_df.columns
    assert chrom_size_df is not None
    assert window_size > 0
    assert autosome_size > 0
    assert sigma > 0
    # Estimate the total length of archaic segments across the autosomes
    # tot_lengths = (
    #     segments_df.group_by("samplename")
    #     .agg(pl.col("length(bp)").sum().alias("tot_length"))["tot_length"]
    #     .to_numpy()
    #     / autosome_size
    # )
    # mu, s_sigma = np.mean(tot_lengths), np.std(tot_lengths)
    n = segments_df["samplename"].unique().to_numpy().size
    freqs = []
    for i in tqdm(range(1, 23)):
        chrom_len = chrom_size_df.filter(pl.col("chrom") == f"chr{i}")[
            "size"
        ].to_numpy()[0]
        # Create a chromosome-specific interval tree
        chrom_specific_df = segments_df.filter(pl.col("chromosome") == f"chr{i}")
        interval_list = [
            Interval(s, e, samp_id)
            for (s, e, samp_id) in zip(
                chrom_specific_df["start"].to_numpy(),
                chrom_specific_df["end"].to_numpy(),
                chrom_specific_df["samplename"].to_numpy(),
            )
        ]
        t = IntervalTree(interval_list)
        pts = np.arange(0, int(chrom_len), window_size)
        for s, e in zip(pts[:-1], pts[1:]):
            freqs.append(len(t[s:e]) / n)
    # This is now defining the estimates of mu, sigma...
    freqs = np.array(freqs)
    mu = np.mean(freqs[freqs > 0])
    s_sigma = np.std(freqs[freqs > 0])
    # Now we iterate through the chromosomes and create interval trees
    peak_list = []
    for i in tqdm(range(1, 23)):
        tot_list = []
        chrom_len = chrom_size_df.filter(pl.col("chrom") == f"chr{i}")[
            "size"
        ].to_numpy()[0]
        # Create a chromosome-specific interval tree
        chrom_specific_df = segments_df.filter(pl.col("chromosome") == f"chr{i}")
        interval_list = [
            Interval(s, e, samp_id)
            for (s, e, samp_id) in zip(
                chrom_specific_df["start"].to_numpy(),
                chrom_specific_df["end"].to_numpy(),
                chrom_specific_df["samplename"].to_numpy(),
            )
        ]
        t = IntervalTree(interval_list)
        pts = np.arange(0, int(chrom_len), window_size)
        for s, e in zip(pts[:-1], pts[1:]):
            # this checks if we have a higher peak across all samples
            if (len(t[s:e]) / n) >= (mu + sigma * s_sigma):
                tot_list.append((f"chr{i}", s, e + 1, len(t[s:e]) / n))
        # Create a merged set here ...
        t2 = IntervalTree([Interval(x[1], x[2], data=x[3]) for x in tot_list])
        t2.merge_overlaps(data_reducer=lambda x, y: np.minimum(x, y))
        for iv in t2:
            # Need some estimate of the frequency of the segments across samples
            peak_list.append((f"chr{i}", iv.begin, iv.end, iv.data))
    # make the tot_list into a dataframe (potentially merging nearby regions)
    peak_df = pl.DataFrame(
        {
            "chrom": [x[0] for x in peak_list],
            "start": [x[1] for x in peak_list],
            "end": [x[2] for x in peak_list],
            "nfreq": [x[3] for x in peak_list],
            "nhaps": n,
            "mu": mu,
            "s_sigma": s_sigma,
            "genome_length": autosome_size,
        }
    )
    return peak_df


if __name__ == "__main__":

    df = pl.read_csv(
        snakemake.input["filtered_segments"],
        separator="\t",
        columns=[
            "assign_label",
            "chromosome",
            "start",
            "end",
            "nderived_strict",
            "ND00_strict",
            "ND10_strict",
            "ND01_strict",
            "ND11_strict",
            "samplename",
            "length(bp)",
        ],
    )
    for c in [
        "assign_label",
        "chromosome",
        "start",
        "end",
        "nderived_strict",
        "ND00_strict",
        "ND10_strict",
        "ND01_strict",
        "ND11_strict",
    ]:
        assert c in df.columns
    filt_df = (
        df.filter(pl.col("assign_label") == snakemake.params["target_pop"])
        .filter(pl.col("nderived_strict") >= snakemake.params["n_derived_strict"])
        .filter(
            (
                pl.col("ND00_strict")
                + pl.col("ND10_strict")
                + pl.col("ND01_strict")
                + pl.col("ND11_strict")
            )
            >= snakemake.params["nd_filter"]
        )
    )
    print(filt_df.shape, snakemake.params["target_pop"])
    # Determine total autosomal length
    chrom_size_df = pl.read_csv(
        snakemake.input["chrom_sizes"], has_header=False, separator="\t"
    )
    chrom_size_df.columns = ["chrom", "size"]
    autosome_size = (
        chrom_size_df.filter(pl.col("chrom").is_in([f"chr{i}" for i in range(1, 23)]))[
            "size"
        ]
        .to_numpy()
        .sum()
    )
    # Read in accessibility mask
    accessibility_mask_df = pl.read_csv(
        snakemake.input["bed_mask"], has_header=False, separator="\t"
    )
    accessibility_mask_df.columns = ["chrom", "start", "end"]
    accessibility_mask_df = accessibility_mask_df.with_columns(
        (pl.col("end") - pl.col("start")).alias("size")
    )
    masked_size = (
        accessibility_mask_df.filter(
            pl.col("chrom").is_in([f"chr{i}" for i in range(1, 23)])
        )["size"]
        .to_numpy()
        .sum()
    )
    peak_df = determine_peaks(
        segments_df=filt_df,
        chrom_size_df=chrom_size_df,
        autosome_size=masked_size,
        sigma=snakemake.params["sigma"],
    )
    peak_df = peak_df.with_columns(pl.lit(snakemake.wildcards["target_pop"]))
    peak_df.write_csv(snakemake.output["bed"], separator="\t", include_header=False)
