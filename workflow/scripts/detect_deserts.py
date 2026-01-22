import polars as pl
from intervaltree import IntervalTree, Interval
from tqdm import tqdm
import numpy as np


def determine_deserts(
    segments_df,
    chrom_size_df=None,
    window_size=1000,
    min_len=10e6,
    max_freq=0.001,
    segment_len=50e3,
):
    """Estimate peak regions based on a chromosome-wide scan."""
    # Determine the average frequency per-individual ...
    for c in ["chromosome", "start", "end", "samplename", "length(bp)"]:
        assert c in segments_df.columns
    assert chrom_size_df is not None
    assert window_size > 0
    assert min_len > 0
    assert max_freq > 0
    n = segments_df["samplename"].unique().to_numpy().size
    desert_list = []
    for i in tqdm(range(1, 23)):
        # Create a chromosome-specific interval tree
        chrom_len = chrom_size_df.filter(pl.col("chrom") == f"chr{i}")[
            "size"
        ].to_numpy()[0]
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
        tot_list = []
        pts = np.arange(0, int(chrom_len), window_size)
        for s in pts:
            # Get the number of segments intersecting here
            tot_len = n * min_len
            n_segments = len(t[s : int(s + min_len)])
            min_agg_len = 50e3 * n_segments
            # agg_len = np.sum([iv.end - iv.begin for iv in t[s : int(s + min_len)]])
            if (min_agg_len / tot_len) <= max_freq:
                agg_len = np.sum([iv.end - iv.begin for iv in t[s : int(s + min_len)]])
                if (agg_len / tot_len) <= max_freq:
                    tot_list.append((f"chr{i}", s, s + min_len, agg_len / tot_len))
        t2 = IntervalTree([Interval(x[1], x[2], data=x[3]) for x in tot_list])
        t2.merge_overlaps(data_reducer=lambda x, y: np.maximum(x, y))
        for iv in t2:
            desert_list.append((f"chr{i}", iv.begin, iv.end, iv.data))
    desert_df = pl.DataFrame(
        {
            "chrom": [x[0] for x in desert_list],
            "start": [int(x[1]) for x in desert_list],
            "end": [int(x[2]) for x in desert_list],
            "nfreq": [x[3] for x in desert_list],
            "nhaps": n,
        }
    )
    return desert_df


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
    deserts_df = determine_deserts(
        segments_df=filt_df,
        chrom_size_df=chrom_size_df,
        window_size=1000000,
        min_len=snakemake.params["min_length"],
        max_freq=snakemake.params["max_freq"],
    )
    deserts_df = deserts_df.with_columns(
        pl.lit(snakemake.wildcards["target_pop"]).alias("target_pop")
    )
    deserts_df = deserts_df.with_columns(
        pl.lit(snakemake.wildcards["population"]).alias("population")
    )
    deserts_df.write_csv(
        snakemake.output["deserts"], separator="\t", include_header=False
    )
