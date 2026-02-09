import polars as pl
from intervaltree import IntervalTree, Interval
from tqdm import tqdm
import numpy as np


def detect_enrichment_depletion(
    segments_df,
    chrom_size_df=None,
    window_size=1000,
    quantile=0.01,
    autosome_size=2875001522,
):
    """Estimate peak regions based on a chromosome-wide scan."""
    # Determine the average frequency per-individual ...
    for c in ["chromosome", "start", "end", "samplename", "length(bp)"]:
        assert c in segments_df.columns
    assert chrom_size_df is not None
    assert window_size > 0
    assert autosome_size > 0
    assert (quantile > 0) and (quantile < 1)
    peak_list = []
    freq_list = []
    n = segments_df["samplename"].unique().to_numpy().size
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
            peak_list.append((f"chr{i}", s, e + 1, len(t[s:e]) / n))
            freq_list.append(len(t[s:e]) / n)

    freq_list = np.array(freq_list)
    min_threshold, max_threshold = np.quantile(freq_list, quantile), np.quantile(
        freq_list, 1 - quantile
    )
    enriched_flattened = [
        (c, s, e, i) for (c, s, e, i) in tqdm(peak_list) if i >= max_threshold
    ]
    depleted_flattened = [
        (c, s, e, i / n) for (c, s, e, i) in tqdm(peak_list) if i <= min_threshold
    ]
    enriched_df = pl.DataFrame(
        {
            "chrom": [x[0] for x in enriched_flattened],
            "start": [x[1] for x in enriched_flattened],
            "end": [x[2] for x in enriched_flattened],
            "freq": [x[3] for x in enriched_flattened],
            "nhaps": n,
        }
    )
    depleted_df = pl.DataFrame(
        {
            "chrom": [x[0] for x in depleted_flattened],
            "start": [x[1] for x in depleted_flattened],
            "end": [x[2] for x in depleted_flattened],
            "freq": [x[3] for x in depleted_flattened],
            "nhaps": n,
        }
    )
    return enriched_df, depleted_df


def concatenate_intervals(windows_df, enrichment=True):
    """Look for larger regions which have effectively no ghost ancestry."""
    assert "chrom" in windows_df.columns
    uniq_chrom = windows_df["chrom"].unique().to_numpy()
    peak_list = []
    for c in uniq_chrom:
        chrom_specific_df = windows_df.filter(pl.col("chrom") == c)
        interval_list = [
            Interval(s, e, f)
            for (s, e, f) in zip(
                chrom_specific_df["start"].to_numpy(),
                chrom_specific_df["end"].to_numpy(),
                chrom_specific_df["freq"].to_numpy(),
            )
        ]
        t2 = IntervalTree(interval_list)
        if enrichment:
            t2.merge_overlaps(data_reducer=lambda x, y: np.minimum(x, y))
        else:
            t2.merge_overlaps(data_reducer=lambda x, y: np.maximum(x, y))
        for iv in t2:
            # Need some estimate of the frequency of the segments across samples
            peak_list.append((c, iv.begin, iv.end, iv.data))
    if enrichment:
        peak_df = pl.DataFrame(
            {
                "chrom": [x[0] for x in peak_list],
                "start": [x[1] for x in peak_list],
                "end": [x[2] for x in peak_list],
                "min_freq": [x[3] for x in peak_list],
                "nhaps": windows_df["nhaps"].max(),
            }
        )
    else:
        peak_df = pl.DataFrame(
            {
                "chrom": [x[0] for x in peak_list],
                "start": [x[1] for x in peak_list],
                "end": [x[2] for x in peak_list],
                "max_freq": [x[3] for x in peak_list],
                "nhaps": windows_df["nhaps"].max(),
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
    raw_enrichments_df, raw_depletion_df = detect_enrichment_depletion(
        segments_df=filt_df,
        chrom_size_df=chrom_size_df,
        autosome_size=masked_size,
        quantile=snakemake.params["quantile"],
    )

    enrichments_df = concatenate_intervals(raw_enrichments_df, enrichment=True)
    depletion_df = concatenate_intervals(raw_depletion_df)
    enrichments_df = enrichments_df.with_columns(
        pl.lit(snakemake.wildcards["target_pop"]).alias("target_pop"),
        pl.lit(snakemake.wildcards["population"]).alias("population"),
    )
    depletion_df = depletion_df.with_columns(
        pl.lit(snakemake.wildcards["target_pop"]).alias("target_pop"),
        pl.lit(snakemake.wildcards["population"]).alias("population"),
    )
    # Write out all of the files ...
    raw_enrichments_df.write_csv(snakemake.output["raw_enrichments"], separator="\t")
    raw_depletion_df.write_csv(snakemake.output["raw_depletions"], separator="\t")
    enrichments_df.write_csv(snakemake.output["enrichments"], separator="\t")
    depletion_df.write_csv(snakemake.output["depletions"], separator="\t")
