# TRACE Results Replication

This repository is meant to contain all of the 

## Dependencies and Installation

We highly suggest reproducing our results using a conda environment which contains all of the relevant dependencies (including several custom ones).

```
mamba env create -f env.yaml
conda activate trace_paper
```

We also recommend downloading the `trace_zenodo` subdirectory direclty into the top-level directory for easier reproduction of all of the results. 

## Reproducing Analyses

### Main Manuscript Figures

To reproduce all of the main figures within the manuscript, you can execute all of the cells in `plot_main.ipynb`. 

**NOTE: executing this directory creates ~125 GB in accessory files, so we recommend you do this on a computing environment with enough space.**

### Main Manuscript Analyses

There are two additional subdirectories pertaining to data analysis and reproduceability: 

1. `realdata/` - contains workflows for analysis of real data within the manuscript
2. `trace_benchmark` contains the primary workflow for evaluating TRACE on simulations

## Contact

* Yulin Zhang (zhangyulin9806@berkeley.edu)
* Arjun Biddanda (abiddan1@jhu.edu)
