library(data.table, quietly=TRUE)
library(GenomicRanges, quietly=TRUE)
library(rGREAT, quietly=TRUE)


# if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# BiocManager::install("rGREAT")

## -- Set command line arguments -- ##
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    print("run using enrichment_great.R <bedfile> <outfix>!")
    stop("Requires command line arguments!")
}
set.seed(42)

bed_fp <- args[1]
outfix <- args[2]

## -- Read in a bed file into a GRanges object -- ##
bed_df <- read.table(bed_fp)
colnames(bed_df) <- c("chrom", "start", "end")
gr <-  makeGRangesFromDataFrame(bed_df)

## -- Run GREAT Analysis -- ##
job = submitGreatJob(gr, genome = "hg38")
tbl = getEnrichmentTables(job)

## -- Write output -- ##
go_cellular_component_df <- tbl$`GO Cellular Component`
write.table(go_cellular_component_df, file = paste(outfix, ".hg38.cell_component.csv", sep=""), sep = ",", row.names=FALSE)

go_bio_process_df <- tbl$`GO Biological Process`
write.table(go_cellular_component_df, file = paste(outfix, ".hg38.bioprocess.csv", sep=""), sep = ",", row.names=FALSE)

go_mol_function_df <- tbl$`GO Molecular Function`
write.table(go_cellular_component_df, file = paste(outfix, ".hg38.go_molecular.csv", sep=""), sep = ",", row.names=FALSE)