import numpy as np
import sys
from workflow.scripts.utils import Analysis_workflow_utils, SUMMARIZE
import pandas as pd
import os

def merge_df_count_branch_mut(count, count_t1t2, output, pop="AFR", unrec_label = "Ghost"):
    data = pd.read_csv(count_t1t2, sep="\s+")
    dff = pd.read_csv(count, sep="\s+")
    if pop == "AFR":
        data = data.merge(dff, on=dff.columns[0:11].tolist(), how='left')
    elif pop == "EUR" or pop == "OCN" or pop == "EAS" or pop == "SEA" or pop == "SAS":
        data = data.merge(dff, on=dff.columns[0:16].tolist(), how='left')
    else:
        raise ValueError("Unsupported population type. Please use AFR, EUR, OCN, EAS, SAS or SEA.")
    nd00_aff = []
    nd10_aff = []
    nd01_aff = []
    nd11_aff = []
    yri_aff = []
    tot_aff = []
    for i in data.index:
        nd00 = 0
        nd10 = 0
        nd01 = 0
        nd11 = 0
        tot = 0
        nyri = 0
        mks = data.loc[i, 'dsnps_marks'].split(',')
        ons = data.loc[i, 'branch_mark'].split(',')
        yri = str(data.loc[i, 'DAF_YRI']).split(',')
        for j in range(len(mks)):
            if ons[j] == 'on' and mks[j] in ['ND00_strict', 'ND10_strict', 'ND01_strict', 'ND11_strict']:
                tot += 1
                if mks[j] == 'ND00_strict':
                    nd00 += 1
                elif mks[j] == 'ND10_strict':
                    nd10 += 1
                elif mks[j] == 'ND01_strict':
                    nd01 += 1
                elif mks[j] == 'ND11_strict':
                    nd11 += 1
                if float(yri[j]) > 0:
                    nyri += 1
        if tot == 0:
            nd00_aff.append(np.nan)
            nd10_aff.append(np.nan)
            nd01_aff.append(np.nan)
            nd11_aff.append(np.nan)
            tot_aff.append(np.nan)
            yri_aff.append(np.nan)
        else:
            nd00_aff.append(nd00)
            nd10_aff.append(nd10)
            nd01_aff.append(nd01)
            nd11_aff.append(nd11)
            tot_aff.append(tot)
            yri_aff.append(nyri)
    nd00_aff = np.array(nd00_aff)
    nd10_aff = np.array(nd10_aff)
    nd01_aff = np.array(nd01_aff)
    nd11_aff = np.array(nd11_aff)
    tot_aff = np.array(tot_aff)
    yri_aff = np.array(yri_aff)
    data['nd00_b'] = nd00_aff
    data['nd10_b'] = nd10_aff
    data['nd01_b'] = nd01_aff
    data['nd11_b'] = nd11_aff
    data['tot_b'] = tot_aff
    data['yri_b'] = yri_aff
    data['nea_prop'] = (nd10_aff + nd11_aff) / tot_aff
    data['den_prop'] = (nd01_aff + nd11_aff) / tot_aff
    data['nd00_prop'] = nd00_aff / tot_aff
    data['nd11_prop'] = nd11_aff / tot_aff
    data['nd10_prop'] = nd10_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nd01_prop'] = nd01_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nYRI'] = yri_aff / tot_aff
    data['assign_label'] = "NONE"
    data.loc[(data['nd10_prop'] > 0.2) & (data['nd10_b'] > data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
    data.loc[(data['nd01_prop'] > 0.2) & (data['nd10_b'] < data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
    data.loc[(data['nd00_prop'] > 0.8) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = unrec_label
    data.to_csv(output, sep="\t", index=False)

ind = int(sys.argv[1])
summarypref = sys.argv[2]
filepref = sys.argv[3]
chrom = sys.argv[4]
unrec_label = sys.argv[5] if len(sys.argv) > 5 else "Ghost"

df = pd.read_csv("/global/scratch/users/zhangyulin9806/github/ghost_admixture_hmm/results/realdata/1000g_hg38_2022/GhostHMM/LWK/treeID.txt", sep="\t")
hmmixpath = "results/realdata/1000g_hg38_2022/hmmix/02decode/"
inds = [ind]
# hmmixfiles = [f"{hmmixpath}{df[df['ID']==inds[i]]['Name'].values[0]}.txt" for i in range(len(inds))]
# ibdmixeur = f"results/realdata/1000g_hg38_2022/ibdmix/ibdmix/EAS/{chrom}.txt"
ibdmixafr = f"results/realdata/1000g_hg38_2022/ibdmix/ibdmix/AFR/{chrom}.txt"
bscore="/global/scratch/users/zhangyulin9806/github/ArchaicMutRate/helper/hg38_bmap.txt"
recombrate= "/global/scratch/users/zhangyulin9806/github/ArchaicMutRate/helper/hg38_recombination_map/genetic_map_autosome.bed"
individualID = [df[df['ID']==inds[i]]['SampleID'].values[0] for i in range(len(inds))]
xssfile = f"{summarypref}{filepref}{inds[0]}.{chrom}.xss.npz"

for i in range(len(inds)):
    prefix = summarypref + filepref + str(inds[i]) + ".summary"
    if df[df['ID']==inds[i]]['SuperGroup'].values[0] == "EAS":
        prefixn = prefix + ".hmmix"
        SUMMARIZE().append_hmmix_info(hmmixfiles[i], prefix + ".txt", prefixn, inference = "hmmix", individualID = individualID[i])
        prefix = prefixn
        prefixn = prefix + ".ibdmix"
        SUMMARIZE().append_hmmix_info(ibdmixeur, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
        os.remove(prefix + ".txt")
    else:
        prefixn = prefix + ".ibdmix"
        SUMMARIZE().append_hmmix_info(ibdmixafr, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
    

snpinfo = "results/realdata/1000g_hg38_2022/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.lwkonly.outgroup.strictmask"
mutage = "results/realdata/1000g_hg38_2022/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.lwkonly.outgroup.strictmask.mutage"
bcfpref = "results/realdata/1000g_hg38_2022/vcf_files/1000g_hg38_chr"
for i in range(len(inds)):
    hap = "left" if inds[i]%2 == 0 else "right"
    if df[df['ID']==inds[i]]['SuperGroup'].values[0] == "EAS":
        SUMMARIZE().final_ind_count(
            samplename = individualID[i], 
            summary = summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt", 
            hap = hap, 
            snpinfo = snpinfo, 
            bcfpref = bcfpref, 
            outpref = summarypref + filepref + str(inds[i]) + ".count"
        )
        SUMMARIZE().append_t1_t2(
            npzpref = summarypref + filepref + str(inds[i]) + f".{chrom}.xss",
            summary = summarypref + filepref + str(inds[i]) + ".count_DAF.txt", 
            snpinfo = mutage + f".{chrom}.txt", 
            func = "median",
            outpref = summarypref + filepref + str(inds[i]) + ".count_t1t2",
        )
        SUMMARIZE().append_bscore_recombrate(
            popcountfile = summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt", 
            bscore = bscore,
            recombrate = recombrate, 
        )
    else:
        SUMMARIZE().final_ind_count(
            samplename = individualID[i], 
            summary = summarypref + filepref + str(inds[i]) + ".summary.ibdmix.txt", 
            hap = hap, 
            snpinfo = snpinfo, 
            bcfpref = bcfpref, 
            outpref = summarypref + filepref + str(inds[i]) + ".count"
        )
        SUMMARIZE().append_t1_t2(
            npzpref = summarypref + filepref + str(inds[i]) + f".{chrom}.xss",
            summary = summarypref + filepref + str(inds[i]) + ".count_DAF.txt", 
            snpinfo = mutage + f".{chrom}.txt",
            func = "median",
            outpref = summarypref + filepref + str(inds[i]) + ".count_t1t2",
        )
        SUMMARIZE().append_bscore_recombrate(
            popcountfile = summarypref + filepref + str(inds[i]) + ".summary.ibdmix.txt", 
            bscore = bscore,
            recombrate = recombrate, 
        )
    os.remove(summarypref + filepref + str(inds[i]) + ".count_DAF.txt")
    merge_df_count_branch_mut(
        count = summarypref + filepref + str(inds[i]) + ".count.txt", 
        count_t1t2 = summarypref + filepref + str(inds[i]) + ".count_t1t2.txt", 
        output = summarypref + filepref + str(inds[i]) + ".count_merge.txt",
        pop = df[df['ID']==inds[i]]['SuperGroup'].values[0],
        unrec_label = unrec_label
    )
    os.remove(summarypref + filepref + str(inds[i]) + ".count.txt")
    os.remove(summarypref + filepref + str(inds[i]) + ".count_t1t2.txt")
    with np.load(xssfile) as d:
        data = {k: d[k] for k in d.files} 
    states = data["states"]
    treespan_phy = data["treespan_phy"]
    window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    nea_states = np.zeros(shape=states.shape, dtype=states.dtype)
    den_states = np.zeros(shape=states.shape, dtype=states.dtype)
    ghost_states = np.zeros(shape=states.shape, dtype=states.dtype)
    df = pd.read_csv(summarypref + filepref + str(inds[i]) + ".count_merge.txt", sep="\t")
    for j in range(len(df)):
        if df.loc[j, 'assign_label'] == "NEA":
            nea_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
        elif df.loc[j, 'assign_label'] == "DEN":
            den_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
        elif df.loc[j, 'assign_label'] == unrec_label:
            ghost_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
    data["nea_states"] = nea_states
    data["den_states"] = den_states
    data["ghost_states"] = ghost_states
    np.savez_compressed(
        summarypref + filepref + str(inds[i]) + f".{chrom}.xss.npz", **data
    )

