import numpy as np
import sys
import pandas as pd
import os
import argparse

def parse_snpinfo_anc(snpinfo):
    infile=open(str(snpinfo))
    lines=infile.readlines()
    infile.close()
    outdict = dict({})
    for i in range(1, len(lines)):
        s=lines[i].strip('\n').strip('\t').split('\t')
        pos = int(s[1])
        ref = str(s[2])
        alt = str(s[3])
        instrict = True if ((int(s[-2]) == 1) & (int(s[-1]) == 1)) else False
        outdict["_".join([str(pos), ref, alt])] = instrict
    return outdict

def merge_df_count_branch_mut(countmerge, snpinfo, classify_rule = 1):
    if countmerge.endswith('txt'):
        data = pd.read_csv(countmerge, sep="\t")
    elif countmerge.endswith('csv'):
        data = pd.read_csv(countmerge)
    else:
        raise ValueError("Unsupported file format for countmerge. Please provide a .txt or .csv file.")
    data = data.sort_values(by=['chromosome', 'start']).reset_index(drop=True)
    cur_chr = data.loc[0, 'chromosome']
    print("Processing chromosome:", cur_chr)
    snpinfo_file = str(snpinfo) + '.' + cur_chr + '.txt'
    snpinfo_dict = parse_snpinfo_anc(snpinfo_file)
    nd00_aff = []
    nd10_aff = []
    nd01_aff = []
    nd11_aff = []
    yri_aff = []
    tot_aff = []
    for i in data.index:
        chr = data.loc[i, 'chromosome']
        if chr != cur_chr:
            cur_chr = chr
            print("Processing chromosome:", cur_chr)
            snpinfo_file = str(snpinfo) + '.' + cur_chr + '.txt'
            snpinfo_dict = parse_snpinfo_anc(snpinfo_file)
        nd00 = 0
        nd10 = 0
        nd01 = 0
        nd11 = 0
        tot = 0
        nyri = 0
        snps = data.loc[i, 'dsnps'].split(',')
        mks = data.loc[i, 'dsnps_marks'].split(',')
        ons = data.loc[i, 'branch_mark'].split(',')
        yri = str(data.loc[i, 'DAF_YRI']).split(',')
        new_mks = ""
        for j in range(len(mks)):
            if ons[j] == 'on' and mks[j] in ['ND00_strict', 'ND10_strict', 'ND01_strict', 'ND11_strict'] and snpinfo_dict[snps[j]]:
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
            if mks[j] in ['ND00_strict', 'ND10_strict', 'ND01_strict', 'ND11_strict'] and not snpinfo_dict[snps[j]]:
                if mks[j] == 'ND00_strict':
                    new_mks += 'ND00'
                elif mks[j] == 'ND10_strict':
                    new_mks += 'ND10'
                elif mks[j] == 'ND01_strict':
                    new_mks += 'ND01'
                elif mks[j] == 'ND11_strict':
                    new_mks += 'ND11'
            else:
                new_mks += mks[j]
            new_mks += ','
        data.loc[i, 'dsnps_marks'] = new_mks.strip(',')
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
    data['nd00_prop'] = nd00_aff / tot_aff
    data['nd10_prop'] = nd10_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nd01_prop'] = nd01_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nYRI'] = yri_aff / tot_aff
    data['assign_label'] = "NONE"
    if classify_rule == 1:
        data.loc[(data['nd10_prop'] > 0.2) & (data['nd10_b'] > data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
        data.loc[(data['nd01_prop'] > 0.2) & (data['nd10_b'] < data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
        data.loc[(data['nd00_prop'] > 0.8) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "Ghost"
    elif classify_rule == 2:
        data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['nd10_b'] > data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
        data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['nd10_b'] < data['nd01_b']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
        data.loc[(data['nd00_prop'] > 0.8) & (data['nYRI'] > 0.1) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "Ghost"
    return data


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Filter annotated tables from TRACE with manifesto filter.")
    
    # Add arguments with prefixes
    parser.add_argument("--input", required=True, help="Input table file (.count_merge.txt file)")
    parser.add_argument("--snpinfo-pref", required=True, help="prefix to snpinfo file with manifesto info appended (parts before .[chrom].txt)")
    parser.add_argument("--output", required=True, help="Output file for the filtered table")
    parser.add_argument("--classify-group", type=int, choices=[1, 2], default=1, help="classification rule group for final annotation, 1 for more stringent classification and 2 for more relaxed classification")
    # parser.add_argument("--npz-pref", required=True, help="prefix for summarized npz files)

    args = parser.parse_args()
    countmerge = args.input
    output = args.output
    classify_rule = args.classify_group
    snpinfo = args.snpinfo_pref
    # xss_pref = args.npz_pref
    

    data = merge_df_count_branch_mut(countmerge, snpinfo, classify_rule)
    data.to_csv(output, sep="\t", index=False)
    
    ## code for annotating npz files for plotting purpose
    # for i in range(1, 23):
    #     print("Processing chromosome:", i)
    #     d = np.load(xss_pref + f"/chr{i}/singerave_t15000_{pop}_DEN.new.npz")
    #     states_shape = d["states"].shape
    #     treespan_phy = d["treespan_phy"]
    #     window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    #     nea_states = np.zeros(states_shape, dtype=d["states"].dtype)
    #     den_states = np.zeros(states_shape, dtype=d["states"].dtype)
    #     ghost_states = np.zeros(states_shape, dtype=d["states"].dtype)
    #     chrom_df = data[data['chromosome'] == f'chr{i}']
    #     ind_ids = chrom_df['ID'].values - np.min(data['ID'].values)
    #     starts = (chrom_df['start'].values / window_size).astype(int)
    #     ends = (chrom_df['end'].values / window_size).astype(int)
    #     labels = chrom_df['assign_label'].values
    #     nea_mask = (labels == "NEA")
    #     den_mask = (labels == "DEN")
    #     ghost_mask = (labels == unrec_label)
    #     for idx in np.where(nea_mask)[0]:
    #         ind_id = ind_ids[idx]
    #         start_idx = starts[idx]
    #         end_idx = ends[idx]
    #         nea_states[ind_id, start_idx:end_idx] = 1
    #     for idx in np.where(den_mask)[0]:
    #         ind_id = ind_ids[idx]
    #         start_idx = starts[idx]
    #         end_idx = ends[idx]
    #         den_states[ind_id, start_idx:end_idx] = 1
    #     for idx in np.where(ghost_mask)[0]:
    #         ind_id = ind_ids[idx]
    #         start_idx = starts[idx]
    #         end_idx = ends[idx]
    #         ghost_states[ind_id, start_idx:end_idx] = 1
    #     np.savez(xss_pref + f"/chr{i}/singerave_t15000_{pop}_NEA.manifesto.npz", states=nea_states, treespan_phy=treespan_phy)
    #     np.savez(xss_pref + f"/chr{i}/singerave_t15000_{pop}_DEN.manifesto.npz", states=den_states, treespan_phy=treespan_phy)
    #     np.savez(xss_pref + f"/chr{i}/singerave_t15000_{pop}_{unrec_label}.manifesto.npz", states=ghost_states, treespan_phy=treespan_phy)


if __name__ == "__main__":
    main()