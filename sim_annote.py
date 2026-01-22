import demes
import msprime
import tszip
import yaml
from arg_hmm.utils import *
import tskit
import numpy as np
import pandas as pd
import pybedtools
import sys
import os

def append_hmmix_info(hmmixfile, summaryfile, outpref, inference = "hmmix", individualID = None):
    """
    Append the HMMIX info to the summary file.
    """
    try:
        hmmix = pd.read_csv(hmmixfile, sep="\s+")
        summary = pd.read_csv(summaryfile, sep="\s+", dtype={"chromosome": str, "start": int, "end": int})
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    if inference == "hmmix":
        hmmix['end'] = hmmix['end'] + 1000
        hmmix = hmmix[hmmix["state"] == "Archaic"]
        hmmix['archaic'] = "Ambiguous"
        hmmix.loc[(hmmix["tsk_200"] > hmmix['tsk_201']) & (hmmix['state'] == 'Archaic'), 'archaic'] = 'Neanderthal'
        hmmix.loc[(hmmix["tsk_200"] < hmmix['tsk_201']) & (hmmix['state'] == 'Archaic'), 'archaic'] = 'Denisova'
        hmmix['archaic'] = hmmix["mean_prob"].astype(str) + "_" + hmmix['archaic']
        out = ("\t").join(summary.columns) + "\thmmix_start\thmmix_end\thmmix_overlap_length(bp)\tmean_pp\thmmix_assign\n"
    elif inference == "ibdmix":
        if individualID is None:
            print("Please provide the individual ID for ibdmix inference.")
            sys.exit(1)
        hmmix = hmmix[hmmix["ID"] == individualID]
        hmmix['archaic'] = (hmmix['end'] - hmmix['start']).astype(str) + "_" + hmmix['archaic'].astype(str)
        out = ("\t").join(summary.columns) + "\tibdmix_start\tibdmix_end\tibdmix_overlap_length(bp)\tslod\tibdmix_assign\n"
    elif inference == "truth":
        hmmix = pd.read_csv(hmmixfile, sep="\s+")
        hmmix.columns = ["chrom", "start", "end", "ID", "archaic"]
        hmmix = hmmix[hmmix["ID"] == individualID]
        out = ("\t").join(summary.columns) + "\ttruth_start\ttruth_end\ttruth_overlap_length(bp)\ttruth_assign\n"
    s_segs = pybedtools.BedTool(summary[["chromosome", "start", "end"]].to_csv(sep="\t", index=False, header=False), from_string=True)
    h_segs = pybedtools.BedTool(hmmix[["chrom", "start", "end", "archaic"]].to_csv(sep="\t", index=False, header=False), from_string=True)
    overlap = s_segs.intersect(h_segs, wao=True)
    ovl = {}
    for i in range(len(overlap)):
        ol = int(overlap[i][-1])
        if ol > 0:
            key = f"{overlap[i][0]}_{overlap[i][1]}_{overlap[i][2]}"
            if key not in ovl:
                ovl[key] = []
            ovl[key].append([f"{overlap[i][4]}_{overlap[i][5]}", ol, overlap[i][6]])
    for s in range(len(summary)):
        chrom = summary['chromosome'][s]
        start = summary['start'][s]
        end = summary['end'][s]
        out += ("\t").join(summary.iloc[s].astype(str).tolist())
        if f"{chrom}_{start}_{end}" in ovl:
            ss = ovl[f"{chrom}_{start}_{end}"]
            if len(ss) > 1:
                hmm_start = ""
                hmm_end = ""
                hmm_ol = 0
                hmm_pp = []
                hmm_assign = ""
                for i in range(len(ss)):
                    ss_s1 = ss[i][0].split("_")
                    hmm_start += f"{ss_s1[0]},"
                    hmm_end += f"{ss_s1[1]},"
                    hmm_ol += ss[i][1]
                    ss_s2 = ss[i][2].split("_")
                    if not inference == "truth":
                        hmm_pp.append(float(ss_s1[0]))
                        hmm_assign += f"{ss_s2[1]},"
                    else:
                        hmm_assign += f"{ss_s2[0]},"
                if inference == "truth":
                    out += f"\t{hmm_start[:-1]}\t{hmm_end[:-1]}\t{hmm_ol}\t{hmm_assign[:-1]}\n"
                else:
                    out += f"\t{hmm_start[:-1]}\t{hmm_end[:-1]}\t{hmm_ol}\t{np.max(hmm_pp)}\t{hmm_assign[:-1]}\n"
            else:
                ss = ss[0]
                ss_s1 = ss[0].split("_")
                ss_s2 = ss[2].split("_")
                if inference == "truth":
                    out += f"\t{ss_s1[0]}\t{ss_s1[1]}\t{ss[1]}\t{ss_s2[0]}\n"
                else:
                    out += f"\t{ss_s1[0]}\t{ss_s1[1]}\t{ss[1]}\t{ss_s2[0]}\t{ss_s2[1]}\n"
        else:
            if inference == "truth":
                out += "\t-1\t-1\t-1\t-1\n"
            else:
                out += "\t-1\t-1\t-1\t-1\t-1\n"
    with open(f"{outpref}.txt", "w") as f:
        f.write(out)
    return
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
        dnea = (s[4] in ['1','2'])
        dden = (s[5] in ['1','2'])
        freqs = np.array([float(s[-3]), float(s[-2]), float(s[-1])])
        outdict["_".join([str(pos), ref, alt])] = [dnea, dden, freqs]
    return outdict

def count_snps(seg_snps, snp_dict, hap="left"):
    snpcount = {"nsnps":0, "ND00":0, "ND10":0, "ND01":0, "ND11":0}
    dsnps = []
    dsnps_freqs = []
    dsnps_marks = []
    hapdict = {"left":0, "right":1}
    infile=open(seg_snps)
    lines=infile.readlines()
    infile.close()
    for i in range(len(lines)):
        s=lines[i].strip('\n').strip('\t').split('\t')
        pos = "_".join([s[0], s[1], s[2]])
        if not pos in snp_dict:
            print(pos)
            print("Error: SNP not found in snp_dict.")
            sys.exit(1)
        if str(s[3]) in ['./.', './1', './0', '0/.', '1/.']: # skip missing sites
            continue
        genos = str(s[3]).split('|')
        geno = int(genos[hapdict[hap]])
        dnea, dden, freqs = snp_dict[pos]
        if geno > 0:
            snpcount['nsnps'] += 1
            mark = "other"
            if not dnea and not dden :
                snpcount['ND00'] += 1
                mark = "ND00"
            elif dnea and not dden:
                snpcount['ND10'] += 1
                mark = "ND10"
            elif not dnea and dden:
                snpcount['ND01'] += 1
                mark = "ND01"
            elif dnea and dden:
                snpcount['ND11'] += 1
                mark = "ND11"
            dsnps.append(pos)
            dsnps_freqs.append(freqs)
            dsnps_marks.append(mark)
    return snpcount, np.array(dsnps), np.array(dsnps_freqs), np.array(dsnps_marks)

def get_regions_bed(file):
    out = ""
    infile=open(file)
    lines=infile.readlines()
    infile.close() 
    for i in range(len(lines)):
        s = lines[i].strip('\n').strip('\t').split('\t')
        out += s[0] + ':' + s[1] + '-' + s[2]
        if not i == len(lines) - 1:
            out += ','
    return out

def final_ind_count(samplename, summary, hap, snpinfo, bcfpref, outpref):      
    if os.path.exists(str(outpref) + ".txt") and hap != 2:
        os.remove(str(outpref) + ".txt")
    infile=open(summary)
    lines=infile.readlines()
    infile.close()
    out = lines[0].strip('\n') + '\tnderived\tND00\tND10\tND01\tND11\n'
    out1 = lines[0].strip('\n') + "\tdsnps\tdsnps_marks\tDAF_GBR\tDAF_YRI\tDAF_GBRYRI\n"
    if len(lines) >= 2:
        cur_chr = lines[1].strip('\n').split('\t')[0]
        snpfile = str(snpinfo) + '.txt'
        bcffile = str(bcfpref) + '.bcf'
        snp_dict = parse_snpinfo_anc(snpfile)
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            os.system('echo "' + str(s[0]) + "\t" + str(s[1]) + "\t" + str(s[2]) + '" > ' + str(outpref) + str(i) + 'seg.bed')
            reg = get_regions_bed(str(outpref) + str(i) + "seg.bed")
            os.system(
                "bcftools view -s " + str(samplename) + " -v snps -M 2 -m 2 -r " + str(reg) + " " + str(bcffile) + " | bcftools query -f'[%POS\t%REF\t%ALT\t%GT\n]' > "+ str(outpref) + str(i) + "seg_snps"
            )
            snpcount, dsnps, dsnps_freqs, dsnps_marks = count_snps(str(outpref) + str(i) + "seg_snps", snp_dict, hap)
            out += lines[i].strip('\n') + '\t' + str(snpcount['nsnps']) + '\t' + str(snpcount['ND00']) + '\t' 
            out += str(snpcount['ND10']) + '\t' + str(snpcount['ND01']) + '\t' + str(snpcount['ND11']) + "\n"
            dsnps_freqs = np.round(dsnps_freqs, 3)
            if len(dsnps) > 0:
                out1 += lines[i].strip('\n') + '\t' + ','.join(dsnps) + '\t'
                out1 += ','.join(dsnps_marks) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 0]]) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 1]]) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 2]]) + '\n'
            else:
                out1 += lines[i].strip('\n') + '\t' + 'NA\tNA\tNA\tNA\tNA\n'
            os.remove(str(outpref) + str(i) + 'seg.bed')
            os.remove(str(outpref) + str(i) + "seg_snps")
    outfile=open(outpref + '.txt','a')
    outfile.write(out)
    outfile.close()
    outfile=open(outpref + '_DAF.txt','w')
    outfile.write(out1)
    outfile.close()

def append_t1_t2(npzpref, summary, snpinfo, func, outpref):
    mutage = {}
    infile = open(snpinfo)
    lines = infile.readlines()
    infile.close()
    for i in range(1, len(lines)):
        s = lines[i].strip('\n').strip('\t').split('\t')
        pos = int(s[1])
        if not s[-1] in ['nan', 'Not_mapped', 'NA']:
            ag = float(s[-1])
            mutage[pos] = ag
    with np.load(f"{npzpref}.npz") as data:
        t1 = data["t1s"]
        t2 = data["t2s"]
        nleaves = data["nleaves"]
        treespan_phy = data["treespan_phy"]
    windowsize = treespan_phy[0][1] - treespan_phy[0][0]
    if func == "mean":
        func = np.mean
    elif func == "median":
        func = np.median
    else:
        sys.exit(f"Unrecognized function {func}")
    # t1 = func(t1, axis = 0)
    # t2 = func(t2, axis = 0)
    # nleaves = np.nanmean(nleaves, axis = 0)
    infile=open(summary)
    lines=infile.readlines()
    infile.close()
    out = lines[0].strip('\n') + '\tt1s\tt2s\tn_leaves\tmutages\tbranch_mark\n'
    if len(lines) >= 2:
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            st = int(int(s[1]) / windowsize)
            ed = int(int(s[2]) / windowsize)
            t1_val = ",".join(t1[st:ed].astype('str'))
            t2_val = ",".join(t2[st:ed].astype('str'))
            nlv_val = ",".join(nleaves[st:ed].astype('str'))
            out += lines[i].strip('\n') + '\t' + str(t1_val) + '\t' + str(t2_val) + '\t' + str(nlv_val) + '\t'
            muts = s[-5].strip('\n').strip('\t').split(',')
            mks = []
            ags = []
            if len(muts) == 1 and muts[0] == 'NA':
                out += 'NA\tNA\n'
                continue
            for m in muts:
                pos = int(m.split('_')[0])
                if pos in mutage:
                    ags.append(mutage[pos])
                    if mutage[pos] > t1[int(pos / windowsize)] and mutage[pos] < t2[int(pos / windowsize)]:
                        mks.append("on")
                    elif mutage[pos] <= t1[int(pos / windowsize)]:
                        mks.append("below")
                    elif mutage[pos] >= t2[int(pos / windowsize)]:
                        mks.append("above")
                else:
                    ags.append("NA")
                    mks.append("NA")
            out += "\t" + ",".join([str(i) for i in ags]) + "\t" + ",".join(mks) + '\n'
    outfile=open(outpref + '.txt','w')
    outfile.write(out)
    outfile.close()

def filter_tracts(
    indiv_pp,
    treespan,
    treespan_phy,
    pp_cutoff=0.9,
    arc_cutoff=0.5,
    phy_cutoff=5e4,
    l_cutoff=0.05,
    remove_margin=0,
):
    tracts = []
    states = np.zeros(indiv_pp.shape[0])
    i = 0
    while i < indiv_pp.shape[0]:
        if indiv_pp[i] >= pp_cutoff:
            j = i
            temp_pos = []
            temp_pp = []
            while (
                j < len(indiv_pp) and indiv_pp[j] >= arc_cutoff
            ):
                temp_pos.append(j)
                temp_pp.append(indiv_pp[j])
                j += 1
            if (
                np.mean(temp_pp) >= pp_cutoff 
                and treespan[np.max(temp_pos)][1] 
                - treespan[np.min(temp_pos)][0]
                >= l_cutoff
                and treespan_phy[np.max(temp_pos)][1]
                - treespan_phy[np.min(temp_pos)][0]
                >= phy_cutoff
                ):
                start = treespan_phy[temp_pos[0]][0]
                end = treespan_phy[temp_pos[-1]][1]
                tracts.append([start, end, np.mean(temp_pp), end - start, treespan[temp_pos[-1]][1] - treespan[temp_pos[0]][0]])
                states[np.array(temp_pos[remove_margin:(len(temp_pos) - remove_margin)])] = 1
            i = j
        else:
            i += 1
    return tracts, states

def merge_df_count_branch_mut(count, count_t1t2, output, pop="AFR", unrec_label = "Ghost"):
    data = pd.read_csv(count_t1t2, sep="\s+")
    dff = pd.read_csv(count, sep="\s+")
    if unrec_label == "Ghost":
        if pop == "African":
            data = data.merge(dff, on=dff.columns[0:15].tolist(), how='left')
        elif pop == "NonAfrican":
            data = data.merge(dff, on=dff.columns[0:20].tolist(), how='left')
        else:
            raise ValueError("Unsupported population type. Please use African or NonAfrican.")
    elif unrec_label == "SUPER":
        if pop == "NonAfrican":
            data = data.merge(dff, on=dff.columns[0:10].tolist(), how='left')
        else:
            raise ValueError("Unsupported population type for SUPER. Please use NonAfrican.")
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
        nyri = 0
        tot = 0
        mks = str(data.loc[i, 'dsnps_marks']).split(',')
        ons = str(data.loc[i, 'branch_mark']).split(',')
        yri = str(data.loc[i, 'DAF_YRI']).split(',')
        for j in range(len(mks)):
            if ons[j] == 'on' and mks[j] in ['ND00', 'ND10', 'ND01', 'ND11']:
                tot += 1
                if mks[j] == 'ND00':
                    nd00 += 1
                elif mks[j] == 'ND10':
                    nd10 += 1
                elif mks[j] == 'ND01':
                    nd01 += 1
                elif mks[j] == 'ND11':
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
    data['nd10_prop'] = nd10_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nd01_prop'] = nd01_aff / (nd00_aff + nd10_aff + nd01_aff)
    data['nYRI'] = yri_aff / tot_aff
    data['assign_label'] = "NONE"
    data['nea_aff'] = (data['ND10'] + data['ND11']) / (data['ND00'] + data['ND10'] + data['ND01'] + data['ND11'])
    data['den_aff'] = (data['ND01'] + data['ND11']) / (data['ND00'] + data['ND10'] + data['ND01'] + data['ND11'])
    if unrec_label == "SUPER":
        data.loc[(data['nd10_prop'] > 0.2) & (data['nd10_b'] > data['nd01_b']) & (data['nderived'] >= 10) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 0), 'assign_label'] = "NEA"
        data.loc[(data['nd01_prop'] > 0.2) & (data['nd10_b'] < data['nd01_b']) & (data['nderived'] >= 10) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 0), 'assign_label'] = "DEN"
        data.loc[(data['nd10_prop'] <= 0.2) & (data['nd01_prop'] <= 0.2) & (data["nea_aff"] < 0.5) & (data["nYRI"] < 0.5) & (data["tot_b"] >= 10) & (data['nderived'] >= 10) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 0), 'assign_label'] = unrec_label
    elif unrec_label == "Ghost":
        data.loc[(data['nd10_prop'] > 0.2) & (data['nd10_b'] > data['nd01_b']) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
        data.loc[(data['nd01_prop'] > 0.2) & (data['nd10_b'] < data['nd01_b']) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
        data.loc[(data['nd00_prop'] > 0.8) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = unrec_label
        # data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['nd10_b'] > data['nd01_b']) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
        # data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['nd10_b'] < data['nd01_b']) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
        # data.loc[(data['nd00_prop'] > 0.8) & (data['nYRI'] > 0.1) & (data['nderived'] >= 30) & (data[["ND00", "ND10", "ND01", "ND11"]].sum(axis=1) >= 10), 'assign_label'] = unrec_label
    else:
        raise ValueError("Unsupported unrec_label. Please use SUPER or Ghost.")
    data.to_csv(output, sep="\t", index=False)

def count_nd_subtrees(count_merge, output, maf=0.05):
    data = pd.read_csv(count_merge, sep="\t")
    and00 = np.zeros(len(data))
    and10 = np.zeros(len(data))
    and01 = np.zeros(len(data))
    and11 = np.zeros(len(data))
    dnd00 = np.zeros(len(data))
    dnd10 = np.zeros(len(data))
    dnd01 = np.zeros(len(data))
    dnd11 = np.zeros(len(data))
    for i in data.index:
        dmks = str(data.loc[i, 'dsnps_marks']).split(',')
        dyri = str(data.loc[i, 'DAF_YRI']).split(',')
        amks = str(data.loc[i, 'asnps_marks']).split(',')
        ayri = str(data.loc[i, 'aDAF_YRI']).split(',')
        for j in range(len(dmks)):
            if float(dyri[j]) > maf:
                if dmks[j] == 'ND00':
                    dnd00[i] += 1
                elif dmks[j] == 'ND10':
                    dnd10[i] += 1
                elif dmks[j] == 'ND01':
                    dnd01[i] += 1
                elif dmks[j] == 'ND11':
                    dnd11[i] += 1
        for j in range(len(amks)):
            if float(ayri[j]) > maf:
                if amks[j] == 'ND00':
                    and00[i] += 1
                elif amks[j] == 'ND10':
                    and10[i] += 1
                elif amks[j] == 'ND01':
                    and01[i] += 1
                elif amks[j] == 'ND11':
                    and11[i] += 1
    data['and00'] = and00
    data['and10'] = and10
    data['and01'] = and01
    data['and11'] = and11
    data['dnd00'] = dnd00
    data['dnd10'] = dnd10
    data['dnd01'] = dnd01
    data['dnd11'] = dnd11
    data.to_csv(output, sep="\t", index=False)

def seq_den_affinity(snpinfo, summary):
    infile = open(snpinfo)
    lines = infile.readlines()
    infile.close()
    snp_dict = {}
    for i in range(1, len(lines)):
        s = lines[i].strip('\n').strip('\t').split('\t')
        pos = int(s[1])
        dnea = int(s[4])
        dden = int(s[5])
        snp_dict[pos] = [dnea, dden]
    infile = open(summary)
    lines = infile.readlines()
    infile.close()
    out = lines[0].strip('\n') + '\tseq_den_affinity\n'
    for i in range(1, len(lines)):
        s = lines[i].strip('\n').strip('\t').split('\t')
        start = int(s[1])
        end = int(s[2])
        dden = 0
        dnea = 0
        for pos in snp_dict:
            if pos >= start and pos <= end:
                if snp_dict[pos][1] > 0:
                    if snp_dict[pos][0] > 0:
                        dnea += snp_dict[pos][1]
                    dden += 1
        out += lines[i].strip('\n') + '\t' + str(dnea / (2 * dden)) + '\n'
    outfile = open(summary, 'w')
    outfile.write(out)
    outfile.close()

ind = int(sys.argv[1])
s = int(sys.argv[2])
unrec_label = sys.argv[3]
input_model = sys.argv[4]
df = pd.read_csv(f"results/realdata/test_simulation/{input_model}/singer_afr/treeID.txt", sep="\t")
individualID = df[df['ID'] == ind]['SampleID'].values[0]

if unrec_label == "SUPER":
    xss_15k = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t15000_ind{ind}.chr1.xss.npz"
    with np.load(xss_15k) as d:
        den_states = d["den_states"]
        nea_states = d["nea_states"]
        treespan = d["treespan"]
        treespan_phy = d["treespan_phy"]
        window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
        t2s = d["t2s"]
    # t2s = np.median(t2s, axis=0)
    tracts, states = filter_tracts(
        indiv_pp=t2s,
        treespan=treespan,
        treespan_phy=treespan_phy,
        pp_cutoff=31500,
        arc_cutoff=31500,
        phy_cutoff=10000,
        l_cutoff=0.01,
        remove_margin=0,
    )
    den_tracts = []
    nea_tracts = []
    fil_den_states = np.zeros(states.shape)
    fil_nea_states = np.zeros(states.shape)
    for t in tracts:
        start_idx = int(t[0] / window_size)
        end_idx = int(t[1] / window_size)
        # must inside den segs
        if np.all(den_states[start_idx:end_idx] > 0):
            # not on the margin
            if den_states[start_idx - 1] > 0 and den_states[end_idx] > 0:
                # find den tract start and end
                den_start = start_idx
                while den_start > 0 and den_states[den_start - 1] > 0:
                    den_start -= 1
                den_end = end_idx
                while den_end < len(den_states) and den_states[den_end] > 0:
                    den_end += 1
                den_start_phy = treespan_phy[den_start][0]
                den_end_phy = treespan_phy[den_end - 1][1]
                # find prev and pos tracts
                prev_len = t[0] - np.max([den_start_phy, int(t[0] - t[3])])
                pos_len = np.min([den_end_phy, int(t[1] + t[3])]) - t[1]
                # if prev_len >= 5000 and pos_len >= 5000:
                #     den_tracts.append(t)
                #     fil_den_states[start_idx:end_idx] = 1
                den_tracts.append(t)
                fil_den_states[start_idx:end_idx] = 1
        if np.all(nea_states[start_idx:end_idx] > 0):
            # not on the margin
            if nea_states[start_idx - 1] > 0 and nea_states[end_idx] > 0:
                # find nea tract start and end
                nea_start = start_idx
                while nea_start > 0 and nea_states[nea_start - 1] > 0:
                    nea_start -= 1
                nea_end = end_idx
                while nea_end < len(nea_states) and nea_states[nea_end] > 0:
                    nea_end += 1
                nea_start_phy = treespan_phy[nea_start][0]
                nea_end_phy = treespan_phy[nea_end - 1][1]
                # find prev and pos tracts
                prev_len = t[0] - np.max([nea_start_phy, int(t[0] - t[3])])
                pos_len = np.min([nea_end_phy, int(t[1] + t[3])]) - t[1]
                # if prev_len >= 5000 and pos_len >= 5000:
                #     nea_tracts.append(t)
                #     fil_nea_states[start_idx:end_idx] = 1
                nea_tracts.append(t)
                fil_nea_states[start_idx:end_idx] = 1
    out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(den_tracts)):
        out += (
            "1\t"
            + str(int(den_tracts[i][0]))
            + "\t"
            + str(int(den_tracts[i][1]))
            + "\t"
            + str(round(den_tracts[i][2], 2))
            + "\t"
            + str(int(den_tracts[i][3]))
            + "\t"
            + str(round(den_tracts[i][4], 3))
            + "\n"
        )
    outfile = open(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t31500_ind{ind}.summary.txt", "w")
    outfile.write(out)
    outfile.close()
    out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(nea_tracts)):
        out += (
            "1\t"
            + str(int(nea_tracts[i][0]))
            + "\t"
            + str(int(nea_tracts[i][1]))
            + "\t"
            + str(round(nea_tracts[i][2], 2))
            + "\t"
            + str(int(nea_tracts[i][3]))
            + "\t"
            + str(round(nea_tracts[i][4], 3))
            + "\n"
        )
    outfile = open(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t31500_ind{ind}.nea.summary.txt", "w")
    outfile.write(out)
    outfile.close()
    

if unrec_label == "Ghost":
    t = 15000
else:
    t = 31500

if os.path.exists(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt"):
    os.remove(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt")
hmmixfile = f"results/realdata/test_simulation/{input_model}/msprime/n100_seed{s}.arc.indiv.tsv"
summaryfile = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.summary.txt"
outpref = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth"
append_hmmix_info(hmmixfile, summaryfile, outpref, inference = "truth", individualID = ind)
if unrec_label == "Ghost":
    if df[df['ID']==ind]['Population'].values[0] == "NonAfrican":
        summaryfile = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt"
        hmmixfile = f"results/realdata/test_simulation/{input_model}/hmmix/n100_seed{s}.{df[df['ID']==ind]['Name'].values[0]}.txt"
        outpref = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix"
        append_hmmix_info(hmmixfile, summaryfile, outpref, inference = "hmmix")
        summaryfile = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix.txt"
        hmmixfile = f"results/realdata/test_simulation/{input_model}/ibdmix/n100_seed{s}.arc.ibdmix"
        outpref = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix.ibdmix"
        append_hmmix_info(hmmixfile, summaryfile, outpref, inference = "ibdmix", individualID = individualID)
    else:
        summaryfile = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt"
        hmmixfile = f"results/realdata/test_simulation/{input_model}/ibdmix/n100_seed{s}.arc.ibdmix"
        outpref = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix.ibdmix"
        append_hmmix_info(hmmixfile, summaryfile, outpref, inference = "ibdmix", individualID = individualID)
elif unrec_label == "SUPER":
    for p in ["nea"]:
        hmmix = pd.read_csv(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t31500_ind{ind}.{p}.summary.txt", sep="\s+")
        hmmix["truth_start"] = "-1"
        hmmix["truth_end"] = "-1"
        hmmix["truth_overlap_length(bp)"] = "-1"
        hmmix["truth_assign"] = "-1"
        hmmix.to_csv(
            f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.{p}.truth.txt",
            sep="\t",
            index=False,
        )


if unrec_label == "Ghost":
    summary = [f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix.ibdmix.txt"]
    outpref = [f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.count"]
elif unrec_label == "SUPER":
    summary = [
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt", 
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.nea.truth.txt",
    ]
    outpref = [
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.count",
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.nea.count",
    ]
hap = "left" if ind%2 == 0 else "right"
snpinfo = f"results/realdata/test_simulation/{input_model}/snpinfo/n100_seed{s}_afr.af"
bcfpref = f"results/realdata/test_simulation/{input_model}/msprime/n100_seed{s}"
for i in range(len(summary)):
    final_ind_count(individualID, summary[i], hap, snpinfo, bcfpref, outpref[i])

if unrec_label == "Ghost":
    pref = [f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}"]
elif unrec_label == "SUPER":
    pref = [
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}",
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.nea",
    ]
for i in range(len(pref)):
    append_t1_t2(
        npzpref = f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t15000_ind{ind}.chr1.xss",
        summary = pref[i] + ".count_DAF.txt", 
        snpinfo = str(snpinfo) + ".mutage.txt",
        func = "median",
        outpref = pref[i] + ".count_t1t2"
    )
    os.remove(f"{pref[i]}.count_DAF.txt")
    merge_df_count_branch_mut(
        count = pref[i] + ".count.txt",
        count_t1t2 = pref[i] + ".count_t1t2.txt",
        output = pref[i] + ".count_merge.txt",
        pop = df[df['ID']==ind]['Population'].values[0],
        unrec_label = unrec_label
    )
    os.remove(f"{pref[i]}.count.txt")
    os.remove(f"{pref[i]}.count_t1t2.txt")
    # mafs = [0.05, 0.5, 0.95]
    # for maf in mafs:
    #     count_nd_subtrees(
    #         count_merge = pref[i] + ".count_merge.txt",
    #         output = pref[i] + f".count_merge_maf{maf}.txt",
    #         maf=maf
    #     )

if unrec_label == "Ghost":
    with np.load(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t15000_ind{ind}.chr1.xss.npz") as d:
        data = {k: d[k] for k in d.files} 
    states = data["states"]
    treespan_phy = data["treespan_phy"]
    window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    nea_states = np.zeros(shape=states.shape, dtype=states.dtype)
    den_states = np.zeros(shape=states.shape, dtype=states.dtype)
    ghost_states = np.zeros(shape=states.shape, dtype=states.dtype)
    dff = pd.read_csv(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.count_merge.txt", sep="\t")
    for j in range(len(dff)):
        if dff.loc[j, 'assign_label'] == "NEA":
            nea_states[int(dff.loc[j, 'start'] / window_size):int(dff.loc[j, 'end'] / window_size)] = 1
        elif dff.loc[j, 'assign_label'] == "DEN":
            den_states[int(dff.loc[j, 'start'] / window_size):int(dff.loc[j, 'end'] / window_size)] = 1
        elif dff.loc[j, 'assign_label'] == unrec_label:
            ghost_states[int(dff.loc[j, 'start'] / window_size):int(dff.loc[j, 'end'] / window_size)] = 1
    data["nea_states"] = nea_states
    data["den_states"] = den_states
    data["ghost_states"] = ghost_states
    np.savez_compressed(
        f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t15000_ind{ind}.chr1.xss.npz", **data
    )
    os.remove(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.txt")
    if df[df['ID']==ind]['Population'].values[0] == "NonAfrican":
        os.remove(f"results/realdata/test_simulation/{input_model}/singer_afr/n100_seed{s}_t{t}_ind{ind}.truth.hmmix.txt")