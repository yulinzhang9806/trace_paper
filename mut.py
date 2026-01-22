from workflow.scripts.utils import SNPINFO
import sys

i = int(sys.argv[1])

snpinfo = f"/global/scratch/users/zhangyulin9806/github/ghost_admixture_hmm/results/realdata/1000g_hg38_2022/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{i}"
mutage_pref = f"/global/scratch/users/zhangyulin9806/github/ghost_admixture_hmm/results/realdata/1000g_hg38_2022/singer/polarized_full_phase3samples/mutage_files/mutage_chr{i}_"
mutage_range = range(250, 300)
if i == 21:
    mutage_range = range(237, 287)
outpref = f"/global/scratch/users/zhangyulin9806/github/ghost_admixture_hmm/results/realdata/1000g_hg38_2022/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.mutage.chr{i}"
SNPINFO().append_mutage_info(snpinfo, mutage_pref, mutage_range, outpref)