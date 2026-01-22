    def decode_tmrca(self, ts, xss, f0, g0, p, q, intro_prop, t_archaic, seed):
        hmm = GhostProductHmm()
        hmm.init_hmm(ts, xss, f0=f0, g0=g0, intro_prop=intro_prop, t_archaic=t_archaic, version='tmrca', p = p, q=q)
        res_dict = hmm.train(niter=80, seed=seed, version='tmrca')
        gammas, alphas, betas = hmm.decode(version='tmrca')
        return np.exp(gammas), res_dict

    def get_weighted_pp(self, seglen, pp, treespan):
        """Get weighted pp for segments based on pp and treespan."""
        if treespan[max(treespan)][-1] % seglen < 1:
            c = int(treespan[max(treespan)][-1] / seglen)
        else:
            c = int(treespan[max(treespan)][-1] / seglen) + 1
        out = np.zeros(shape=(1, c))
        j = 0
        for i in range(c):
            while j < len(treespan) and not treespan[j][1] > i * seglen:
                j += 1
            k = 0
            while j + k < len(treespan) and not treespan[j + k][1] >= (i + 1) * seglen:
                k += 1
            if j + k >= len(treespan):
                sumlen = treespan[j + k][1] - i * seglen
                if k >= 1:
                    out[0][i] = pp[j] * (treespan[j][1] - i * seglen) / sumlen + sum(
                        [
                            pp[t] * (treespan[t][1] - treespan[t][0]) / sumlen
                            for t in range(j, j + k)
                        ]
                    )
                else:
                    out[0][i] = pp[j]
            else:
                if k == 0:
                    out[0][i] = pp[j]
                if k == 1:
                    out[0][i] = (
                        pp[j] * (treespan[j][1] - i * seglen) / seglen
                        + pp[j + k]
                        * ((i + 1) * seglen - treespan[j + k - 1][1])
                        / seglen
                    )
                if k > 1:
                    out[0][i] = (
                        pp[j] * (treespan[j][1] - i * seglen) / seglen
                        + pp[j + k]
                        * ((i + 1) * seglen - treespan[j + k - 1][1])
                        / seglen
                        + sum(
                            [
                                pp[t] * (treespan[t][1] - treespan[t][0]) / seglen
                                for t in range(j + 1, j + k)
                            ]
                        )
                    )
            j = j + k
        return out[0]

    def get_pairwise_times(self, ts, s, k1, k2):
        """Function for getting pairwise average tmrca over the genome."""
        windows = np.arange(0, ts.sequence_length, s)
        windows = np.append(windows, ts.sequence_length)
        times = ts.divergence(sample_sets=[[k1], [k2]], windows=windows, mode='branch')
        return windows, 0.5*times

    def singer_avg_pairwise_times(self, s, k1, js, filepath, fsample, outpath):
        """Pairwise tmrca from one singer output.
        
        fsample: int
        """
        ts = tskit.load(filepath + "_" + str(fsample) + ".trees")
        other_samples = np.delete(js, np.where(js == k1))
        out = np.zeros(shape=(int(ts.sequence_length / s + (ts.sequence_length % s > 0)), len(other_samples) + 1))
        for i in range(len(other_samples)):
            windows, div = self.get_pairwise_times(ts, s, k1, other_samples[i])
            out[:, i] = div
        out[:, -1] = np.random.uniform(0, 1, size = out.shape[0])
        np.savez_compressed(outpath + "." + str(fsample) + ".npz", windows = windows, xss = out)
        return windows, out
    def record_tmrca(self, ind, s, js, seed, njob, outpath, filepath, fsample):
        np.random.seed(seed)
        rerun_list = []
        for i in fsample:
            if not Path(outpath + "." + str(i) + ".npz").exists():
                rerun_list.append(i)
        if len(rerun_list) > 0:
            output = Parallel(n_jobs=njob)(delayed(self.singer_avg_pairwise_times)(s = s, k1 = ind, js = js, filepath = filepath, fsample = i, outpath = outpath) for i in rerun_list)
        def decode_gamma_smc(
        zst,
        reader,
        n,
        mu,
        t1,
        t2,
        ):
        sys.path.append(reader)
        import reader
        alphas, betas, meta = reader.open_posteriors(str(zst))
        Ne = meta['scaled_mutation_rate'] / (4 * float(mu))
        tmrca = (alphas / betas) * 2 * Ne
        tmrca = tmrca.iloc[1:, :]
        tmrca = tmrca.T
        tmrca = tmrca.reset_index()
        tmrca[['hap1','hap2']] = df['index'].str.split('_',expand=True)
        pos = tmrca.columns.to_list()[1:]
        treespan = dict()
        for i in range(len(pos)):
            treespan[i] = [pos[i], pos[i+1] - 1]
        n_coal = np.zeros(shape = (2*n, len(pos)))

        # Match sample name here, return ms_to_gmc

        for i in range(2*n):
            dff = tmrca[(tmrca['hap1'] == ms_to_gmc[i]) | (tmrca['hap2'] == ms_to_gmc[i])]
            for j in range(len(pos)):
                n_coal[i][j] = len(dff[(df[pos[j]] > t1) & (df[pos[j]] < t2)][pos[j]].unique())
        return n_coal, treespan 
