import scanpy as sc
from utils import calculate_adj_matrix
import numpy as np
import pandas as pd
import random
from scipy.sparse import issparse

# 1003
def sup_grained(input_adata, target_cluster, nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list = nbr_list + [target_cluster]
        adata = input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata = input_adata.copy
    adata.obs["target"] = ((adata.obs[label_col] == target_cluster) * 1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target", reference="rest", n_genes=adata.shape[1], method='wilcoxon',
                            key_added='wilcoxon_3')
    genes = [i[1] for i in adata.uns['wilcoxon_3']["names"]]
    if issparse(adata.X):
        obs_tidy = pd.DataFrame(adata.X.A)
    else:
        obs_tidy = pd.DataFrame(adata.X)
    obs_tidy.index = adata.obs["target"].tolist()
    obs_tidy.columns = adata.var.index.tolist()
    obs_tidy = obs_tidy.loc[:, genes]
    mean_obs = obs_tidy.groupby(level=0).mean()
    df = {'genes': genes,
          "mean_dif": (mean_obs.loc[1] - mean_obs.loc[0]).tolist()
          }
    df = pd.DataFrame(data=df)
    return df

# 1003
def coarse_grained(input_adata, target_cluster, nbr_list, label_col, adj_nbr=True, log=False):
    adata = input_adata.copy()
    adata.obs["target"] = ((adata.obs[label_col] == target_cluster) * 1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target", reference="rest", n_genes=adata.shape[1], method='wilcoxon',
                            key_added='wilcoxon_test_2')
    pvals_adj = list(map(lambda x: x[0], adata.uns['wilcoxon_test_2']["pvals_adj"]))
    genes = list(map(lambda x: x[1], adata.uns['wilcoxon_test_2']["names"]))
    if issparse(adata.X):
        obs_tidy = pd.DataFrame(adata.X.A)
    else:
        obs_tidy = pd.DataFrame(adata.X)
    obs_tidy.index = adata.obs["target"].tolist()
    obs_tidy.columns = adata.var.index.tolist()
    df = {'genes': genes,
          "pvals_adj": pvals_adj
         }
    df = pd.DataFrame(data=df)
    return df


# 1003
def fine_grained(input_adata, target_cluster, nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list = nbr_list + [target_cluster]
        adata = input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata = input_adata.copy
    adata.obs["target"] = ((adata.obs[label_col] == target_cluster) * 1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target", reference="rest", n_genes=adata.shape[1], method='wilcoxon',
                            key_added='wilcoxon_test_1')
    pvals_adj = list(map(lambda x: x[0], adata.uns['wilcoxon_test_1']["pvals_adj"]))
    genes = list(map(lambda x: x[1], adata.uns['wilcoxon_test_1']["names"]))
    if issparse(adata.X):
        obs_tidy = pd.DataFrame(adata.X.A)
    else:
        obs_tidy = pd.DataFrame(adata.X)
    obs_tidy.index = adata.obs["target"].tolist()
    obs_tidy.columns = adata.var.index.tolist()
    obs_tidy = obs_tidy.loc[:, genes]
    mean_obs = obs_tidy.groupby(level=0).mean()
    if log:
        fold_change = np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
    else:
        fold_change = (mean_obs.loc[1] / (mean_obs.loc[0] + 1e-9)).values
    df = {'genes': genes,
          "fold_change": fold_change.tolist(),
          "pvals_adj": pvals_adj}
    df = pd.DataFrame(data=df)
    return df


# 1003
def find_neighbor_clusters(target_cluster, cell_id, x, y, pred, radius, ratio=1 / 2):
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    df = {'cell_id': cell_id, 'x': x, "y": y, "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    nbr_num = {}
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr_num[p] = nbr_num.get(p, 0) + 1
    del nbr_num[target_cluster]
    nbr_num_back = nbr_num.copy()  # Backup
    nbr_num = [(k, v) for k, v in nbr_num.items() if v > (ratio * cluster_num[k])]
    nbr_num.sort(key=lambda x: -x[1])
    ret = [t[0] for t in nbr_num]
    if len(ret) == 0:
        nbr_num_back = [(k, v) for k, v in nbr_num_back.items()]
        nbr_num_back.sort(key=lambda x: -x[1])
        ret = [nbr_num_back[0][0]]
    return ret

# 1003
def count_nbr(target_cluster, cell_id, x, y, pred, radius):
    df = {'cell_id': cell_id, 'x': x, "y": y, "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)

def search_radius(target_cluster, cell_id, x, y, pred, start, end, num_min=8, num_max=15, max_run=100):
    run = 0
    num_low = count_nbr(target_cluster, cell_id, x, y, pred, start)
    num_high = count_nbr(target_cluster, cell_id, x, y, pred, end)
    if num_min <= num_low <= num_max:
        return start
    elif num_min <= num_high <= num_max:
        return end
    elif num_low > num_max:
        # print("Try smaller start.")
        return None
    elif num_high < num_min:
        # print("Try bigger end.")
        return None
    while (num_low < num_min) and (num_high > num_min):
        run += 1
        if run > max_run:
            return None
        mid = (start + end) / 2
        num_mid = count_nbr(target_cluster, cell_id, x, y, pred, mid)
        if num_min <= num_mid <= num_max:
            return mid
        if num_mid < num_min:
            start = mid
            num_low = num_mid
        elif num_mid > num_max:
            end = mid
            num_high = num_mid

# 1003
def define_SVGs(adata, target, x_name, y_name, domain_name, all_targets):
    nbr_all = [x for x in all_targets if x != target]
    adj_2d = calculate_adj_matrix(x=adata.obs[x_name].tolist(), y=adata.obs[y_name].tolist())
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)
    r = search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=adata.obs[x_name].tolist(),
                      y=adata.obs[y_name].tolist(), pred=adata.obs[domain_name].tolist(), start=start, end=end,
                      num_min=10, num_max=14, max_run=100)
    if r is not None:
        nbr_domians = find_neighbor_clusters(target_cluster=target,
                                             cell_id=adata.obs.index.tolist(),
                                             x=adata.obs[x_name].tolist(),
                                             y=adata.obs[y_name].tolist(),
                                             pred=adata.obs[domain_name].tolist(),
                                             radius=r,
                                             ratio=1 / 2)
        nbr_domians = nbr_domians[0:3]
        other_domians = [x for x in nbr_all if x not in nbr_domians]
        n = int(len(other_domians) / 2)
        random.seed(0)
        rand_domians = random.sample(other_domians, k=n)
    else:
        random.seed(0)
        nbr_domians = random.sample(nbr_all, 2)
        other_domians = [x for x in nbr_all if x not in nbr_domians]
        n = int(len(other_domians) / 2)
        rand_domians = random.sample(other_domians, k=n)

    fine_info = fine_grained(input_adata=adata,
                                        target_cluster=target,
                                        nbr_list=nbr_domians,
                                        label_col=domain_name,
                                        adj_nbr=True,
                                        log=True)
    min_fold_change = 1.5
    fine_info = fine_info[(fine_info["fold_change"] > min_fold_change)]
    fine_info = fine_info.sort_values(by="pvals_adj", ascending=False)
    fine_info["target_dmain"] = target
    fine_info["neighbors"] = str(nbr_domians)

    coarse_info = coarse_grained(input_adata=adata,
                                          target_cluster=target,
                                          nbr_list=nbr_domians,
                                          label_col=domain_name,
                                          adj_nbr=True,
                                          log=True)
    coarse_info = coarse_info[(coarse_info["pvals_adj"] < 0.05)]
    coarse_info = coarse_info.sort_values(by="pvals_adj", ascending=False)
    coarse_info["target_dmain"] = target
    coarse_info["neighbors"] = str(nbr_domians)

    sup_info = sup_grained(input_adata=adata,
                                          target_cluster=target,
                                          nbr_list=rand_domians,
                                          label_col=domain_name,
                                          adj_nbr=True,
                                          log=True)
    sup_info = sup_info[(sup_info["mean_dif"] > 0)]

    fine = fine_info["genes"].tolist()
    coarse = coarse_info["genes"].tolist()
    sup = sup_info["genes"].tolist()
    union_set = set(fine).intersection(set(coarse)).intersection(set(sup))
    return union_set, fine, coarse


def MGL(adata):
    targets = sorted(set(adata.obs['domain']))
    adata.obs['x_pixel'] = adata.obs["imagerowsvg"].tolist()
    adata.obs['y_pixel'] = adata.obs["imagecolsvg"].tolist()
    adata.obs["pred"] = list(adata.obs['domain'])
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    sc.pp.log1p(adata)
    svgrelust = []
    fine_result = []
    coarse_result = []
    adata.obs["x_array"] = adata.obsm["spatial"][:, 0].tolist()
    adata.obs["y_array"] = adata.obsm["spatial"][:, 1].tolist()
    # targets=['1']
    for ii in targets:
        svg_list, fine, coarse = define_SVGs(adata, target=ii, x_name="x_array", y_name="y_array",
                                             domain_name="pred", all_targets=targets)
        svgrelust.append(svg_list)
        fine_result.append(fine)
        coarse_result.append(coarse)
    mgl_svg = [item for sublist in svgrelust for item in sublist]
    mgl_svg = list(set(mgl_svg))
    fine_svg = [item for sublist in fine_result for item in sublist]
    fine_svg = list(set(fine_svg))
    coarse_svg = [item for sublist in coarse_result for item in sublist]
    coarse_svg = list(set(coarse_svg))
    print('xi,cu:', len(fine_svg), len(coarse_svg))
    print('mgl_svg:', len(mgl_svg))
    return mgl_svg
