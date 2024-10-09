from MML import MML
from MGL import MGL
from load_data import load_ST_file
from utils import clustering
import torch
import pandas as pd
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = "Human_Breast_Cancer"
    file_fold = '../Human_Breast_Cancer/'
    adata_ori = load_ST_file(file_fold)
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata_ori.obs['ground_truth'] = df_meta_layer.values
    adata_ori = adata_ori[~pd.isnull(adata_ori.obs['ground_truth'])]
    n_clusters = len(adata_ori.obs['ground_truth'].unique())
    print("n_clusters:", n_clusters)
    adata_1 = adata_ori.copy()  # for svg
    path = "../Human_Breast_Cancer"
    epochs = 1000
    model = MML(adata_ori, device=device, epochs=epochs, dataset=dataset, name='', path=path)
    adata_result = model.train()
    clustering(adata_result, n_clusters, radius=50, refinement=True)
    ARI = metrics.adjusted_rand_score(adata_result.obs['domain'], adata_result.obs['ground_truth'])
    print('ARI:', ARI)
    adata_1.obs['domain'] = list(adata_result.obs['domain'])
    mgl_svg = MGL(adata_1)
    print(mgl_svg)
