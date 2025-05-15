from MML import MML
from MGL import MGL
from load_data import load_ST_file
from utils import clustering
import torch
import pandas as pd
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")


class spaMMCL():
    def __init__(self, adata_1, adata_2, epochs, device, dataset, name, path):
        self.n_clusters = len(adata_1.obs['ground_truth'].unique())
        self.adata_1 = adata_1
        self.adata_2 = adata_2
        self.epochs = epochs
        self.device = device
        self.dataset = dataset
        self.name = name
        self.path = path
    # It is recommended to give priority to using the version with images (MML.py). If there are no images, use the version without images (MML_without_img.py).
    def run_MML(self):
        print("Running MML")
        print("n_clusters:", self.n_clusters)
        model_1 = MML(self.adata_1, device=self.device, epochs=self.epochs, dataset=self.dataset, name=self.name, path=self.path)
        adata_result = model_1.train()
        clustering(adata_result, self.n_clusters, radius=50, refinement=True)
        ARI = metrics.adjusted_rand_score(adata_result.obs['domain'], adata_result.obs['ground_truth'])
        print('ARI:', ARI)
        return adata_result

    def run_MGL(self, adata_mml):
        print("Running MGL")
        adata_2.obs['domain'] = list(adata_mml.obs['domain'])
        mgl_svg = MGL(adata_2)
        print(mgl_svg)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = "Human_Breast_Cancer"
    file_fold = '../Human_Breast_Cancer/'
    path = "../Human_Breast_Cancer"
    name = ''
    adata_1 = load_ST_file(file_fold)
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata_1.obs['ground_truth'] = df_meta_layer.values
    adata_1 = adata_1[~pd.isnull(adata_1.obs['ground_truth'])]
    adata_2 = adata_1.copy()  # for svg
    epochs = 1000
    model = spaMMCL(adata_1, adata_2, epochs, device, dataset, name, path)
    adata_mml = model.run_MML()
    model.run_MGL(adata_mml)


