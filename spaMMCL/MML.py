import torch
import os
import random
import numpy as np
from torch.backends import cudnn
import scanpy as sc
import ot
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from torch.utils import data
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import torch.nn as nn
import pandas as pd
import scipy.sparse as sp
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, act=F.relu):
        super(Encoder, self).__init__()
        self.p_drop = 0.0
        self.mask_rate = 0.3
        self.in_features = in_features
        self.img_features = 2048
        projection_dims_1 = 1024
        projection_dims_2 = 512
        gcn_hid_1 = 256
        gcn_hid_2 = 128
        img_out_features = 16
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        # projector
        self.gene_pre = nn.Sequential(nn.Linear(self.in_features, projection_dims_1), nn.ELU())
        self.img_pre = nn.Sequential(nn.Linear(self.img_features, projection_dims_1), nn.ELU())
        self.projector = nn.Sequential(
            nn.Linear(projection_dims_1, projection_dims_2),
            nn.ELU()
        )
        # gene_encoder
        self.gc_gene_1 = GraphConvolution(projection_dims_2, gcn_hid_1, self.p_drop)
        self.gc_gene_2 = GraphConvolution(gcn_hid_1, gcn_hid_2, self.p_drop)
        # img_encoder
        self.liner_img = nn.Sequential(nn.Linear(gcn_hid_1, img_out_features), nn.ELU())
        self.fusion = nn.Sequential(nn.Linear(gcn_hid_2 + img_out_features, self.out_features), nn.ELU())
        # # Decoder
        self.decoder_gene = nn.Linear(gcn_hid_2, self.in_features)
        self.decoder_all = nn.Sequential()
        self.decoder_all.add_module('decoder_1', nn.Linear(self.out_features, self.in_features))
        # other
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def encoding_mask_noise(self, x, image_feat, if_mask, mask_rate=0.3):
        if if_mask:
            out_x = x.clone()
            num_nodes = x.shape[0]
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes:]
            token_nodes = mask_nodes
            out_x[token_nodes] = image_feat[token_nodes]
            return out_x, mask_nodes, keep_nodes
        else:
            return x, 'mask_nodes', 'keep_nodes'

    def forward(self, feat, feat_a, adj, image_feat, if_mask=True, a=0.1):
        # mask
        feat = self.gene_pre(feat)
        image_feat = self.img_pre(image_feat)
        feat = self.projector(feat)
        image_feat = self.projector(image_feat)
        feat, mask_nodes, keep_nodes = self.encoding_mask_noise(feat, image_feat, if_mask, self.mask_rate)

        # xg and xi
        xg = self.gc_gene_1(feat, adj)
        xg = self.gc_gene_2(xg, adj)
        z_gene = xg
        xi = self.gc_gene_1(image_feat, adj)
        xi = self.liner_img(xi)
        z_img = xi
        z_fusion = torch.cat((z_gene, z_img), dim=1)
        z = self.fusion(z_fusion)
        # decoder(shard)
        h_cross = self.decoder_gene(z)
        h_gene = self.decoder_gene(z_gene)
        # positive
        z_p = z
        z_p = self.read(z_p, self.graph_neigh)
        z_p = self.sigm(z_p)
        # negative
        feat_a = self.gene_pre(feat_a)
        feat_a = self.projector(feat_a)
        z_n = self.gc_gene_1(feat_a, adj)
        z_n = self.gc_gene_2(z_n, adj)
        # ret
        ret = self.disc(z_p, z, z_n)
        return z, z_gene, z_img, h_cross, ret, mask_nodes, keep_nodes, h_gene


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_normalized = adj_normalized.tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat


def graph_construction(adata, n=6):
    adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    adj_m1 = sp.coo_matrix(adj_m1)
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()
    adj_norm_m1 = preprocess_graph(adj_m1)
    return adj_norm_m1


def image_handel(xi, device, image_encoder):
    xi = xi.to(device)
    return image_encoder(xi)


def tiling(
        adata, out_path=None, library_id: str = None, crop_size: int = 40,
        target_size: int = 299, verbose: bool = False, copy: bool = False):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Check the exist of out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    tile_names = []

    with tqdm(
            total=len(adata),
            desc="Tiling image",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )

            tile_name = str(barcode) + str(crop_size)
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))

            tile.save(out_tile, "JPEG")
            pbar.update(1)
    # adata.obs["tile_path"] = tile_names
    tile_names = np.array(tile_names)

    out_ti_path = os.path.join(out_path, "ti_path_{}.npy".format(crop_size))
    np.save(out_ti_path, tile_names)
    return adata if copy else None


class imgDataset(data.Dataset):
    def __init__(self, adata, dataset, path, name, img_size=40, target_size=299, if_tiling=False):
        super(imgDataset, self).__init__()
        self.obs_names = list(adata.obs.index)
        tiling_path = os.path.join(path, name, f'tilingfile')
        out_ti_path = os.path.join(tiling_path, "ti_path_{}.npy".format(img_size))
        if if_tiling:
            print('start tiling')
            tiling(adata, tiling_path, crop_size=img_size, target_size=target_size)
            print("tiling complete")
        patches = []
        ti_path = np.load(out_ti_path)
        for tile_path in ti_path:
            tile = Image.open(tile_path)
            tile = np.asarray(tile, dtype="int32")
            tile = tile.astype(np.float32)
            patches.append(tile)
        patches = np.array(patches)
        adata.obsm['patches'] = patches
        self.adata = adata
        print('patches', adata.obsm['patches'].shape)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        xi = self.img_transform(self.adata.obsm['patches'][idx])
        na = self.obs_names[idx]
        return xi, na


def image_load(if_img, adata, path, name, img_size, target_size, if_tiling, dataset):
    if if_img:
        imgset = imgDataset(adata, dataset, path, name, img_size=img_size, target_size=target_size, if_tiling=if_tiling)
        imgloader = DataLoader(imgset, batch_size=128, shuffle=False, pin_memory=False)
        image_encoder = resnet152(pretrained=True)
        image_encoder.requires_grad_(False)
        image_encoder.fc = nn.Identity()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_encoder.to(device)
        feature_dim = []
        barcode = []
        for i, batch in enumerate(imgloader):
            image, image_code = batch
            feature = image_handel(image, device, image_encoder)
            feature_dim.append(feature.data.cpu().numpy())
            barcode.append(image_code)
        feature_dim = np.concatenate(feature_dim)
        barcode = np.concatenate(barcode)
        data_frame = pd.DataFrame(data=feature_dim, index=barcode)
        save_fileName = os.path.join(path, name, '{}_image_feat.csv'.format(dataset))
        data_frame.to_csv(save_fileName)
        image_representation = pd.read_csv(save_fileName, index_col='Unnamed: 0')
    else:
        save_fileName = os.path.join(path, name, '{}_image_feat.csv'.format(dataset))
        image_representation = pd.read_csv(save_fileName, index_col='Unnamed: 0')

    print('image_representation.shape:', image_representation.shape)
    image_representation = np.array(image_representation)
    adata.obsm['image_representation'] = image_representation


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated

def get_feature(adata):
    adata_Vars = adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]
    feat_a = permutation(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a

def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL

def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
    adata.obsm['graph_neigh'] = interaction

def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class MML():
    def __init__(self,
                 adata_o,
                 device=torch.device('cuda'),
                 learning_rate=0.001,
                 weight_decay=0.001,
                 epochs=1000,
                 dim_output=128,
                 random_seed=2024,
                 dataset="Human_Breast_Cancer",
                 path="../Human_Breast_Cancer",
                 img_size=80,
                 target_size=299,
                 if_img=False,
                 if_tiling=False,
                 name=""
                 ):
        '''
        Parameters
        ----------
        adata : adata_o
            AnnData object of spatial data.
        device : string, optional
            Using GPU or CPU? The default is 'cuda'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.001.
        epochs : int, optional
            Epoch for model training. The default is 1000.
        dim_output : int, optional
            Dimension of output representation. The default is 128.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2024.
        dataset : string, optional
            The name of the data set. The default is "Human_Breast_Cancer".
        path : string, optional
            The address of the data set. The default is "../Human_Breast_Cancer".
        img_size : int, optional
            The size of the picture after cutting. The default is 80.
        target_size: int, optional
            This parameter is deprecated
        if_img: bool, optional
            Whether to perform image-related operations. The default is False.
        if_tiling: bool, optional
            Whether to perform image cutting. The default is False.
        path : string, optional
            The subdirectory name of the data set. The default is "".
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata_o.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.dataset = dataset
        self.path = path
        self.img_size = img_size
        self.target_size = target_size
        self.if_img = if_img
        self.if_tiling = if_tiling
        self.name = name

        fix_seed(self.random_seed)

        # 1 highly_variable
        if 'highly_variable' not in self.adata.var.keys():
            preprocess(self.adata)
        # 2 graph_neigh
        if 'graph_neigh' not in self.adata.obsm.keys():
            construct_interaction(self.adata)
        # 3 label_CSL
        if 'label_CSL' not in self.adata.obsm.keys():
            add_contrastive_label(self.adata)
        # 4 gene_feat
        if 'feat' not in self.adata.obsm.keys():
            get_feature(self.adata)
        # 5 image_feat
        if 'image_representation' not in self.adata.obsm.keys():
            image_load(self.if_img, self.adata, self.path, self.name, self.img_size, self.target_size, self.if_tiling,
                       self.dataset)

        adj_norm = graph_construction(self.adata, 7)
        self.adj_norm = adj_norm.to(self.device)
        self.image_features = torch.FloatTensor(self.adata.obsm['image_representation'].copy()).to(self.device)
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.features.shape[0])).to(
            self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate,
                                           weight_decay=self.weight_decay)
        print('Begin to train ST data...')
        self.model.train()
        for _ in tqdm(range(self.epochs)):
            self.model.train()
            z, z_gene, z_img, h_cross, ret, mask_nodes, keep_nodes, h_gene = self.model(
                self.features,
                self.features_a,
                self.adj_norm,
                self.image_features,
                if_mask=True)
            x_init_msak = self.features[mask_nodes]
            x_rec_mask = h_gene[mask_nodes]
            loss_mask = F.mse_loss(x_rec_mask, x_init_msak)
            loss_nomask = F.mse_loss(h_cross, self.features)
            loss_sl_att = self.loss_CSL(ret, self.label_CSL)
            loss = loss_nomask + 10 * loss_mask + loss_sl_att
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = \
                self.model(self.features, self.features_a, self.adj_norm, self.image_features, if_mask=False)[
                    0].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            return self.adata