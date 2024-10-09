import scanpy as sc
import os
import pandas as pd


def convert_scanpy(adata, use_quality='hires'):
    adata.var_names_make_unique()
    library_id = list(adata.uns["spatial"].keys())[0]
    if use_quality == "fulres":
        image_coor = adata.obsm["spatial"]
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + use_quality + "_scalef"
            ]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = use_quality
    # imagecolsvg and imagerowsvg only for plot svg
    adata.obs["imagecolsvg"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagerowsvg"] = adata.obsm["spatial"][:, 1]
    return adata

def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5 = convert_scanpy(adata_h5)
    if load_images is False:
        if file_adj is None:
            file_adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")
        positions = pd.read_csv(file_adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)
    return adata_h5