# A multi-modality and multi-granularity collaborative learning framework for identifying spatial domains and spatially variable genes




![](https://github.com/liangxiao-cs/spaMMCL/blob/main/Framework.jpg)

## Overview
The spaMMCL framework consists of two components: multi-modality learning module (MML) for spatial domains identification and multi-granularity learning module (MGL) for SVGs detection. The MML module contains three components: modality bias mitigation with feature mask-like strategy, multi-modal features fusion with joint learning and noise mitigation with graph self-supervised learning. The MGL module also contains three components: a fine-grained screening strategy, a coarse-grained screening strategy and a granularity-supplemented constraint strategy. spaMMCL first employs MML module to collaboratively learn gene expression, histological images and spatial context, while accounting for modal deviation phenomena during the integration of multi-modal data. Subsequently, spaMMCL employs MGL module, a granularity- guided approach, to identify more accurate spatial domain- specific SVGs at different scales.


## Example

For training spaMMCL model, run

'python spaMMCL.py'


## Citation
Liang et al. A multi-modality and multi-granularity collaborative learning framework for identifying spatial domains and spatially variable genes. Bioinformatics, 2024, 40(10): btae607
