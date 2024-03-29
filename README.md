This is the repository of TPAMI 2023 **Unsupervised Point Cloud Representation Learning with Deep Neural Networks: A Survey**, a comprehensive survey of recent progress in deep learning methods for unsupervised point clouds learning. For details, please refer to:

 **Unsupervised Representation Learning for Point Clouds: A Survey**  
 [[Paper](https://arxiv.org/abs/2202.13589)] 
 
 [![arXiv](https://img.shields.io/badge/arXiv-2202.13589-b31b1b.svg)](https://arxiv.org/abs/2202.13589)
 [![Survey](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

## Abstract
Point cloud data have been widely explored due to its superior accuracy and robustness under various adverse situations.
Meanwhile, deep neural networks (DNNs) have achieved very impressive success in various applications such as surveillance and
autonomous driving. The convergence of point cloud and DNNs has led to many deep point cloud models, largely trained under the
supervision of large-scale and densely-labelled point cloud data. Unsupervised point cloud representation learning, which aims to learn
general and useful point cloud representations from unlabelled point cloud data, has recently attracted increasing attention due to the
constraint in large-scale point cloud labelling. This paper provides a comprehensive review of unsupervised point cloud representation
learning using DNNs. It first describes the motivation, general pipelines as well as terminologies of the recent studies. Relevant
background including widely adopted point cloud datasets and DNN architectures is then briefly presented. This is followed by an
extensive discussion of existing unsupervised point cloud representation learning methods according to their technical approaches. We
also quantitatively benchmark and discuss the reviewed methods over multiple widely adopted point cloud datasets. Finally, we share
our humble opinion about several challenges and problems that could be pursued in the future research in unsupervised point cloud
representation learning.

# Citation
If you find our work useful in your research, please consider citing:
```
@article{xiao2022unsupervised,
  title={Unsupervised Representation Learning for Point Clouds: A Survey},
  author={Xiao, Aoran and Huang, Jiaxing and Guan, Dayan and Lu, Shijian},
  journal={arXiv preprint arXiv:2202.13589},
  year={2022}
}
```

## Menu
- [Datasets](#datasets)
- [Generation-based Methods](#generation-based-methods)
- [Context-based Methods](#context-based-methods)
- [Multiple modal-based methods](#multiple-modal-based-methods)
- [Local descriptor-based methods](#local-descriptor-based-methods)
## Datasets
1. KITTI [[Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)] [[Project Page](http://www.cvlibs.net/datasets/kitti/)]
2. ModelNet [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf)] [[Project Page](https://modelnet.cs.princeton.edu/)]
3. ShapeNet [[Paper](https://arxiv.org/pdf/1512.03012.pdf)] [[Project Page](https://shapenet.org/)]
4. SUN RGB-D [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_SUN_RGB-D_A_2015_CVPR_paper.pdf)] [[Project Page](https://rgbd.cs.princeton.edu/)]
5. S3DIS [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf)] [[Project Page](http://buildingparser.stanford.edu/dataset.html)]
6. ScanNet  [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf)] [[Project Page](http://www.scan-net.org/)]
7. ScanObjectNN  [[Paper](https://arxiv.org/pdf/1908.04616.pdf)] [[Project Page](https://hkust-vgd.github.io/scanobjectnn/)]
8. ONCE [[Paper](https://arxiv.org/abs/2106.11037)] [[Project Page](https://once-for-auto-driving.github.io/)]
## Generation-based Methods
1. VConv-DAE: Deep Volumetric Shape Learning Without Object Labels. ECCV 2016. [[PDF](https://arxiv.org/abs/1604.03755)]; [[Reproduced code #1](https://github.com/Not-IITian/VCONV-DAE)]; [[Reproduced code #2](https://github.com/diskhkme/VCONV_DAE_TF)]
2. Learning a Predictable and Generative Vector Representation for Objects. ECCV 2016. [[PDF](https://arxiv.org/pdf/1603.08637v2.pdf)]
3. Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling. NIPS2016. [[PDF](https://arxiv.org/pdf/1610.07584v2.pdf)] [[Torch 7](https://github.com/zck119/3dgan-release)]
4. Learning Descriptor Networks for 3D Shape Synthesis and Analysis. CVPR 2018. [[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xie_Learning_Descriptor_Networks_CVPR_2018_paper.pdf)] [[Tensorflow](https://github.com/jianwen-xie/3DDescriptorNet)]
5. FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation. CVPR 2018. [[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf)] [[Project Page](https://www.merl.com/research/license/FoldingNet)]
6. SO-Net: Self-Organizing Network for Point Cloud Analysis. CVPR 2018. [[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)] [[Pytorch]( https://github.com/lijx10/SO-Net)]
7. Learning representations and generative models for 3d point clouds. ICML 2018. [[PDF](http://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf)] [[Tensorflow](https://github.com/optas/latent_3d_points)]
8. Multiresolution Tree Networks for 3D Point Cloud Processing. ECCV 2018. [[PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matheus_Gadelha_Multiresolution_Tree_Networks_ECCV_2018_paper.pdf)] [[Pytorch](https://github.com/matheusgadelha/MRTNet)]
9. Deep Spatiality: Unsupervised Learning of Spatially-Enhanced Global and Local 3D Features by Deep Neural Network With Coupled Softmax. TIP 2018. [[PDF](https://yushen-liu.github.io/main/pdf/LiuYS_TIP18DS.pdf)]
10. View Inter-Prediction GAN: Unsupervised Representation Learning for 3D Shapes by Learning Global Shape Memories to Support Local View Predictions. AAAI 2019. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/4852/4725)]
11. Learning localized generative models for 3D point clouds via graph convolution. ICLR 2019. [[PDF](https://openreview.net/pdf?id=SJeXSo09FQ)] [[Tensorflow](https://github.com/diegovalsesia/GraphCNN-GAN-codeonly)]
12. 3D Point Capsule Networks. CVPR 2019. [[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_3D_Point_Capsule_Networks_CVPR_2019_paper.pdf)] [[Pytorch](https://github.com/yongheng1991/3D-point-capsule-networks)]
13. Point cloud gan. ICMLW 2019. [[PDF](https://arxiv.org/pdf/1810.05795.pdf)] [[Code](https://github.com/chunliangli/Point-Cloud-GAN)]
14. L2G Auto-Encoder: Understanding Point Clouds by Local-to-Global Reconstruction with Hierarchical Self-Attention. ACM MM 2019. [[PDF](https://arxiv.org/pdf/1908.00720.pdf)] [[Tensorflow](https://github.com/liuxinhai/L2G-AE)]
15. Multi-Angle Point Cloud-VAE: Unsupervised Feature Learning for 3D Point Clouds From Multiple Angles by Joint Self-Reconstruction and Half-to-Half Prediction. ICCV 2019. [[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Multi-Angle_Point_Cloud-VAE_Unsupervised_Feature_Learning_for_3D_Point_Clouds_ICCV_2019_paper.pdf)] 
16. Pointflow: 3d point cloud generation with continuous normalizing flows. ICCV 2019. [[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_PointFlow_3D_Point_Cloud_Generation_With_Continuous_Normalizing_Flows_ICCV_2019_paper.pdf)] [[Pytorch](https://github.com/stevenygd/PointFlow)]
17. Unsupervised Deep Shape Descriptor with Point Distribution Learning. CVPR 2020. [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Unsupervised_Deep_Shape_Descriptor_With_Point_Distribution_Learning_CVPR_2020_paper.pdf)] [[Pytorch](https://github.com/WordBearerYI/Unsupervised-Deep-Shape-Descriptor-with-Point-Distribution-Learning)]
18. Point cloud completion by skip-attention network with hierarchical folding. CVPR 2020. [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wen_Point_Cloud_Completion_by_Skip-Attention_Network_With_Hierarchical_Folding_CVPR_2020_paper.pdf)] [[Pytorch](https://github.com/RaminHasibi/SA_Net)]
19. Pointgrow: Autoregressively learned point cloud generation with self-attention. WACV 2020. [[PDF](https://openaccess.thecvf.com/content_WACV_2020/papers/Sun_PointGrow_Autoregressively_Learned_Point_Cloud_Generation_with_Self-Attention_WACV_2020_paper.pdf)] [[Project](https://liuziwei7.github.io/projects/PointGrow)]
20. Deep Unsupervised Learning of 3D Point Clouds via Graph Topology Inference and Filtering. IEEE TIP 2020. [[PDF](https://arxiv.org/pdf/1905.04571.pdf)]
21. Self-Supervised Learning on 3D Point Clouds by Learning Discrete Generative Models. CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Eckart_Self-Supervised_Learning_on_3D_Point_Clouds_by_Learning_Discrete_Generative_CVPR_2021_paper.pdf)] 
22. Diffusion Probabilistic Models for 3D Point Cloud Generation. CVPR2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.pdf)] [[PyTorch](https://github.com/luost26/diffusion-point-cloud)]
23. Progressive Seed Generation Auto-Encoder for Unsupervised Point Cloud Learning. ICCV 2021. [[DPF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Progressive_Seed_Generation_Auto-Encoder_for_Unsupervised_Point_Cloud_Learning_ICCV_2021_paper.pdf)]
24. Unsupervised Learning of Geometric Sampling Invariant Representations for 3D Point Clouds. ICCVW 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021W/GSP-CV/papers/Chen_Unsupervised_Learning_of_Geometric_Sampling_Invariant_Representations_for_3D_Point_ICCVW_2021_paper.pdf)] 
25. Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling. CVPR2022. [[PDF](https://arxiv.org/abs/2111.14819)] [[Project](https://point-bert.ivg-research.xyz/)]
26. Point Cloud Pre-training with Natural 3D Structures. CVPR2022. [[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yamada_Point_Cloud_Pre-Training_With_Natural_3D_Structures_CVPR_2022_paper.pdf)]
27. Masked Autoencoders for Point Cloud Self-supervised Learning. ECCV2022. [[PDF](https://arxiv.org/abs/2203.06604)] [[Pytorch](https://github.com/Pang-Yatian/Point-MAE)]
28. Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training. NeurIPS2022. [[PDF](https://arxiv.org/pdf/2205.14401.pdf)] [[pytorch](https://github.com/ZrrSkywalker/Point-M2AE)]

## Context-based methods
1. Unsupervised Multi-Task Feature Learning on Point Clouds. ICCV 2019. [[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf)] 
2. Self-supervised deep learning on point clouds by reconstructing space. NIPS 2019. [[PDF](https://proceedings.neurips.cc/paper/2019/file/993edc98ca87f7e08494eec37fa836f7-Paper.pdf)] [[Pytorch](https://github.com/COMP6248-Reproducability-Challenge/Self-supervised-deep-learning-on-point-clouds-by-reconstructing-space)]
3. Unsupervised feature learning for point cloud understanding by contrasting and clustering using graph convolutional neural networks. 3DV 2019. [[PDF](https://par.nsf.gov/servlets/purl/10124686)] [[Tensorflow+Matlab](https://github.com/lingzhang1/ContrastNet)]
4. Context Prediction for Unsupervised Deep Learning on Point Clouds. arXiv 2019. [[PDF](https://arxiv.org/pdf/1901.08396.pdf)]
5. Info3D: Representation Learning on 3D Objects Using Mutual Information Maximization and Contrastive Learning. ECCV 2020. [[PDF](https://arxiv.org/pdf/2006.02598.pdf%5C%22)]
6. PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding. ECCV 2020. [PDF](https://arxiv.org/pdf/2007.10985.pdf?ref=https://githubhelp.com)] [[Pytorch](https://github.com/facebookresearch/PointContrast)]
7. Label-efficient learning on point clouds using approximate convex decompositions. ECCV 2020. [[PDF](https://arxiv.org/pdf/2003.13834.pdf)] [[Pytorch](https://github.com/matheusgadelha/PointCloudLearningACD)]
8. Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds. CVPR 2020. [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rao_Global-Local_Bidirectional_Reasoning_for_Unsupervised_Representation_Learning_of_3D_Point_CVPR_2020_paper.pdf)] [[PyTorch](https://github.com/raoyongming/PointGLR)]
9. Self-supervised learning of local features in 3d point clouds. CVPRW 2020. [[PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Thabet_Self-Supervised_Learning_of_Local_Features_in_3D_Point_Clouds_CVPRW_2020_paper.pdf)] [Pytorch](https://github.com/alitabet/morton-net)]
10. Self-Supervised Learning of Point Clouds via Orientation Estimation. 3DV 2020. [[PDF](https://arxiv.org/pdf/2008.00305.pdf%5C%22)] [[Tensorflow](https://github.com/OmidPoursaeed/Self_supervised_Learning_Point_Clouds)]
11. Unsupervised 3D Learning for Shape Analysis via Multiresolution Instance Discrimination. AAAI 2021. [[Project](https://wang-ps.github.io/pretrain.html)]
12. Self-Contrastive Learning with Hard Negative Sampling for Self-supervised Point Cloud Learning. ACMMM 2021. [[Paper](https://arxiv.org/pdf/2107.01886.pdf)]
13. Exploring data-efficient 3D scene understanding with contrastive scene contexts. CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Exploring_Data-Efficient_3D_Scene_Understanding_With_Contrastive_Scene_Contexts_CVPR_2021_paper.pdf)] [[Pytorch](https://github.com/facebookresearch/ContrastiveSceneContexts)]
14. Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Spatio-Temporal_Self-Supervised_Representation_Learning_for_3D_Point_Clouds_ICCV_2021_paper.pdf)] [[Project](https://siyuanhuang.com/STRL/)]
15. RandomRooms: Unsupervised Pre-training from Synthetic Shapes and Randomized Layouts for 3D Object Detection. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Rao_RandomRooms_Unsupervised_Pre-Training_From_Synthetic_Shapes_and_Randomized_Layouts_for_ICCV_2021_paper.pdf)]
16. Unsupervised Point Cloud Pre-Training via View-Point Occlusion, Completion. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Unsupervised_Point_Cloud_Pre-Training_via_Occlusion_Completion_ICCV_2021_paper.pdf)] [[Project](https://hansen7.github.io/OcCo/)]
17. Self-Supervised Pretraining of 3D Features on any Point-Cloud. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Self-Supervised_Pretraining_of_3D_Features_on_Any_Point-Cloud_ICCV_2021_paper.pdf)] [[Pytorch](https://github.com/facebookresearch/DepthContrast)]
18. Shape Self-Correction for Unsupervised Point Cloud Understanding. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Shape_Self-Correction_for_Unsupervised_Point_Cloud_Understanding_ICCV_2021_paper.pdf)]
19. Pri3D: Can 3D Priors Help 2D Representation Learning?. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Pri3D_Can_3D_Priors_Help_2D_Representation_Learning_ICCV_2021_paper.pdf)] [[Pytorch](https://github.com/Sekunde/Pri3D)]
20. Exploring Geometry-Aware Contrast and Clustering Harmonization for Self-Supervised 3D Object Detection. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Exploring_Geometry-Aware_Contrast_and_Clustering_Harmonization_for_Self-Supervised_3D_Object_ICCV_2021_paper.pdf)]
21. 4DContrast: Contrastive Learning with Dynamic Correspondences for 3D Scene Understanding. ECCV 2022. [[PDF](https://arxiv.org/pdf/2112.02990.pdf)] [[Project](http://www.niessnerlab.org/projects/chen2021_4dcontrast.html)]
22. A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision. ECCV 2022. [[PDF](https://arxiv.org/pdf/2207.04997.pdf)] [[PyTorch](https://github.com/lilanxiao/Invar3D)]
23. ProposalContrast: Unsupervised Pre-training for LiDAR-based 3D Object Detection. ECCV2022. [[PDF](https://arxiv.org/pdf/2207.12654.pdf)] [[Project](https://github.com/yinjunbo/ProposalContrast)]
24. Masked Discrimination for Self-Supervised Learning on Point Clouds. ECCV2022. [[PDF](https://arxiv.org/pdf/2203.11183.pdf)] [[Project](https://github.com/haotian-liu/MaskPoint)]
25. OGC: Unsupervised 3D Object Segmentation from Rigid Dynamics of Point Clouds. NeurIPS2022. [[PDF](https://arxiv.org/pdf/2210.04458)] [[Pytorch](https://github.com/vLAR-group/OGChttps://arxiv.org/pdf/2210.04458.pdf)]
26. PointClustering: Unsupervised Point Cloud Pre-training using Transformation Invariance in Clustering. CVPR 2023.
27. Complete-to-Partial 4D Distillation for Self-Supervised Point Cloud Sequence Representation Learning. CVPR 2023. [[PDF](https://arxiv.org/pdf/2212.05330.pdf)]
## Multiple modal-based methods
1. Self-supervised feature learning by cross-modality and cross-view correspondences. CVPRW 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021W/MULA/papers/Jing_Self-Supervised_Feature_Learning_by_Cross-Modality_and_Cross-View_Correspondences_CVPRW_2021_paper.pdf)]
2. CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding. CVPR2022. [[PDF](https://arxiv.org/pdf/2203.00680.pdf)] [[Pytorch](https://github.com/MohamedAfham/CrossPoint)]
3. P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting. NeurIPS 2022. [[PDF](https://arxiv.org/pdf/2208.02812.pdf)] [[project](https://github.com/wangzy22/P2P)]
## Local descriptor-based methods
1. Ppf-foldnet: Unsupervised learning of rotation invariant 3d local descriptors. ECCV 2018. [[PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tolga_Birdal_PPF-FoldNet_Unsupervised_Learning_ECCV_2018_paper.pdf)] [[Pytorch](https://github.com/XuyangBai/PPF-FoldNet)]
2. Corrnet3d: unsupervised end-to-end learning of dense correspondence for 3d point clouds. CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zeng_CorrNet3D_Unsupervised_End-to-End_Learning_of_Dense_Correspondence_for_3D_Point_CVPR_2021_paper.pdf)] [[Pytorch](https://github.com/ZENGYIMING-EAMON/CorrNet3D)]
3. Dpc: Unsupervised deep point correspondence via cross and self construction. 3DV 2021. [[PDF](https://arxiv.org/pdf/2110.08636.pdf)] [[Pytorch](https://github.com/dvirginz/dpc)]
4. Sampling network guided cross-entropy method for unsupervised point cloud registration. ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Sampling_Network_Guided_Cross-Entropy_Method_for_Unsupervised_Point_Cloud_Registration_ICCV_2021_paper.pdf)] [[Pytorch](https://github.com/jiang-hb/cemnet)]
5. Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration. CVPR 2023. [[pdf](https://arxiv.org/pdf/2303.13290.pdf)] [[Pytorch](https://github.com/gfmei/UDPReg)]
