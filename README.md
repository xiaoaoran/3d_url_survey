This is the official repository of **Unsupervised Representation Learning for Point Clouds: A Survey**, a comprehensive survey of recent progress in deep learning methods for unsupervised point clouds learning. For details, please refer to:

 **Unsupervised Representation Learning for Point Clouds: A Survey**
 
 [[Paper]()]
 

## Menu
- Datasets
- Generation-based Methods
- Context-based Methods
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
22. Progressive Seed Generation Auto-Encoder for Unsupervised Point Cloud Learning. ICCV 2021. [[DPF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Progressive_Seed_Generation_Auto-Encoder_for_Unsupervised_Point_Cloud_Learning_ICCV_2021_paper.pdf)]
23. Unsupervised Learning of Geometric Sampling Invariant Representations for 3D Point Clouds. ICCVW 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021W/GSP-CV/papers/Chen_Unsupervised_Learning_of_Geometric_Sampling_Invariant_Representations_for_3D_Point_ICCVW_2021_paper.pdf)] 
