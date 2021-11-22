## 2016 ECCV
1. Learning a predictable and generative vector representation for objects. **[Generative|AutoEncoder]**
   - [要点] A good vector representation of an object should be **generative in 3D** (it can produce new 3D objects) 
   as well as be **predictable from 2D** (it can be perceived from 2D images).
      - [网络] 一个3D auto-encoder，中间得到latent embedding A; 一个2D convnet，输出另一个latent embedding B。Loss包含3D reconstruction loss和两个embedding之间的距离

   - [评价] Pretrain on Shapenet; Linear SVM on ModelNet40
2. Vconv-dae: Deep volumetric shape learning without object labels. **[Context|Completion]**
   - [要点] Given shapes with different poses, to learn the shape distributions of various classes by **predicting the missing voxels from the rest**.
   - 第一个提出非监督学习3D特征的工作(concurrent)；24x24x24的Volumetric Grid，部分点设置为0，**auto encoder** output要预测
   - [评价] Pretrain on ModelNet; Linear SVM on ModelNet40
