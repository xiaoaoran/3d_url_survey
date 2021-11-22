## 2016 ECCV
1. Learning a predictable and generative vector representation for objects
   - [要点] A good vector representation of an object should be **generative in 3D** (it can produce new 3D objects) 
   as well as be **predictable from 2D** (it can be perceived from 2D images).
      - [网络] 一个3D auto-encoder，中间得到latent embedding A; 一个2D convnet，输出另一个latent embedding B。Loss包含3D reconstruction loss和两个embedding之间的距离

   - [评价] Pretrain on Shapenet; Linear SVM on ModelNet40
