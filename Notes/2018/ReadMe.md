## 2018 ICML
1. ppp
   - [要点] The first deep generative models for point clouds.
   - [网络] 
      - First train an AutoEncoder to learn a latent representation and then train a generative model in that fixed latent space. 
      The GANs are trained in the latent space.
      - GAN是随机采样，generator生成embedding，discriminator判断AE的embedding和generator得到的embedding；最后生成点云是利用AE的decoder.
   - [评价] ShapeNet ptr-train; Linear SVM on ModelNet
