3D Convolution Networks for Movement Decoding

I used neural activity across many electrodes throughout the cortical sheet unfolding over time to predict movement directions (categorical variable) or movement speed (continuous variable). This is in the same spirit of using video clips to label content. In the process, I compared among many architectures of networks with different combinations 3D or 2D convolutions, in combination of recurrent networks, with/without dropouts. Additionally, I chekced whether having the correct spatial relationship among electrodes are crucial for decoding. All implementations were executed using GPU clusters of HPC on Uchicago Midway. Next step is to extract the spatio-temporal features important for determining movements using deConvNet.


c3d network: 
Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 4489-4497).
https://openaccess.thecvf.com/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf


Implementations are inspired and adapted from https://github.com/harvitronix/five-video-classification-methods
