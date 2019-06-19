# Dense-Net-pixel-crack-detection

1. The DenseNet implementation is based on the https://github.com/seasonyc/densenet for the DenseNet implementation, except the classfication layer is a multi-label sigmod layer.

2. The preprocessing file process the AigleRN dataset (Amhaz et al., 2016), Retrieved from https://www.irit.fr/∼Sylvie.Chambon/AigleRN.html. 

3. The network should be used as a structured prediction of crack images.

4. The crack detector inputs 27 x 27 patches and output 5 x 5 patch output. Binary result is obtained by averaging all sliding windows.
