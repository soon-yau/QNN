# QNN
## Quantized Neural Network


Traditionally, deep learning uses single precision floating point (float32) data type. Recent researches show that using lower precision say half precision floating point (float16) or even unsigned 8-bit integer (uint8) doesn’t impact the neural network accuracy significantly. Although there are now tonnes of tutorials in using machine learning framework like Tensorflow but I couldn't find many tutorials on Tensorflow Lite or quantization. Therefore, I decided to write some tutorials to explains quantization and fast inference.

You don’t need to have prior knowledge in quantization but I do expect you be familiar with Tensorflow and basic of deep neural network. In this tutorials I will use TensorflowLite in Tensorflow 1.10 and Python 3.
