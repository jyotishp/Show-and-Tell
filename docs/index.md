# Introduction
Visual Understanding of images and expressing them in our language has always been a part of human lives. This has been a difficult problem to be done by a machine, using traditional methods in AI. This task is significantly harder than the
object recognition tasks, as the semantic knowledge gained in viewing a particular image must be utilized to generate a meaningful sentence in a required language with proper grammar.

Recent Advances in AI, with the development of Deep-CNNs and the rise of Recurrent Neural Networks in language processing, now provides the tools to design an effective Neural Image Caption Generator.

In this Project, we have explored various current state-of-the-art ways to implement the Image Caption Generator.

# Background
## Theory
The main framework of the Image Caption Generator can be abstracted into two units. The first one being an image encoding CNN, later followed a Recurrent Neural Network decoder. This has been inspired from machine translation neural networks, where an encoder-decoder RNN pair is utilized to translate sentences. In contrast, here instead of an RNN language encoder, a Deep CNN is used to encode the input
image, which is later decoded by the LSTM decoder network into the language of choice.

## Deep CNN Encoder - Inception V3
We used the InceptionV3 CNN as the image encoder. Inception V3 is the CNN architecture that won the Imagenet classification challenge in 2014.

We choose it over other CNN architectures because it is widely used and a pretrained model trained on the Imagenet dataset is available in Keras The Inception V3 network is a combination of smaller simpler model called Inception modules connected in series with fully connected dense layer in the end.

An inception block has convolutional layers in parallel with multiple kernel sizes to mitigate the problem of selecting a specific kernel size for layer.

## RNN Decoder - LSTM
The output of an RNN at a particular timestep is expressed as a function of its hidden state, ht and the input at that time step. where the hidden state is the output of the RNN during the previous time step. This hidden state is used as an input to the successive time steps as $h_t+1 = f(h_t, x_t)$.
