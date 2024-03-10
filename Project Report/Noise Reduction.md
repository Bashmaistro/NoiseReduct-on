

Emirhan Yıldız1

**1** Computer Engineering, Engineering, Eskişehir Technical University, Eskişehir, Türkiye.

ABSTRACT

Noise cancellation, one of the most challenging problems, is one of the only tools that can eliminate the financial inadequacies that may occur, especially in increasing remote work, education, etc. During the project, we tried to make a noise cancellation model that could be as simple and powerful as possible by utilizing sound waves according to the hardware power we have.

**Keywords:** Noice Cancelation,Model,Wave.

1. INTRODUCTION

Noise removal is a process that deals with removing noise from a signal [1]. This process can be used for both images and audios but here the focus is on noise removal for audios. All the signals have characteristics that make them vulnerable to noise. This noise could be white noise [2] or random noise with an even frequency distribution or frequency- dependent noise. The noise could be in the form of but not limited to hum and hiss, drillings, air conditioner, or car horn. If the audio has an unwanted and unpleasant sound then it becomes difficult to use it further for transmission, recognition, or improvement. Considering the various edge devices like phones, laptops, conferencing systems, etc, traditional noise suppression methods have been implemented. Since it is an edge device such an approach seems intuitive. It captures the user’s voice in the first place and then device filters are used to noise out the captured voice. Later the output is saved. Another common and consistent solution for controlling or suppressing background noise is sound masking. It adds the white noise commonly referred to as the pink noise into an environment to mask the unwanted sound. But this method is not an active noise control, that is it only reduces or eliminates the perception of noise [3]. Our project aims to provide noise cancellation in as real-time as possible with a trained CNN AutoEncoder. Convolutional neural network (CNN) is a regularized type of feed-forward neural network that learns feature engineering by itself via filters (or kernel) optimization.[4]

2. METHODOLOGY

Deep learning is one of the only things that underpins our project, and thanks to deep learning, and of course the GPUs and TPUs in our computers, we are able to capture the connections between images. Our project will be based on CNNs, one of the most important elements of deep learning. CNN is a deep learning model that has proven itself in capturing and learning images, patterns, details. analytical device), supply the name and location of the manufacturer. Give the model number for equipment used. Results should be stated concisely and without interpretation. Focus on the rigorously supported aspects of your study. Carefully differentiate the results of your study from data obtained from other sources. Interpret your results, relate them to the results of previous research, and discuss the implications of your results or interpretations. This variety of neural networks are formed by three types of

layers: convolutional layers, pooling layers, and fully connected layers. Firstly, filters of the convolutional layers are convolved to the input in a sliding window resulting in a feature map summarizing the presence of detected features in the input. The feature maps are processed through an activation layer as rectified linear unit (ReLU), exponential linear unit (ELU), leaky ReLU, hyperbolic tangent (tanh), etc. The complexity of the learned features increases across the depth of the layers. The following layer is usually a pooling layer, frequently a Max-pooling layer, that reduces the computational costs as it ensures the shift invariance which is a common feature of the architecture. Depending on the applications of the DL model, the previous layers are followed by a fully connected layer for a classification or regression task. In case an overfitting phenomenon is noticed, several methods of regularization could be employed. A widely used regularization method is Dropout . On the other hand, autoencoders are the most common architecture used for noise reduction purposes. They learn an identity function reconstructing the clean version of its input in the outputs by tuning the encoder and decoder fully connected layer’s parameters, minimizing the reconstruction error. It starts by compacting the input into a latent space, which is leveraged to reconstruct the input at the output during the decompression process.

2.1. CNN Auto Encoder

![[Pasted image 20240311012630.png]]

Autoencoders actually look like two CNNs added end to end. It consists of two parts, Encoder and Decoder (1). The Encoder part compresses the input into a lower dimensional representation. This representation contains features that highlight important features of the input. The decoder part tries to make the representation from the encoder resemble the desired output in the best possible way. When creating an autoencoder, the parameters between the layers are very important because the output result is not a classification like in other CNNs and therefore the output size and the input size must be the same.

2.2 Time Series

The time series of a sound contains the value of the audio signal within 1 s according to the sampling frequency of that sound.(2) There are different ways to obtain these time series. These are Short Time Fourier Transform (STFT), Fast Fourier Transform (FTF), etc. [5] In the light of this information, we can examine the behaviour of the sound and learn where and what data it reaches.

![[Pasted image 20240311012708.png]]

2.3 Dataset

The entire dataset consists of 20,000 noisy sounds and their corresponding clean sounds. Then, in order to eliminate the problem of mismatch in the inputs and outputs, each sound was divided by 1 s each at a sampling frequency of 22050 hz, resulting in 26,000 noisy sounds and their corresponding clean sounds.

**3. IMPLEMENTATION**

In the implementation part, the goal will be to train a cnn model using our already existing dataset[6] and then use this cnn model as a noise canceller. the implementation consists of 2 parts: preprocessing the data, generating the model.

3.1 Preprocessing of Data

We have a very rich data set, but we need to make it possible to analyze the data from the sounds in this data set. In our first attempt, we aimed to produce spectrograms from sound data and train our model because of these spectrograms, but this goal was abandoned as it was impossible to convert the image files produced by the model back to sound. Later, it seemed logical to use the sound time series value that we used when producing the spectrogram, and we decided to use it. However, this has revealed another problem for us. Another problem we had to solve was that each sound had a different length. To solve this problem, we aimed to bring the sound to a fixed sampling frequency and then divide each sound into 1-second sounds. As a result, we have an array of size 1x22050, each index containing the sound value of that moment. We have completed the preprocessing part of our sound by applying these steps one by one to our clean and noisy sounds.

**3.1 Generating Model**

The model creation phase was the most challenging part of the job. First, we started with a simple CNN model. It was a simple CNN AutoEncoder model that had 4 layers and used Relu as the activation layer. We trained our model to test it using 1000 data points and our results were disgraceful. Our input (3), our model's output (4) and our target (5) were very unrelated to each other.

![[Pasted image 20240311012733.png]]
                                   *Figure 3*
 ![[Pasted image 20240311012852.png]]
                                   *Figure 4*             
 ![[Pasted image 20240311012914.png]]
                                 *Figure 5*      

![[Pasted image 20240311012959.png]]
                 *Figure 6*

![[Pasted image 20240311013015.png]]
                *Figure 7*

The reason our output was so bad was that there was almost no change in the negative data distributions of our model, and the solution to this was in the activation layer we used. We were using ReLu(6) as the activation layer. But as can be seen from the graph, ReLu ignored our negative data, so we used Leaky ReLu(7) instead and made our model more complex. And the results (8) were slightly satisfactory. Although we did not show a huge improvement graphically, we were getting positive results when we listened to the sound.

![[Pasted image 20240311013209.png]]
                                  *Figure 8*

But as you can see from the output, our model was much more similar to the input data than the output data. After much research, we found that our model was too busy decoding input attributes and could not even spare time to decode the output data, and therefore our model predictions were more similar to inputs than outputs. We added Dropout and MaxPooling layers to our model and our model took its current final form. The dropout layer prevents our model from trying too hard to decipher the input data and prevents it from giving us better results. The MaxPooling layer summarized the extracted features and significantly reduced our calculation time. And our output with the last version of our model (10) was like this. (9) and don’t consider the y label of the table.

![[Pasted image 20240311013335.png]]
  
**4 RESULT**

As a result, the outputs we received are unfortunately not satisfactory enough. It can capture noisy sounds and silence them quite well, but while doing this, it also silences the speech sound. Even if it is slightly, much better results could have been obtained by using more data, but the hardware we have has a data count of 5 thousand data. It made us stay.               

**5. AI DECLERATION**

During the project, help and information was received regarding error notifications encountered in artificial intelligence applications, how to perform basic functions in Python, and what features of sound are available.

While preparing the project, I did not take any action that could be considered as plagiarism

[1]Noisereduction;URL" https://en.wikipedia.org/wiki/Noise_reduction".

[2] White noise; https://en.wikipedia.org/wiki/White_noise.

[3]Soundmasking; https://en.wikipedia.org/wiki/Sound_masking.

[4] CNN;URL https://en.wikipedia.org/wiki/Convolutional_neural_network

[5] Time Series; https://librosa.org/doc/main/core.html#audio-loading.

[6] https://datashare.ed.ac.uk/handle/10283/2791