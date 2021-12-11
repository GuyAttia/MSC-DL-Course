r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. Since there are 128 samples, were each is using a Jacobian (weights vector) with the size of W
[1024,2048], the shape of the output of the Jacobian tensor f the output layer will be- [128, 1024,2048]. 
2. Using 32 precision means that each number of the output tensor is represented by 32 bits. 
Therefore, the total GPU memory required is- 32 X 128 X 1024 X 2048 = 8589934592[bit] = 1[GB]
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.1, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = 0.01, 0.03, 0.005, 0, 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-3
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
We can clearly see that without dropout we got a situation of overfitting the training set- from epoch 5 only the train set accuracy increasing while accuracy on test set is stuck.
With dropout, ther is a more wanted picture, where the test set accuracy keeps increasing while the training set also increasing (maybe there is a slightly noticable plato of the test accuracy at the last epochs).
2.
We can see that too large dropout can give negative affect on the training procedure. In our case, the using of dropout=0.8 gave lower accuracy both on the train and test sets. 
"""

part2_q2 = r"""
The cross-entropy loss depends on the softmax function, maeaning that it is not only determined by if the classifier was right or wrong (binary), it is also affected by how much the prediction was positive and close to the right classification. 

So, there can be a situation where the accuracy is increasing because more and more instances are classified correctly, but the desicion of the classifier is more ambiguous, what will increase also the loss value.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1.
A conv layer number of parameters is -
kernel_hight X kernel_width X filters number in previous layer (+1) X filterss_number in current layer

So, in basic Residual Block- 
conv layer1&2 each have- 64 X 64 X 4 X 3 = 36928
Together - 36928 X 2 = 73856

In bottleneck Residual Block-
conv layer 1- 64 X 256 X 2 X 1 = 16448
conv layer 2- 64 X 64 X 4 X 3 = 36928
conv layer 3- 256 X 64 X 2 X 1 = 16640
Together- 36928 + 16448 + 16640 = 70016

2.
The number of floating point operations required to compute an output is-
number of parameters of the block (calculated above) X width X height (of each output feature map.
Basic Residual Block- 73856 X width X length
Bottleneck Residual Block- 70016 X width X length

3.
1) Combine the input spatially within feature maps-
The regular Residual Block has higher ability to combine the input within the feature map, since each output feature depends on more features (output of eachlayer depends on 9 features) than in bottleneck block (output of the first and last layers depends only on 1 feature).
2) Combine the input spatially across feature maps-
In this case the Bottlenack Residual Blovk has higher ability to combine the input across the feature map, since it projects the input to a smaller channel number as the regular block and then projects it back to the original size.
The regular Residual Block acts as a convolutional layer, thus changes the input feature map.
"""

part3_q2 = r"""
1.
We can see that in the case of K=32- the best results appeared with L=2,4 - for deeper net with L=8,16 there was no training at all.In the case of K=64-best results are still L=2,4 - in this case also L=8 got measurable results (but after twice epochs as with L=2,4).
This can be explained by the affect of vanishing gradients as the net gets deeper, what causes it's learning ability to be poor.
We can see that increasing the number of filters in each layer helps the training procedure of deeper networs.
2.
In case of L=16 the network seems untrainable. 
We can help it b using ResBlocks or increasing the number of filters in each layer (as we saw for L=8 in the previous section).
"""

part3_q3 = r"""
In experiment 1.2 we can see that L=4 got us better results that L=2,8.
We can observe that for each L, there is different K that fits it the most.
We can conclude from this section that using deeper and high filters number doesn't nessacarily gives us the best results.
"""

part3_q4 = r"""
We can see that we got better results for L=1,2 that for L=3,4.
This is for the same reason we saw before- as the network gets deeper the training gets harder (and sometimes omossible).
"""

part3_q5 = r"""
With fixed K- we can clearly see that the accuracy gets lower as we increase the depth of the network.
Different filters per layer (with skip connection)-
With different filters number in each block, we can see that the training proccess is better even for deep networks, although we still see slightly better results for L=2,4 than L=8.
We can parise the skip connection in this case that helps in regularizing the network and helps deeper layers to not reduce their gradients almost to zero.
"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
