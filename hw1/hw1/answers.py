r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False - in-sample error is the error rate we get on the same data set we used to train our model/predictor. We don't use the test set for that, the test set is the out-of-sample error.
2. True (partially) - That's true while we don't care about the samples timestamps, because we don't want to decide our test set to be specific values we can choose it randomly. In contrast, if time is issue and our features use information from the future the answer is False because we should keep the test as the later samples (not to raise a leakage error).
3. True - Cross-validation is useful for choosing the pipeline's hyperparameters, while the test set designed for testing our model's generalization.
4. False - We use the test set for that. 
"""

part1_q2 = r"""
My friend's approach isn't justified! We shouldn't tune our hyperparameters on the test-set, we should generate validation set for that.
This approach disable our ability to validate our model on a new clean dataset.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Yes, increasing k leads to generalization improvement of unseen data.
When k is larger, we are taking into account more neighbours in each prediction hence increase classes variance.
It's improving the models generalization up to the point we covered all the data-points and all our prediction are from the same class.
"""

part2_q2 = r"""
1. If we evaluate our models and choose the best one using the training-set we can accidentally choose over-fitted model
with perfect fit for the training data, but when new data from slightly different distribution arrive it's predictions 
accuracy will drop immediately.
2. It is statistically possible to get a test-set which is quite similar to the training data, so the model can't 
actually evaluate it's generalization. When using CV we are decreasing this probability. 
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
$\Delta$ is related to how much we allow some errors to occur, but so as the weights regularization.
If we will change $\Delta$, so does the weights will change accordingly and we will get the same loss.
"""

part3_q2 = r"""
1) The model learns the labels position and then fits the digit which resembles the sample the most.
We can explain the classification error where the model classified a 2 digit as 7 because the model 7's label isn't
clear enough and looks similar to 2.
2) In KNN due to it's nature, the predicted value is based on close neighbors classes, and here in the linear
classifier, it's based on the previous samples of the same class. 
"""

part3_q3 = r"""
1) The learning rate we choose looks between Good and High.
Too high learning rate graph should look a small decrease at the beginning and than high jump.
Too low learning rate graph should look like slowly convergence over epochs.
2) Based on the accuracies graph it's looks like our model is slightly overfitted to the training set, because
their is a quite big difference between our training and validation scores.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
We can see in the final plot that the residuals are closer to 0 comparing to the top 5 features used before, means that the prediction error is closer to zero. We can conclude that the cross-validation method shown in the final plot produces better results than linear regression over the top-5 features.
"""

part4_q2 = r"""
1. Yes, the non linear features were added by us, and they are still used as the regular features with the weights matrix.
2. Yes, By moving to higher dimension (need to be careful not too high for complexity manners), we can engineer any non linear relation between features and use it in linear regression.
3. It still will be an hyperplane, the engineering of the new features will cause the parameters to change and therefore to change the decision boundary.
"""

part4_q3 = r"""
1. Using logarithmic scale instead of linear scale helps us see more information under smaller area. One can think of it as it 'shrinks' the linear space, so more data can be reviewed.
2. number of folds = 3; degree range = 1,2,3; lambda_range=20;
In conclusion- the model was fitted 180 times.
"""

# ==============
