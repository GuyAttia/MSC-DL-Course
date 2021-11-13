r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:
1. False - in-sample error is the error rate we get on the same data set we used to train our model/predictor. We don't use the test set for that, the test set is the out-of-sample error.
2. True (partially) - That's true while we don't care about the samples timestamps, because we don't want to decide our test set to be specific values we can choose it randomly. In contrast, if time is issue and our features use information from the future the answer is False because we should keep the test as the later samples (not to raise a leakage error).
3. True - Cross-validation is useful for choosing the pipeline's hyperparameters, while the test set designed for testing our model's generalization.
4. False - We use the test set for that. 
**
"""

part1_q2 = r"""
**Your answer:
My friend's approach isn't justified! We shouldn't tune our hyperparameters on the test-set, we should generate validation set for that.
This approach disable our ability to validate our model on a new clean dataset.
**
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

part3_q3 = r"""
**Your answer:**
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
"""

part4_q2 = r"""
**Your answer:**
"""

part4_q3 = r"""
**Your answer:**
"""

# ==============
