r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "cowards die many times before their deaths"
    temperature = 3e-5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
The main reason we split the text into sequences instead of training on the entire text is mainly to avoid overfitting.
When we split the data into different sequences, we allow it to see multiple variations of the text and combinations of characters.
If we were using the entire text at once, the model could tend to memorize the specific text and just overfit to it.
"""

part1_q2 = r"""
The output text length depends on the hidden state size.
The hidden state doesn't depend on the input size, what may cause the output to be of a different size.
"""

part1_q3 = r"""
Because we pass the hidden state between batches during training, we must keep the original text orientation.
If we would randomize the batches, the model couldn't learn from moving from one batch to another and only inside each batch- 
What would cause a decrease in the performance.
"""

part1_q4 = r"""
1. In order for the softmax to output a more detrmined decision, and less uniform distribution.
2. The distribution out of the hot softmax when T is very high will get closer to uniform, what will cause the decision to be less definite,
and will affect the learning proccess badly.
3. The distribution out of the hot softmax when T is very low will get much more spiky- the characters with high probability will get more high scores while the others wil get lower scores. This will help the model to be more determined regarding it's decisions during training. 
"""
# ==============

