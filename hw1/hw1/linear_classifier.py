import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.normal(mean=0, std=weight_std, size=(self.n_features, self.n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred = class_scores.argmax(dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = sum(y == y_pred) / y.shape[0]
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # Train set
            train_batches_loss = []
            train_batches_acc = []
            for x_train_batch, y_train_batch in dl_train:
                y_pred_batch, class_scores_batch = self.predict(x=x_train_batch)
                train_batch_loss = loss_fn(x=x_train_batch,
                                           y=y_train_batch,
                                           x_scores=class_scores_batch,
                                           y_predicted=y_pred_batch)
                train_batches_loss.append(train_batch_loss)
                train_batches_acc.append(self.evaluate_accuracy(y=y_train_batch, y_pred=y_pred_batch))
                grad = loss_fn.grad() + (weight_decay * self.weights)
                self.weights = self.weights - (learn_rate * grad)
            train_epoch_loss = torch.mean(torch.tensor(train_batches_loss))
            train_epoch_acc = torch.mean(torch.tensor(train_batches_acc))
            train_res.loss.append(train_epoch_loss)
            train_res.accuracy.append(train_epoch_acc)

            # Valid set
            valid_batches_loss = []
            valid_batches_acc = []
            for x_valid_batch, y_valid_batch in dl_valid:
                y_pred_batch, class_scores_batch = self.predict(x=x_valid_batch)
                valid_batch_loss = loss_fn(x=x_valid_batch,
                                           y=y_valid_batch,
                                           x_scores=class_scores_batch,
                                           y_predicted=y_pred_batch)
                valid_batch_loss += (weight_decay * torch.norm(self.weights, p=2) / 2)
                valid_batches_loss.append(valid_batch_loss)
                valid_batches_acc.append(self.evaluate_accuracy(y=y_valid_batch, y_pred=y_pred_batch))
            valid_epoch_loss = torch.mean(torch.tensor(valid_batches_loss))
            valid_epoch_acc = torch.mean(torch.tensor(valid_batches_acc))
            valid_res.loss.append(valid_epoch_loss)
            valid_res.accuracy.append(valid_epoch_acc)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if has_bias:
            weights_ = self.weights[1:, :]
        else:
            weights_ = self.weights
        n_classes = weights_.shape[1]
        w_images = weights_.transpose(0, 1).view(n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['learn_rate'] = 0.01
    hp['weight_std'] = 0.01
    hp['weight_decay'] = 0.001
    # ========================

    return hp
