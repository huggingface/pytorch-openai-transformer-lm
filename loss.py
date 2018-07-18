import torch

class MultipleChoiceLossCompute:
    "A Loss compute and train function for multiple choice tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, :, 1:, 0].contiguous().view(-1)  # Shape: 252
            M = M.view(-1, M.size(2))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class ClassificationLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, 1:, 0].contiguous().view(-1)
            M         = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

# TODO Implement a LossCompute class for similiraty tasks.
