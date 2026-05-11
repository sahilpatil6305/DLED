import torch
import torch.nn as nn

from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="edl_loss")
class EvidenceLoss(nn.Module):
    """Evidential Deep Learning (EDL) loss for the DLED detector.

    Each branch (spatial / frequency) is supervised independently through
    `separate_forward`, optionally adding the AvU regularizer and a KL
    divergence to a uniform Dirichlet prior.
    """

    def __init__(self, cfg, num_cls, annealing_start=0.01, annealing_step=10):
        super().__init__()
        self.num_cls = num_cls
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step
        self.annealing_method = 'exp'
        self.with_kldiv = cfg['kl_reg']
        self.with_avuloss = cfg['avu_reg']
        self.loss_type = cfg.get('evi_func') or 'mse'
        print("evidence loss type: ", self.loss_type)
        self.eps = 1e-7

    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_cls], dtype=torch.float32).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    def compute_annealing_coef(self, epoch_num, total_epoch):
        if self.annealing_method == 'step':
            return torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
            )
        if self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            return annealing_start * torch.exp(
                -torch.log(annealing_start) / total_epoch * epoch_num
            )
        raise NotImplementedError(f"Unknown annealing method: {self.annealing_method}")

    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.mean(torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True))
        loglikelihood_var = torch.mean(
            torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        )
        return loglikelihood_err, loglikelihood_var

    def avuloss(self, alpha, S, target, annealing_coef):
        # Accuracy-vs-Uncertainty regularizer.
        pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
        uncertainty = self.num_cls / S
        if torch.any(S <= 0):
            raise ValueError(f"S <= 0 leads to inf uncertainty. min S = {torch.min(S)}")

        if uncertainty.dtype == torch.float16:
            eps = torch.tensor(1e-3, dtype=uncertainty.dtype, device=uncertainty.device)
        else:
            eps = torch.tensor(self.eps, dtype=uncertainty.dtype, device=uncertainty.device)
        new_uncertainty = torch.clamp(uncertainty, min=eps, max=1 - eps)

        acc_match = torch.reshape(torch.eq(pred_cls, target.unsqueeze(1)).float(), (-1, 1))
        acc_uncertain = -torch.log(1 - new_uncertainty)
        inacc_certain = -torch.log(new_uncertainty)
        return (
            annealing_coef * acc_match * acc_uncertain
            + (1 - annealing_coef) * (1 - acc_match) * inacc_certain
        )

    def mse_loss(self, y, alpha, annealing_coef=0, func=None, target=None):
        losses = {}
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(y, alpha)
        losses['cls_loss'] = loglikelihood_err
        losses['var_loss'] = loglikelihood_var

        S = torch.sum(alpha, dim=1, keepdim=True)
        if self.with_avuloss:
            losses['avu_loss'] = torch.mean(self.avuloss(alpha, S, target, annealing_coef))
        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            losses['kl_loss'] = torch.mean(annealing_coef * self.kl_divergence(kl_alpha))
        return losses

    def edl_loss(self, y, alpha, annealing_coef=0, func=torch.log, target=None):
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)
        losses['cls_loss'] = torch.mean(torch.sum(y * (func(S) - func(alpha)), dim=1))
        if self.with_avuloss:
            losses['avu_loss'] = torch.mean(self.avuloss(alpha, S, target, annealing_coef))
        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            losses['kl_loss'] = torch.mean(annealing_coef * self.kl_divergence(kl_alpha))
        return losses

    def get_loss_func(self):
        if self.loss_type == 'mse':
            return self.mse_loss, None
        if self.loss_type == 'log':
            return self.edl_loss, torch.log
        if self.loss_type == 'digamma':
            return self.edl_loss, torch.digamma
        raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def separate_forward(self, pred_alpha, target, epoch, total_epoch, isTraining=True):
        """Compute the EDL loss for a single branch (spatial or frequency).

        Args:
            pred_alpha: Dirichlet evidence + 1 for the branch, shape (B, K).
            target: integer class labels, shape (B,).
            epoch / total_epoch: used to anneal the AvU / KL terms.
            isTraining: when False, the annealing coefficient is 0.
        """
        annealing_coef = 0.0
        if isTraining and (self.with_avuloss or self.with_kldiv):
            annealing_coef = self.compute_annealing_coef(epoch, total_epoch)

        loss, func = self.get_loss_func()
        y = torch.zeros(pred_alpha.size()).to(pred_alpha.device, non_blocking=True)
        y[torch.arange(pred_alpha.size(0)), target] = 1
        return loss(y, pred_alpha, annealing_coef=annealing_coef, func=func, target=target)
