"""Util functions for various distributions (DMoL, MoG, etc.)"""

import torch
import torch.nn.functional as F


def ScaleInput(x, num_classes):
    """Scales x into [-1, 1], assuming it is integer in [0, num_classes - 1]."""
    return 2 * (x.float() / (num_classes - 1)) - 1


def discretized_mixture_of_logistics_logprobs(dmol_params,
                                              x,
                                              num_classes,
                                              num_mixtures,
                                              scale_input=False):
    """Computes DMoL for all mixtures on this batch of data.

    Reference: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    Args:
        dmol_params: Contains parameters for dmol distribution for each sample.
            Size = (batch_size, num_mixtures*3).
            First 1/3 of axis 1 contains the log_probs for each mixture,
            the next 1/3 contains means, and the last 1/3 contains log_scales.
        x: Data to train/evaluate on. Size = (batch_size,).
        num_classes: Total number of distinct values for this column.
        num_mixtures: Number of dmol mixtures to use.
        scale_input: If true, scales input to domain [-1, 1].

    Returns:
        The log probs for each sample for each mixture.
        Output size is [batch_size, num_mixtures].
    """

    if scale_input:
        x = ScaleInput(x, num_classes)

    assert dmol_params.size()[1] == num_mixtures * 3

    # Change size of data from [bs] to [bs, num_mixtures] by repeating.
    x_new = x.unsqueeze(1).repeat(1, num_mixtures)
    assert x_new.size()[0] == x.size()[0]
    assert x_new.size()[1] == num_mixtures

    mixture_weights, means, log_scales = torch.chunk(dmol_params, 3, dim=-1)
    log_scales = torch.clamp(log_scales, min=-7.)

    centered_x = x_new - means
    inv_stdv = torch.exp(-log_scales)
    boundary_val = 0.5 if not scale_input else 1. / (num_classes - 1)
    plus_in = inv_stdv * (centered_x + boundary_val)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - boundary_val)
    cdf_min = torch.sigmoid(min_in)

    cdf_delta = cdf_plus - cdf_min
    log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_cdf_min = -F.softplus(min_in)

    min_val = 0.001 if not scale_input else -0.999
    max_val = num_classes - 1 - 1e-3 if not scale_input else 0.999

    x_log_probs = torch.where(
        x_new < min_val, log_cdf_plus,
        torch.where(x_new > max_val, log_cdf_min, log_cdf_delta))
    pi_log_probs = F.log_softmax(mixture_weights, dim=-1)

    log_probs = x_log_probs + pi_log_probs
    return log_probs


def dmol_query(dmol_params, x, num_classes, num_mixtures, scale_input=False):
    """Returns the log probability for entire batch of data."""
    log_probs = discretized_mixture_of_logistics_logprobs(
        dmol_params, x, num_classes, num_mixtures, scale_input)
    # Sum of probs for each mixture. Output size is [batch_size,].
    return torch.logsumexp(log_probs, -1)


def dmol_loss(dmol_params, x, num_classes, num_mixtures, scale_input=False):
    """Returns the nll for entire batch of data."""
    return -dmol_query(dmol_params, x, num_classes, num_mixtures, scale_input)
