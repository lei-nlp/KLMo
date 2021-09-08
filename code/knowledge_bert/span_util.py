import copy
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import math
import numpy
import torch


logger = logging.getLogger(__name__)

T = TypeVar("T")

def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
        
def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices
    
    
def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask

    
def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/master/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.
    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets

if __name__ == '__main__':
    #tensor = torch.rand([5, 7, 3])
    #tensor[1, 6:, :] = 0
    #tensor[3, 2:, :] = 0
    tensor = torch.Tensor([[[0.2899, 0.6183, 0.8659],
         [0.5920, 0.7464, 0.8796],
         [0.4775, 0.0725, 0.1621],
         [0.1994, 0.9321, 0.4750],
         [0.1233, 0.5336, 0.8974],
         [0.2603, 0.4651, 0.4888],
         [0.8069, 0.5710, 0.7957]],

        [[0.5480, 0.0024, 0.6137],
         [0.8250, 0.9700, 0.5803],
         [0.9021, 0.0693, 0.6192],
         [0.4126, 0.6733, 0.0626],
         [0.3685, 0.8444, 0.8789],
         [0.0018, 0.0776, 0.8710],
         [0.0000, 0.0000, 0.0000]],

        [[0.2961, 0.6767, 0.5523],
         [0.2448, 0.2610, 0.1840],
         [0.6098, 0.0269, 0.8767],
         [0.8828, 0.1978, 0.7657],
         [0.9226, 0.1597, 0.0929],
         [0.6258, 0.2483, 0.5089],
         [0.1221, 0.3645, 0.1428]],

        [[0.3966, 0.0705, 0.2829],
         [0.7761, 0.8621, 0.9993],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000]],

        [[0.1549, 0.6177, 0.2002],
         [0.9876, 0.5428, 0.8893],
         [0.7292, 0.1877, 0.1962],
         [0.1768, 0.4164, 0.3980],
         [0.9647, 0.3389, 0.2749],
         [0.1533, 0.1821, 0.0544],
         [0.1412, 0.0954, 0.0329]]])
    '''
    left = torch.LongTensor([[1,5],[0,-1],[0,-1],[1,-1],[2,-1]])
    right = torch.LongTensor([[3,6],[3,-1],[2,-1],[4,-1],[3,-1]])
    left_mask = left.gt(-1)
    print (left_mask)
    left_mask=left_mask.type_as(left)
    print (left_mask)
    #right_mask = right.gt(-1)
    #print (right_mask)
    left = left + left_mask
    print (left)
    right = right - left_mask
    print (right)
    exit()
    left = left.unsqueeze(-1)
    right = right.unsqueeze(-1)
    span = torch.cat((left, right), -1)
    #print (tensor)
    #print (left.size())
    #print (span.size())
    print (span)
    '''
    span = torch.LongTensor([[[1,3],[5,6]],[[0,3],[-1,-1]],[[0,2],[-1,-1]],[[1,4],[-1,-1]],[[2,3],[-1,-1]]])
    #span = torch.LongTensor([[[1,3],[5,6]],[[0,3],[0,0]],[[0,2],[0,0]],[[1,4],[0,0]],[[2,3],[0,0]]])
    print (tensor)
    print (span)
    concat_output, span_mask = batched_span_select(tensor, span)
    print (concat_output)
    #print (concat_output.dim())
    print (span_mask)
    #print (span_mask.dim())
    span_widths = span_mask.sum(dim=-1)
    print (span_widths.size())
    span_widths = span_widths.unsqueeze(-1)
    span_mask = span_mask.unsqueeze(-1)
    print (span_mask.size())
    #print (span_mask.dim())
    span_embs = concat_output * span_mask
    print (span_embs)
    span_embs = span_embs.sum(dim=2)#, keepdim=True)
    span_embs = span_embs / (span_widths + 1e-9)
    print (span_embs)
    #tensor_c = copy.deepcopy(tensor)
    tensor_c = tensor.clone()
    print (tensor_c.size())
    for c in range(5):
        for i in range(2):
            tensor_c[c,span[c][i][0]:span[c][i][1]+1,:] = span_embs[c,i,:]
    print (tensor_c)
    print (tensor)
    print (tensor_c[0,-1:0,:])
