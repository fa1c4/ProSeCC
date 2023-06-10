from tokenize import Single
import torch
from scipy.stats import entropy
from math import log2, floor
from typing import List, Dict, Optional, Union
from classes import SingleEncodeStepOutput
import queue

import config

device = config.device


def find_nearest(anum: float, probs: List[float]) -> int:
    # Returns index_idx (index of indices)
    up = len(probs) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index_idx = int((up + bottom) / 2)
        if probs[index_idx] < anum:
            up = index_idx
        elif probs[index_idx] > anum:
            bottom = index_idx
        else:
            return index_idx
    if up - bottom == 1:
        if probs[bottom] - anum < anum - probs[up]:
            index_idx = bottom
        else:
            index_idx = up
    return index_idx


class ADGNode:
    final_probs = []
    final_indices = []

    # node of tree
    def __init__(self,
                 probs: Union[torch.Tensor, List[float]],
                 indices: Union[torch.Tensor, List[int]],
                 multiplier: float = 1.0) -> None:
        self.probs = probs  # Guaranteed to sort in descending order of probability
        if type(self.probs) == torch.Tensor:
            self.probs = self.probs.tolist()
        self.indices = indices
        if type(self.indices) == torch.Tensor:
            self.indices = self.indices.tolist()

        self.probs_sum = sum(self.probs)
        self.children = []
        self.multiplier = multiplier
        # self.grouping()

    def grouping(self) -> None:
        probs = self.probs[:]
        indices = self.indices[:]

        if self.is_leaf():
            ADGNode.final_probs.extend(list(x * self.multiplier for x in self.probs))
            ADGNode.final_indices.extend(self.indices)
            return
            # print()

        prob_max = max(probs)
        probs_sum = self.probs_sum
        num_groups = 2**floor(-log2(prob_max / probs_sum))
        for i in range(num_groups - 1):
            probs_child_i = []
            indices_child_i = []

            mean_probs_sum_per_group = sum(probs) / (num_groups - i)
            probs_child_i.append(probs[0])
            indices_child_i.append(indices[0])
            del probs[0]
            del indices[0]
            while True:
                delta = mean_probs_sum_per_group - sum(probs_child_i)
                if delta <= 0:
                    break
                index_idx = find_nearest(delta, probs)
                if probs[index_idx] - delta < delta:
                    probs_child_i.append(probs[index_idx])
                    indices_child_i.append(indices[index_idx])
                    del probs[index_idx]
                    del indices[index_idx]
                else:
                    break
            probs_child_i = torch.tensor(probs_child_i, device=device)
            indices_child_i = torch.tensor(indices_child_i, device=device)

            # sorting
            probs_child_i, indices_idx = probs_child_i.sort(descending=True)
            indices_child_i = indices_child_i[indices_idx]

            probs_child_i = probs_child_i.tolist()
            indices_child_i = indices_child_i.tolist()

            self.children.append(
                ADGNode(probs_child_i, indices_child_i, self.multiplier / sum(probs_child_i) * (probs_sum / num_groups)))
        # groups.append({'probs': probs, 'indices': indices})  # sorted
        self.children.append(ADGNode(probs, indices, self.multiplier / sum(probs) * (probs_sum / num_groups)))

    def is_leaf(self) -> bool:
        if max(self.probs) > self.probs_sum / 2:
            return True
        return False

    def get_final_probs_indices():
        # print(len(ADGNode.final_probs))
        final_probs = ADGNode.final_probs[:]
        final_indices = ADGNode.final_indices[:]
        ADGNode.final_probs = []
        ADGNode.final_indices = []
        return final_probs, final_indices


def adg_encode_decode_step(probs: torch.Tensor,
                           indices: torch.Tensor,
                           message_bits: Optional[str] = None,
                           decode: Optional[str] = None,
                           stego_t: Optional[int] = None,
                           need_full_distribution: bool = False):
    # Set `need_full_distribution = True`, if you need to obtain the KL divergence
    # Otherwise, set `need_full_distribution = False`
    def grouping(probs: List[float], indices: List[int]):
        prob_max = probs[0]
        num_groups = 2**floor(-log2(prob_max))
        groups = []
        for i in range(num_groups - 1):
            mean_probs_sum_per_group = sum(probs) / (num_groups - i)
            groups.append({'probs': [], 'indices': []})
            groups[i]['probs'].append(probs[0])
            groups[i]['indices'].append(indices[0])
            del probs[0]
            del indices[0]
            while True:
                delta = mean_probs_sum_per_group - sum(groups[i]['probs'])
                if delta <= 0:
                    break
                index_idx = find_nearest(delta, probs)
                if probs[index_idx] - delta < delta:
                    groups[i]['probs'].append(probs[index_idx])
                    groups[i]['indices'].append(indices[index_idx])
                    del probs[index_idx]
                    del indices[index_idx]
                else:
                    break
        groups.append({'probs': probs, 'indices': indices})
        return groups

    if decode is None:
        total_code_len = 0
        entropy_step = 0
        minimum_entropy_step = 0
        kld_step = 0
        if need_full_distribution:
            original_probs = probs.clone()  # original probability distribution
            original_indices = indices.clone()

            original_indices, indices_idx = original_indices.sort()
            original_probs = original_probs[indices_idx]

            original_probs = original_probs.tolist()

            node_q = queue.Queue()
            node_q.put(ADGNode(probs, indices))
            while not node_q.empty():
                node = node_q.get()
                node.grouping()
                for x in node.children:
                    node_q.put(x)

            final_probs, final_indices = ADGNode.get_final_probs_indices()
            final_probs = torch.tensor(final_probs, device=device)
            final_indices = torch.tensor(final_indices, device=device)

            final_indices, indices_idx = final_indices.sort()
            final_probs = final_probs[indices_idx]

            final_probs = final_probs.tolist()
            minimum_entropy_step = -log2(max(final_probs))

            entropy_step = entropy(final_probs, base=2)
            kld_step = entropy(final_probs, original_probs, base=2)

        while probs[0].item() <= 0.5:
            probs = probs.tolist()
            indices = indices.tolist()

            groups = grouping(probs, indices)
            # for i in range(len(groups)):
            #     print(sum(groups[i]['probs']))
            code_len = floor(log2(len(groups)))

            selected_group_idx = int(message_bits[total_code_len:total_code_len + code_len], 2)
            probs = groups[selected_group_idx]['probs']
            indices = groups[selected_group_idx]['indices']

            probs, indices_idx = torch.tensor(probs, device=device).sort(descending=True)
            indices = torch.tensor(indices, device=device)

            probs = probs / probs.sum(dim=-1)
            indices = indices[indices_idx]

            total_code_len += code_len
        selected = indices[torch.multinomial(probs, 1).item()].item()
        return SingleEncodeStepOutput(selected, total_code_len, entropy_step, kld_step, 1, minimum_entropy_step)
    else:
        raise NotImplementedError(
            'The author does not disclose the decoding algorithm code. Check out "https://github.com/Mhzzzzz/ADG-steganography" for more information.'
        )