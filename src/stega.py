import time
import torch
from transformers import set_seed
from scipy.stats import entropy
from math import log2
from typing import Tuple, List, Dict, Optional, Union

import config
from model import get_model
from huffman import Node, create_huffman_tree
from utils import get_probs_indices_past, my_tokenize
from classes import SingleExampleOutput, Settings, SingleEncodeStepOutput, SingleDecodeStepOutput

device = config.device
table = {}  # token_idx (int) -> [message_decoded (str), n_ptr_consumed (int)]
time_out = config.decode_timeout


def encode_decode_step(algo: str,
                       probs: torch.Tensor,
                       indices: List,
                       ptr: torch.Tensor,
                       message_bits: Optional[str] = None,
                       decode: Optional[str] = None,
                       stego_t: Optional[int] = None) -> Optional[Union[SingleEncodeStepOutput, SingleDecodeStepOutput, Dict]]:
    # Returns `capacity_t`, `sampled_indx` and `num_ptr_consumed`
    # `(probs, indices)` is already sorted descending by `probs`
    if algo != 'sample' and decode is None and message_bits is None:
        raise ValueError("During encoding, `message_bits` must be set!")
    if decode is not None and decode not in ['table', 'directly']:
        raise ValueError("`decode` must be in \{None, 'table', 'directly'\}!")
    if decode is None and stego_t is not None:
        raise ValueError("During encoding, `stego_t` must not be set!")
    if decode == 'directly' and stego_t is None:
        raise ValueError("If decode == 'directly', `stego_t` must be set!")

    if algo not in ['forest', 'adg']:
        # Forest need several PRN
        # random sampling in adg is processed by `multinomial()`
        ptr = ptr[0]

    probs_cumsum = probs.cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=device), probs_cumsum[:-1]), dim=0)

    def ptr_to_index(ptr: torch.Tensor) -> int:
        index_idx = (ptr >= interval_begin).nonzero()[-1].item()
        index = indices[index_idx]
        return index

    if algo == 'sample':
        return SingleEncodeStepOutput(ptr_to_index(ptr), 0, entropy(probs.tolist(), base=2), 0, 1, -log2(probs[0].item()))

    elif algo == 'dc':
        if decode is None:
            # Determine capacity
            capacity = torch.log2(1 / probs[0]).long().item()
            capacity_upper_bound = capacity + 1

            dc_tbl = {}  # message_bits to idx

            while capacity <= capacity_upper_bound:
                shift_distance = 2**-capacity
                is_available = True
                dc_tbl_new = {}
                for i in range(2**capacity):
                    ptr_i = ptr + i * shift_distance
                    if ptr_i.item() > 1.0:
                        ptr_i -= 1
                    idx_order = (ptr_i >= interval_begin).nonzero()[-1].item()
                    idx = indices[idx_order]
                    if idx in dc_tbl_new.values():  # 若`idx`已存在，表明不满足两两互异
                        is_available = False
                        break
                    dc_tbl_new[i] = idx
                if not is_available:
                    break
                dc_tbl = dc_tbl_new
                capacity += 1
            capacity -= 1  # 多加了1

            if capacity < 1:  # 无法嵌入消息，但依然需要返回一个单词
                return SingleEncodeStepOutput(ptr_to_index(ptr), 0, entropy(probs.tolist(), base=2), 0, 1, -log2(probs[0].item()))
            return SingleEncodeStepOutput(dc_tbl[int(message_bits[:capacity], 2)], capacity, entropy(probs.tolist(), base=2), 0,
                                          1, -log2(probs[0].item()))
        elif decode == 'directly':
            pass
        elif decode == 'table':
            pass

    elif algo == 'forest':
        if type(probs) == torch.Tensor:
            probs = probs.tolist()
        node = create_huffman_tree(indices=indices, probs=probs, search_for=stego_t)
        code_len = 0
        d = 0  # n_ptr_consumed
        if decode is None:
            while not node.is_leaf():
                probs_sum = node.prob
                ptr_0 = ptr[d] * probs_sum
                ptr_1 = (ptr[d] + 0.5) * probs_sum
                if ptr_1 > probs_sum:
                    ptr_1 -= probs_sum
                path_table = {}  # message_i (str) -> selected subtree (node)

                path_table['0'] = node.left if ptr_0 < node.left.prob else node.right
                path_table['1'] = node.left if ptr_1 < node.left.prob else node.right

                node = path_table[message_bits[code_len]]
                if path_table['0'] != path_table['1']:  # can embed
                    code_len += 1
                d += 1
            return SingleEncodeStepOutput(node.index, code_len, entropy(probs, base=2), 0, d, -log2(probs[0]))
        elif decode == 'directly':
            message_decoded = ''
            while not node.is_leaf():
                probs_sum = node.prob
                ptr_0 = ptr[d] * probs_sum
                ptr_1 = (ptr[d] + 0.5) * probs_sum
                if ptr_1 > probs_sum:
                    ptr_1 -= probs_sum
                path_table = {}  # message_i (str) -> selected subtree (int)

                path_table['0'] = -1 if ptr_0 < node.left.prob else 1
                path_table['1'] = -1 if ptr_1 < node.left.prob else 1

                if path_table['0'] != path_table['1']:  # can embed
                    path_table = dict(zip(path_table.values(), path_table.keys()))  # selected subtree (int) -> message_i (str)
                    if node.search_path is None:  # fail to decode
                        return
                    message_decoded += path_table[node.search_path]
                    if node.search_path == -1:
                        node = node.left
                    elif node.search_path == 1:
                        node = node.right
                else:
                    if path_table['0'] == -1:
                        node = node.left
                    else:
                        node = node.right
                d += 1
            if node.search_path != 0:  # fail to decode
                return
            return SingleDecodeStepOutput(message_decoded, d)
        elif decode == 'table':
            # 枚举编码，获得译码表
            global table
            table = {}

            def dfs(d: int, node: Node, msg_decoded: str = ''):
                if node.is_leaf():
                    global table
                    table[node.index] = [msg_decoded, d]
                    return
                probs_sum = node.prob
                ptr_0 = ptr[d] * probs_sum
                ptr_1 = (ptr[d] + 0.5) * probs_sum
                if ptr_1 > probs_sum:
                    ptr_1 -= probs_sum
                path_table = {}  # message_i (str) -> selected subtree (node)

                path_table['0'] = node.left if ptr_0 < node.left.prob else node.right
                path_table['1'] = node.left if ptr_1 < node.left.prob else node.right

                if path_table['0'] != path_table['1']:  # can embed
                    dfs(d + 1, path_table['0'], msg_decoded + '0')
                    dfs(d + 1, path_table['1'], msg_decoded + '1')
                else:
                    dfs(d + 1, path_table['0'], msg_decoded)

            dfs(0, node)
            return table
    elif algo == 'adg':
        from adg import adg_encode_decode_step
        return adg_encode_decode_step(probs, torch.tensor(indices, dtype=int, device=device), message_bits, decode, stego_t)


@torch.no_grad()
def encode(context: str,
           message_bits: Optional[str] = None,
           settings: Settings = Settings(),
           verbose: bool = False,
           segment: Optional[int] = None) -> SingleExampleOutput:
    # General architecture of Steganography Encoding (message_bits -> English text)
    algo, temp, top_p, length, seed = settings()
    if algo not in ['sample'] + config.implemented_algos:
        raise NotImplementedError
    if algo != 'sample' and (message_bits is None or len(message_bits) == 0):
        raise ValueError
    if segment is not None and algo != 'forest':
        raise NotImplementedError
    if verbose:
        print('=' * 40 + 'Encoding' + '=' * 40)

    if algo == 'arithmetic':
        from arithmetic import encode_arithmetic
        return encode_arithmetic(context, message_bits, settings, verbose)
    elif algo == 'meteor':
        from meteor import encode_meteor
        return encode_meteor(context, message_bits, settings, verbose)

    start = time.time()

    tokenizer, model = get_model()
    set_seed(seed)

    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before
    generated_ids = None

    ptr_all = None
    if algo != 'adg':
        ptr_all = torch.rand(length * config.ptr_multiplier).to(device)

    total_capacity = 0
    total_entropy = 0
    total_minimum_entropy = 0
    total_log_probs = 0  # for perplexity
    total_kld = 0
    max_kld = 0

    t = 0
    while t < length:
        if segment is None:
            probs, indices, past = get_probs_indices_past(model, prev, past, top_p, temp)
            indices = indices.tolist()
        else:
            probs = torch.tensor([], dtype=int, device=device)
            indices = []  # paired
            probs_1, indices_1, past = get_probs_indices_past(model, prev, past, top_p, temp)
            for i in range(len(indices_1)):
                probs_2, indices_2, past_2 = get_probs_indices_past(model,
                                                                    torch.tensor([indices_1[i]], device=device).unsqueeze(0),
                                                                    past, top_p, temp)
                probs = torch.cat((probs, probs_1[i] * probs_2))
                indices.extend(list([indices_1[i].item(), x] for x in indices_2.tolist()))

        sampled_index, capacity_t, entropy_t, kld_step, n_ptr_consumed, min_entropy_t = encode_decode_step(
            algo, probs, indices, ptr_all, message_bits)()

        indices_idx = indices.index(sampled_index)

        total_entropy += entropy_t
        total_minimum_entropy += min_entropy_t
        total_log_probs += log2(probs[indices_idx].item())
        total_kld += kld_step
        if kld_step > max_kld:
            max_kld = kld_step

        if algo != 'adg':
            ptr_all = ptr_all[n_ptr_consumed:]

        # when `capacity_t == 0`, cannot embed message, but still needs to return a token_index
        if capacity_t > 0:
            total_capacity += capacity_t
            message_bits = message_bits[capacity_t:]  # remove the encoded part of `message_bits`

        # print(sampled_index)
        if generated_ids is None:
            generated_ids = [sampled_index] if type(sampled_index) == int else sampled_index
        else:
            if type(sampled_index) == int:
                generated_ids.append(sampled_index)
            else:
                generated_ids.extend(sampled_index)
        if segment is None:
            t += 1
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
        else:
            t += 2
            prev = torch.tensor(sampled_index, device=device).unsqueeze(0)

    end = time.time()
    embedding_efficiency = total_capacity / total_entropy if total_entropy != 0 else 0
    perplexity = 2**(-1 / length * total_log_probs)
    ave_kld = total_kld / length

    if verbose:
        print(generated_ids)
        print('embeding_rate = {:.2f}bpw'.format(total_capacity / len(generated_ids)))
        print('total_entropy = {:.2f}'.format(total_entropy))
        print('embedding_efficiency = {:.3f}'.format(embedding_efficiency))
        print('perplexity = {:.3f}'.format(perplexity))
    return SingleExampleOutput(generated_ids, total_capacity, total_entropy, ave_kld, max_kld, perplexity, end - start, settings,
                               total_minimum_entropy)


def decode(context: str, stego: Union[str, List[int]], settings: Settings = Settings(), verbose: bool = False) -> str:
    # General architecture of Steganography Decoding (English text -> message_bits)
    # Returns `message_decoded`
    algo, temp, top_p, length, seed = settings()
    if algo != 'forest':
        raise ValueError("We have not implement decode algorithm named '{}'!".format(algo))

    if verbose:
        print('=' * 40 + 'Decoding' + '=' * 40)

    tokenizer, model = get_model()
    set_seed(seed)

    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = context  # indices that were never passed to the model before
    message_decoded = ''

    if algo != 'adg':
        ptr_all = torch.rand(length * config.ptr_multiplier).to(device)

    if type(stego) == str and 'gpt' not in config.model_name:
        stego = tokenizer(stego)['input_ids']

    if type(stego) == list:
        t = 0
        while t < len(stego):
            probs, indices, past = get_probs_indices_past(model, prev, past, top_p, temp)
            indices = indices.tolist()
            single_decode_step_output = encode_decode_step(algo, probs, indices, ptr_all, decode='directly', stego_t=stego[t])
            if single_decode_step_output is None:
                raise ValueError('Failed to decode!')
            message_decoded_t, n_ptr_consumed = single_decode_step_output()
            if algo != 'adg':
                ptr_all = ptr_all[n_ptr_consumed:]

            message_decoded += message_decoded_t
            prev = torch.tensor([stego[t]], device=device).unsqueeze(0)
            t += 1
        return message_decoded
    elif type(stego) == str:
        stego = my_tokenize(stego, tokenizer)
        start = 0

        def dfs(t: int, prev: torch.Tensor, past: Tuple, ptr_idx: int = 0, cmp: int = 1):
            if t > length or len(stego) > length:
                return False, None
            if t == len(stego):
                if t == length:
                    return True, ''
                else:
                    return False, None
            if time.time() - start > time_out:
                raise TimeoutError
            probs, indices, past = get_probs_indices_past(model, prev, past, top_p, temp)
            indices = indices.tolist()
            table = encode_decode_step(algo, probs, indices, ptr_all[ptr_idx:], decode='table')
            matched_table = {}
            stego_t = stego[t]
            for token_idx in table.keys():
                if stego_t.startswith(tokenizer.decoder[token_idx]):
                    matched_table[token_idx] = table[token_idx]
            if len(matched_table) == 0:  # fail to decode
                return False, None
            # 将`matched_table`排序，优先考虑较长匹配
            if cmp == 1:
                matched_table = sorted(matched_table.items(), key=lambda x: len(tokenizer.decoder[x[0]]), reverse=True)
            elif cmp == 2:
                matched_table = sorted(matched_table.items(), key=lambda x: probs[indices.index(x[0])].item(), reverse=True)
            for token_idx, message_decoded_and_n_ptr_consumed in matched_table:
                message_decoded_t = message_decoded_and_n_ptr_consumed[0]
                n_ptr_consumed_t = message_decoded_and_n_ptr_consumed[1]
                stego_next = stego_t[len(tokenizer.decoder[token_idx]):]
                stego_t_new = stego_t[:len(tokenizer.decoder[token_idx])]
                if len(stego_next) > 0:
                    stego[t] = stego_t_new
                    stego.insert(t + 1, stego_next)
                prev = torch.tensor(tokenizer.encoder[stego_t_new], device=device).unsqueeze(0).unsqueeze(0)
                done_future, message_decoded_future = dfs(t + 1, prev, past, ptr_idx + n_ptr_consumed_t, cmp)
                if done_future:
                    return True, message_decoded_t + message_decoded_future
                # recover state
                if len(stego_next) > 0:
                    stego[t] = stego_t
                    stego.pop(t + 1)
            return False, None

        done = False

        try:
            print('Trying `cmp_1`......')
            start = time.time()
            done, message_decoded = dfs(0, prev, past, 0, 1)
            if not done:
                raise TimeoutError
        except TimeoutError:
            try:
                print('Trying `cmp_2`......')
                start = time.time()
                done, message_decoded = dfs(0, prev, past, 0, 2)
            except TimeoutError:
                pass

        if done:
            return message_decoded
        else:
            print('Failed to decode!')
            return ''
