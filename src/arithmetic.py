import time
import torch
import torch.nn.functional as F
from transformers import set_seed
from scipy.stats import entropy
from typing import Optional

from classes import Settings, SingleExampleOutput
from utils import get_probs_indices_past
from model import get_model
import config

device = config.device


# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    # 包括头的公共子串长度
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


@torch.no_grad()
def encode_arithmetic(context: str, message: str, settings: Settings = Settings(), verbose: bool = False, precision: int = 32):
    algo, temp, top_p, length, seed = settings()
    message = list(int(x) for x in message)

    start = time.time()

    tokenizer, model = get_model()
    set_seed(seed)

    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None
    prev = context
    generated_ids = None

    max_val = 2**precision
    threshold = 2**(-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    total_num = 0
    total_log_probs = 0
    total_kld = 0  # in bits
    total_entropy = 0
    max_kld = 0

    i = 0  # index of message_bits

    t = 0  # index of token
    while t < length:
        probs_temp, indices, past = get_probs_indices_past(model, prev, past, top_p, temp)

        # Cutoff low probabilities that would be rounded to 0
        cur_int_range = cur_interval[1] - cur_interval[0]
        cur_threshold = 1 / cur_int_range
        if (probs_temp < cur_threshold).nonzero().numel() > 0:
            k = max(2, (probs_temp < cur_threshold).nonzero()[0].item())  # not less than 2
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k
        else:
            probs_temp_int = probs_temp.clone()
        # Rescale to correct range
        probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

        # Round probabilities to integers given precision
        probs_temp_int = probs_temp_int.round().long()  # rounding introduces bias
        cum_probs = probs_temp_int.cumsum(0)

        # Remove any elements from the bottom if rounding caused the total prob to be too large
        overfill_index = (cum_probs > cur_int_range).nonzero()
        if len(overfill_index) > 0:
            cum_probs = cum_probs[:overfill_index[0]]

        # Add any mass to the top if removing/rounding causes the total prob to be too small
        cum_probs += cur_int_range - cum_probs[-1]  # add

        # Get out resulting probabilities
        probs_final = cum_probs.clone()
        probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

        # Convert to position in range
        cum_probs += cur_interval[0]

        # Get selected index based on binary fraction from message bits
        message_bits = message[i:i + precision]
        if i + precision > len(message):
            message_bits = message_bits + [0] * (i + precision - len(message))
        message_idx = bits2int(reversed(message_bits))
        selection = (cum_probs > message_idx).nonzero()[0].item()
        sampled_index = indices[selection].item()

        # Calculate new range as ints
        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
        new_int_top = cum_probs[selection]

        # Convert range to bits
        new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
        new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

        # Consume most significant bits which are now fixed and update interval
        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
        i += num_bits_encoded

        new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
        new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

        cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
        cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

        # Gather statistics
        total_log_probs += torch.log2(probs_temp[selection]).item()

        actual_distribution = probs_final.double() / probs_final.sum()
        # total_kld += kl(q, logq, log_probs[:len(q)])
        kld_step = entropy(actual_distribution.tolist() + [0] * (len(probs_temp.tolist()) - len(actual_distribution.tolist())),
                             probs_temp.tolist(),
                             base=2)
        total_kld += kld_step
        if kld_step > max_kld:
            max_kld = kld_step

        # total_entropy += entropy(probs_temp, log_probs_temp)
        total_entropy += entropy(actual_distribution.tolist(), base=2)

        if generated_ids is None:
            generated_ids = [sampled_index]
        else:
            generated_ids.append(sampled_index)
        t += 1

        # Update history with new token
        prev = torch.tensor([sampled_index], device=device).unsqueeze(0)

    end = time.time()

    ave_kld = total_kld / t
    perplexity = 2**(-1 / length * total_log_probs)
    embedding_efficiency = i / total_entropy
    # settings.length = t

    if verbose:
        print(generated_ids)
        print('embeding_rate = {:.2f}bpw'.format(i / t))
        print('total_entropy = {:.2f}'.format(total_entropy))
        print('embedding_efficiency = {:.3f}'.format(embedding_efficiency))
        print('perplexity = {:.3f}'.format(perplexity))
    # return output[len(context):].tolist(), avg_NLL, ave_kld, words_per_bit, avg_Hq
    return SingleExampleOutput(generated_ids, i, total_entropy, ave_kld, max_kld, perplexity, end - start, settings)


if __name__ == '__main__':
    with open(config.context_file_path, 'r', encoding='utf-8') as f:
        context = f.read()
    with open(config.message_file_path, 'r', encoding='utf-8') as f:
        message = f.read()
    print(encode_arithmetic(context, message))