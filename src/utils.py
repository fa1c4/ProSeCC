import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict
from transformers import PreTrainedTokenizer, PreTrainedModel

import config

device = config.device

# token_idx should be filter out and corresponding reason
# https://huggingface.co/gpt2/raw/main/vocab.json
# filter_out_indices_gpt = {
#     -1: "endoftext can't happen",
#     198: "1 newline can't happen",
#     628: "2 newlines can't happen",
#     220: "just one space can't happen",
#     302: "`\u0120re` can't happen",
#     797: "`\u0120Re` can't happen",
#     15860: "`\u0120Enh` can't happen",
#     2943: "`EC` can't happen",
#     764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
#     837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
# }
filter_out_indices_gpt = {
    -1: "endoftext can't happen",
    198: "1 newline can't happen",
    628: "2 newlines can't happen",
    764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
    837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
}
contain_dollar_lst = [
    3, 720, 7198, 13702, 16763, 17971, 22799, 25597, 29568, 29953, 32047, 32382, 32624, 34206, 35307, 36737, 38892, 39280, 40111,
    43641, 45491, 47113, 48082
]
contain_bad_ellipsis_lst = [19424, 20004, 39864, 44713, 44912, 47082]


def gen_random_message(seed=None, length: int = 1000, save_path: str = config.message_file_path) -> None:
    # Generating binary message (str) randomly via build-in `random` lib
    import random
    random.seed(seed)

    message = ''
    for _ in range(length):
        message += str(random.randint(0, 1))
    print(message)

    if save_path is None:
        return message
    with open(save_path, 'w', encoding='utf-8') as fout:
        fout.write(message)


def limit_past(past):
    if past is None:
        return None
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -1022:]
    return past


def get_probs_indices_past(model: PreTrainedModel,
                           prev=None,
                           past=None,
                           top_p: Optional[float] = None,
                           temp: Optional[float] = None,
                           filter: bool = True) -> Tuple:
    # first, get logits from the model
    if 'gpt2' in config.model_name:
        past = limit_past(past)
        model_output = model(prev, past_key_values=past)
        past = model_output.past_key_values
    else:
        model_output = model(prev, mems=past)
        past = model_output.mems

    logits = model_output.logits

    if 'gpt2' in config.model_name and filter:
        for ele in filter_out_indices_gpt.keys():
            logits[0, -1, ele] = -1e10
        # for ele in contain_dollar_lst:
        #     logits[0, -1, ele] = -1e10  # '$' may cause problems
        # for ele in contain_bad_ellipsis_lst:
        #     logits[0, -1, ele] = -1e10  # bad '...' may cause problems

    else:
        logits[0, -1, 0] = -1e10  # <eos>
        logits[0, -1, 24] = -1e10  # <unk>

    logits, indices = logits[0, -1, :].sort(descending=True)
    logits = logits.double()

    temp = temp if temp is not None else 1.0
    logits_temp = logits / temp
    probs = F.softmax(logits_temp, dim=-1)

    # Getting the top-p `probs` and `indices` from the last layer of `logits`
    if top_p is not None:
        assert top_p > 0 and top_p < 1.0, '`top_p` must be >0 and <=1!'
        cum_probs = probs.cumsum(0)

        k = (cum_probs > top_p).nonzero()[0].item() + 1
        if config.model_name in ['gpt2', 'distilgpt2']:
            k = min(k, 50257)  # `vocab_size` of GPT-2 is 50257
        else:
            k = min(k, 267735)

        probs = probs[:k]
        indices = indices[:k]

        # Normalizing
        # probs = F.softmax(probs, dim=-1)
        probs = 1 / cum_probs[k - 1] * probs
    return probs, indices, past


def is_alpha(s: str) -> bool:
    # A-Za-z
    for i in range(len(s)):
        c = s[i].lower()
        if ord(c) < ord('a') or ord(c) > ord('z'):
            return False
    return True


def my_tokenize(s: str, enc: PreTrainedTokenizer) -> List[str]:
    if len(s) == 0:
        return None
    token_lst = enc.tokenize(s)
    i = 1
    while i < len(token_lst):
        if (is_alpha(token_lst[i][0]) and is_alpha(token_lst[i - 1][-1])) \
            or (not is_alpha(token_lst[i][0]) and not is_alpha(token_lst[i - 1][-1])):
            token_lst[i - 1] = token_lst[i - 1] + token_lst[i]
            del token_lst[i]
        else:
            i += 1
    return token_lst


def longest_common_prefix(a: str, b: str) -> str:
    up = min(len(a), len(b))
    ret = ''
    for i in range(up):
        if a[i] == b[i]:
            ret += a[i]
        else:
            break
    return ret


def convert_decimal_fraction_part_to_binary(decimal: float, precision: int) -> str:
    assert decimal >= 0 and decimal <= 1
    ret = ''
    for i in range(1, precision + 1):
        if decimal - 2**-i >= 0:
            ret += '1'
            decimal -= 2**-i
        else:
            ret += '0'
    return ret


if __name__ == '__main__':
    gen_random_message(length=1000000)
    # X = [0.2, 0.145, 0.136, 0.125, 0.125, 0.114, 0.105, 0.05]
    # X = [0.6, 0.2, 0.2]
    # # X = [0.1, 0.3, 0.3, 0.3]
    # print(cal_entropy(X))
