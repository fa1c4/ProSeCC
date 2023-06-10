import pandas as pd
from typing import List, Tuple, Union

import config


class Settings:
    def __init__(self,
                 algo: str = config.algo,
                 temp: float = config.temp,
                 top_p: float = config.top_p,
                 length: int = config.max_length,
                 seed: int = config.seed) -> None:
        self.algo = algo
        self.temp = temp
        self.top_p = top_p
        self.length = length
        self.seed = seed

    def __call__(self) -> Tuple:
        return self.algo, self.temp, self.top_p, self.length, self.seed


class SingleEncodeStepOutput:
    def __init__(self,
                 sampled_index: Union[int, List[int]],
                 n_bits: int,
                 entropy: float,
                 kld: float,
                 n_ptr_consumed: int,
                 minimum_entropy: float = 0) -> None:
        self.sampled_index = sampled_index
        self.n_bits = n_bits
        self.entropy = entropy
        self.kld = kld
        self.n_ptr_consumed = n_ptr_consumed
        self.minimum_entropy = minimum_entropy

    def __call__(self) -> Tuple:
        return self.sampled_index, self.n_bits, self.entropy, self.kld, self.n_ptr_consumed, self.minimum_entropy


class SingleDecodeStepOutput:
    def __init__(self, message_decoded, n_ptr_consumed) -> None:
        self.message_decoded = message_decoded
        self.n_ptr_consumed = n_ptr_consumed

    def __call__(self) -> Tuple:
        return self.message_decoded, self.n_ptr_consumed


class SingleExampleOutput:
    def __init__(self,
                 generated_ids: List[int],
                 total_capacity: int,
                 total_entropy: float,
                 ave_kld: float,
                 max_kld: float,
                 perplexity: float,
                 time_cost: float,
                 settings: Settings,
                 total_minimum_entropy: float = 0) -> None:
        self.output = {
            'generated_ids': generated_ids,
            'algorithm': settings.algo,
            'temperature': settings.temp,
            'top-p': settings.top_p,
            'n_bits': total_capacity,
            'n_tokens': len(generated_ids),
            'total_entropy': total_entropy,
            'ave_kld': ave_kld,
            'max_kld': max_kld,
            'embedding_rate': total_capacity / len(generated_ids),
            'utilization_rate': total_capacity / total_entropy if total_entropy != 0 else 0,
            'perplexity': perplexity,
            'time_cost': time_cost,
            'total_minimum_entropy': total_minimum_entropy
        }

    def __str__(self) -> str:
        excluded_attr = ['generated_ids']
        selected_attr = list(self.output.keys())
        for x in excluded_attr:
            selected_attr.remove(x)
        return '\n'.join('{} = {}'.format(x, self.output[x]) for x in selected_attr)


# columns = [
#     'temperature', 'top-p', 'total_time_cost', 'ave_time_cost', 'total_n_bits', 'total_n_tokens', 'ave_embedding_rate',
#     'total_entropy', 'ave_entropy', 'ave_kld', 'max_kld', 'utilization_rate', 'ave_perplexity'
# ]

columns = [
    'temperature', 'top-p', 'total_n_bits', 'total_n_tokens', 'total_entropy', 'total_time_cost', 'ave_time_cost', 'ave_kld',
    'max_kld', 'ave_embedding_rate', 'ave_entropy', 'utilization_rate', 'ave_perplexity', 'total_minimum_entropy',
    'ave_minimum_entropy'
]


class Summary:
    def __init__(self, settings: Settings = Settings()) -> None:
        self.n_examples = 0
        self.total_perplexity = 0
        self.total_ave_kld = 0
        self.output = {
            'algorithm': settings.algo,
            'temperature': settings.temp,
            'top-p': settings.top_p,
            'total_n_bits': 0,
            'total_n_tokens': 0,
            'total_entropy': 0,
            'total_time_cost': 0,
            'max_kld': 0,
            'total_minimum_entropy': 0,
        }

    def __str__(self) -> str:
        self.process()
        selected_attr = list(self.output.keys())
        return '\n'.join('{} = {}'.format(x, self.output[x]) for x in selected_attr)

    def add_example(self, example: SingleExampleOutput):
        self.output['total_n_bits'] += example.output['n_bits']
        self.output['total_n_tokens'] += example.output['n_tokens']
        self.output['total_entropy'] += example.output['total_entropy']
        self.output['total_time_cost'] += example.output['time_cost']
        self.total_perplexity += example.output['perplexity']
        self.total_ave_kld += example.output['ave_kld']
        if example.output['max_kld'] > self.output['max_kld']:
            self.output['max_kld'] = example.output['max_kld']
        self.n_examples += 1
        self.output['total_minimum_entropy'] += example.output['total_minimum_entropy']

    def process(self):
        self.output['ave_embedding_rate'] = self.output['total_n_bits'] / self.output['total_n_tokens']
        self.output['utilization_rate'] = self.output['total_n_bits'] / self.output['total_entropy'] if self.output[
            'total_entropy'] != 0 else 0
        self.output['ave_entropy'] = self.output['total_entropy'] / self.output['total_n_tokens']
        self.output['ave_perplexity'] = self.total_perplexity / self.n_examples
        self.output['ave_kld'] = self.total_ave_kld / self.n_examples
        self.output['ave_time_cost'] = self.output['total_time_cost'] / self.output['total_n_bits'] if self.output[
            'total_n_bits'] != 0 else 0
        self.output['ave_minimum_entropy'] = self.output['total_minimum_entropy'] / self.output['total_n_tokens']

    def gather(self):
        self.process()
        ret_lst = []
        for column in columns:
            ret_lst.append(self.output[column])
        df = pd.DataFrame(ret_lst, index=columns).T
        return df