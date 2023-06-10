import time
import pandas as pd
from nltk import sent_tokenize
from datasets import Dataset, load_dataset
from typing import Optional

import config
from model import get_model
from stega import encode, decode
from classes import Summary, Settings, SingleExampleOutput, columns

device = config.device


def test_one_example(context: str,
                     message_bits: str,
                     settings: Settings,
                     verbose: bool = False,
                     whether_to_decode: bool = False) -> SingleExampleOutput:
    # encode
    example = encode(context, message_bits, settings, verbose=verbose)
    # decode
    if whether_to_decode:
        stego_ids = example.output['generated_ids']
        tokenizer, _ = get_model()
        stego = tokenizer.decode(stego_ids)
        message_encoded = message_bits[:example.output['n_bits']]
        message_decoded = decode(context, stego_ids, settings, verbose=verbose)
        print('message_encoded == message_decoded) = {}'.format(message_encoded == message_decoded))
    return example


def test(num_examples: int = 30,
         dataset: Optional[Dataset] = None,
         message: Optional[str] = None,
         settings: Settings = Settings(),
         verbose: bool = True) -> Summary:
    if dataset is None:
        dataset = load_dataset(config.context_dataset_path, split='train')
        dataset = dataset[:num_examples]
    if message is None:
        with open(config.message_file_path, 'r', encoding='utf-8') as f:
            message = f.read()
        message *= 4
    seed_begin = 0
    summary = Summary(settings)
    print('seed = ', end='')
    for i in range(num_examples):
        context = dataset['text'][i]
        context = context.replace('<br /><br />', ' ').replace('<br />', ' ')  # remove all '<br />'
        context = ' '.join(sent_tokenize(context)[:3])  # Selecting leading 3 sentences as `context`
        settings.seed = seed_begin + i
        print(settings.seed, end=' ')
        example = test_one_example(context, message, settings, verbose=verbose)  # whether_to_decode=True
        summary.add_example(example)
        if verbose:
            print()
            print(example)
            if i != num_examples - 1:
                print('seed = ', end='')
    print()
    if verbose:
        print(summary)
    return summary


def main(num_examples: int = 30):
    algo = 'meteor'
    dataset = load_dataset(config.context_dataset_path, split='train')
    dataset = dataset[:num_examples]
    with open(config.message_file_path, 'r', encoding='utf-8') as f:
        message = f.read()
    message *= 4
    if config.save_result_table:
        df = pd.DataFrame(columns=columns)
    for temp in config.temp_lst:
        for top_p in config.top_p_lst:
            settings = Settings(algo, temp, top_p)
            summary = test(num_examples, dataset, message, settings, verbose=False)
            print(summary)
            df = pd.concat([df, summary.gather()], ignore_index=True)
    if config.save_result_table:
        df.to_excel('{}_result_{}_{}.xlsx'.format(algo, config.model_name, time.strftime("%m%d_%H%M", time.localtime())))


if __name__ == '__main__':
    # test(settings=Settings(seed=1))
    # test()
    main()