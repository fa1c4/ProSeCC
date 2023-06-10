import random
from nltk import sent_tokenize
from datasets import load_dataset, Dataset
import os
import fire
import sys

sys.path.append('src')

from stega import encode
from model import get_model
from classes import Settings
import config


def preprocess_imdb(examples: Dataset, algo: str = config.algo) -> dict:
    tokenizer, _ = get_model()
    message = ''
    with open(config.message_file_path, 'r', encoding='utf-8') as f:
        message = f.read()
    message += message

    ret_dict = {}

    context_lst = []
    generated_lst = []
    ave_er_lst = []
    label_lst = []
    seed_begin = 0

    for i in range(len(examples['text'])):
        context = examples['text'][i]
        context = context.replace(r'<br /><br />', ' ').replace(r'<br />', ' ')  # remove all '<br />'
        context = ' '.join(sent_tokenize(context)[:3])  # Selecting leading 3 sentences as `context`
        config.seed = seed_begin + i  # setting seed

        # stego
        context_lst.append(context)
        single_sample = encode(context, message_bits=message[random.randint(0, 300):])
        generated = single_sample.output['generated_ids']
        capacity = single_sample.output['embedding_rate']
        # generated, capacity = encode(context, message_bits=message[random.randint(0, 300):])
        label_lst.append(1)
        ave_er_lst.append(float(capacity) / len(generated))
        generated_lst.append(tokenizer.decode(generated))

        # cover
        context_lst.append(context)
        single_sample = encode(context, message_bits=message[random.randint(0, 300):], settings=Settings('sample'))
        generated = single_sample.output['generated_ids']
        # generated, _ = encode(context, algo='sample')
        label_lst.append(0)
        ave_er_lst.append(0)
        generated_lst.append(tokenizer.decode(generated))

        if i % 100 == 0:
            print(i)

    ret_dict['context'] = context_lst
    ret_dict['generated'] = generated_lst
    ret_dict['label'] = label_lst
    ret_dict['ave_embedding_rate'] = ave_er_lst

    return ret_dict


def main(index: int, dataset_path: str = config.context_dataset_path):
    if dataset_path not in config.implemented_preprocess_method_datasets:
        raise NotImplementedError("You should implement `preprocess` function for '{}' dataset!".format(dataset_path))
    if config.algo not in config.implemented_algos:
        raise NotImplementedError("Not '{}' algorithm!".format(config.algo))

    if dataset_path == 'imdb':
        preprocess = preprocess_imdb

    # Train set
    print('Start!')
    dataset_train = load_dataset(dataset_path, split='train')[5000 * (index - 1):5000 * index]
    dataset_train = preprocess(dataset_train)  # dict
    dataset_train = Dataset.from_dict(dataset_train)
    dataset_train.to_json(os.path.join('data', config.algo, '{}_train_{}.json'.format(dataset_path, index)))
    print('End!')

    # Test set
    print('Start!')
    dataset_test = load_dataset('imdb', split='test')[5000 * (index - 1):5000 * index]
    dataset_test = preprocess(dataset_test)  # dict
    dataset_test = Dataset.from_dict(dataset_test)
    dataset_test.to_json(os.path.join('data', config.algo, '{}_test_{}.json'.format(dataset_path, index)))
    print('End!')


if __name__ == '__main__':
    fire.Fire(main)
