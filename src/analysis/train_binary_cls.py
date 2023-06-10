import os
import time
import numpy as np
from datasets import Dataset, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import sys

sys.path.append('src')
import config

device = config.device


def main(model_name: str = config.cls_model_name, algo: str = config.algo, max_length: int = config.cls_max_length):
    print('model_name = {}\nalgorithm={}'.format(model_name, algo))

    max_length = 200
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # you can set 'local_files_only=True'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def tokenize_func(examples: Dataset):
        compound = list(x + y for x, y in zip(examples['context'], examples['generated']))
        del examples['context'], examples['generated'], examples['ave_embedding_rate']
        return tokenizer(compound, padding='max_length', truncation=True, max_length=max_length)
    
    def tokenize_func_only_generated(examples: Dataset):
        return tokenizer(examples['generated'], padding='max_length', truncation=True, max_length=max_length)

    datadir = os.path.join('data', algo)
    dataset = load_dataset('json', data_dir=datadir, split='train').map(tokenize_func_only_generated, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    output_dir = os.path.join('checkpoints', '{}_{}_{}'.format(algo,
                                                               model_name.split('-')[0],
                                                               time.strftime("%m%d_%H%M", time.localtime())))

    # set `dataloader_drop_last` to avoid error during training in multi-GPU
    training_args = TrainingArguments(output_dir=output_dir,
                                      evaluation_strategy='epoch',
                                      num_train_epochs=config.num_train_epochs,
                                      dataloader_drop_last=True,
                                      learning_rate=config.learning_rate)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)
    trainer.train()


if __name__ == '__main__':
    main()