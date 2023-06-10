import os
import time
import numpy as np
from datasets import Dataset, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import sys

sys.path.append('src')
import config

device = config.device


def main(model_name: str = 'checkpoints/ss_v4_xlnet_0330_0342/checkpoint-25000',
         algo: str = config.algo,
         max_length: int = config.cls_max_length):
    print('model_name = {}\nalgorithm={}'.format(model_name, algo))

    tokenizer = AutoTokenizer.from_pretrained(config.cls_model_name)  # you can set 'local_files_only=True'
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

    print('Loading dataset....')
    datadir = os.path.join('data', algo)
    dataset = load_dataset('json', data_dir=datadir, split='train').map(tokenize_func, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

    eval_dataset = dataset['test']

    trainer = Trainer(model=model, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

    print('Predicting....')
    predict_output = trainer.predict(eval_dataset)
    
    import pickle
    with open('predict_output', 'wb') as f:
        pickle.dump(predict_output, f)


if __name__ == '__main__':
    main()