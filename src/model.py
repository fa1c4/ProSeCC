import config

model_name = config.model_name
device = config.device

if model_name in ['gpt2', 'distilgpt2']:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
elif model_name == 'transfo-xl-wt103':
    from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
    tokenizer = TransfoXLTokenizer.from_pretrained(model_name, local_files_only=True)
    model = TransfoXLLMHeadModel.from_pretrained(model_name).to(device)
model.eval()


def get_model():
    # Getting model and tokenizer (encoder)
    return tokenizer, model


if __name__ == '__main__':
    tokenizer, model = get_model()
    context = 'We were both young when I first saw you,'
    context_ids = tokenizer(context, return_tensors='pt')
    outputs = model(**context_ids)
    logits = outputs.logits[0, -1, :]

    print()
