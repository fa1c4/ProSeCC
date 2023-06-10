from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

model_name = 'transfo-xl-wt103'
tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
model = TransfoXLLMHeadModel.from_pretrained(model_name)

ids = [
    10, 8463, 62275, 10, 23, 1, 40, 418, 15, 68, 6179, 18795, 602, 3, 21726, 4379, 3197, 6120, 95590, 230, 1, 418, 60, 1098, 19,
    45, 27, 8851, 5, 5300, 16, 70, 23, 1613, 3, 360, 45, 2, 10, 8463, 62275, 10, 6036, 304, 63, 2641, 3106, 14, 268, 3, 122, 2651,
    80, 1, 91, 5, 108, 2651, 1, 1289, 2, 70, 11, 768, 18, 23917, 2, 1, 3663, 4, 68, 6179, 18795, 602, 3, 34369, 718, 14469, 51258,
    32, 977, 6, 313, 8, 91, 80, 1, 229, 4, 21338, 7, 1, 3784, 2, 36, 30, 1477, 2299, 162, 30
]

text = tokenizer.decode(ids)
print(text)
ids2 = tokenizer(text)['input_ids']

# ids2 = 

print(ids2)
print(ids == ids2)