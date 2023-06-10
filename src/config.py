import os
import torch

# steganography algorithm
implemented_algos = ['forest', 'adg', 'arithmetic', 'meteor', 'dc']
algo = 'dc'
max_length = 100
# temp_lst = [0.8, 1.0, 1.5]
temp_lst = [1.0]
top_p_lst = [0.80, 0.92, 0.95, 0.98, None]  # None means p=1.0
temp = 1.0
top_p = 0.80

# model for generating
# model_name = 'distilgpt2' or 'gpt2' or 'transfo-xl-wt103'
# model_name = 'gpt2'
model_name = 'transfo-xl-wt103'
seed = 666
ptr_multiplier = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model for classifying
cls_model_name = 'xlnet-base-cased'
cls_max_length = 300
learning_rate = 1e-1
num_train_epochs = 10

# saveing path of context, message, stego
path_prefix = 'temp'
context_file_path = os.path.join(path_prefix, 'context.txt')
message_file_path = os.path.join(path_prefix, 'message.txt')
stego_file_path = os.path.join(path_prefix, 'stego.txt')
message_encoded_file_path = os.path.join(path_prefix, 'message_encoded.txt')
message_decoded_file_path = os.path.join(path_prefix, 'message_decoded.txt')
save_result_table = True
result_table_file_path = 'result.xlsx'

# generate dataset
implemented_preprocess_method_datasets = ['imdb']
context_dataset_path = 'imdb'

is_debug = True
decode_timeout = 30

if is_debug:
    decode_timeout = 2**1000
