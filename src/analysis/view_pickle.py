import pickle

with open('predict_output', 'rb') as f:
    predict_output = pickle.load(f)

print()