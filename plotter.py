import pickle

with open("./runs/base_cnn/log.pkl", "rb") as f:
    metrics = pickle.load(f)

print(metrics)
