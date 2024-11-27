from cnn import CNN
from experiment import Experiment, EXPERIMENTS


for id, args in EXPERIMENTS.items():
    exp = Experiment(id=id, data_path=args["data_path"], batch_size=args["batch_size"])
    exp.preprocessing(transform=args["transform"])
    exp.create_model(model=args["model"])
    exp.model.train(train_loader=exp.train_loader, validate_loader=exp.validate_loader)
    exp.model.eval()
