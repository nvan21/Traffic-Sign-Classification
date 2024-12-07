import os
from experiment import Experiment, EXPERIMENTS


for id, args in EXPERIMENTS.items():
    exp = Experiment(id=id, data_path=args["data_path"], batch_size=args["batch_size"])
    exp.preprocessing(
        transform=args["transform"],
        do_pca=args.get("do_pca", False),
        n_components=args.get("n_components", 50),
    )
    exp.create_model(model=args["model"])
    exp.model.train(train_loader=exp.train_loader, validate_loader=exp.validate_loader)
    exp.model.eval(test_loader=exp.test_loader)
    save_path = os.path.join(args["save_path"], id)
    os.makedirs(save_path, exist_ok=True)
    exp.model.save(save_path=save_path)
