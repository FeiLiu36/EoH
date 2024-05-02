import importlib
import time
import torch
import foolbox as fb
from utils.get_data import get_data_by_id
from utils.get_model import get_model
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH)  # This is for finding all the modules

from attacks.evo_for_ael_no_mem import EvoAttack

# File Location to Store the datasets
DATASET = "cifar10"  # Dataset name, if use imagenet, must download in advance
DATASET_PATH = os.path.join(ABS_PATH, "datasets", DATASET)  # store the datasets under the directory of current file
TEST_IMAGES = 8
MODEL_PATH = os.path.join(ABS_PATH, "models")  # store the models under the directory of current file



class Evaluation():
    def __init__(self, dataset_path=DATASET_PATH, dataset_name=DATASET, num_test_images=TEST_IMAGES, model_path=MODEL_PATH, **kwargs) -> None:
        self.test_set = get_data_by_id(dataset_name, use_train_data=False, data_path=dataset_path)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=num_test_images, shuffle=False, pin_memory=True
        )
        self.model = get_model(model_path=model_path)
        self.steps = 2000
        self.running_time = 10

    # @func_set_timeout(5)
    def greedy(self, eva):
        if torch.cuda.is_available():
            self.model.cuda()

        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))
        attack_use = EvoAttack(eva, steps=8000)
        for i, (x, y) in enumerate(self.test_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            img_adv = attack_use.run(fmodel, x, y)
            # y_pred = self.model(x)
            # y_pred_adv = self.model(img_adv)
            distance = torch.linalg.norm((x - img_adv).flatten(start_dim=1), axis=1)
            distance = distance.mean()
            distance = float(distance.cpu().numpy())
            print("average dis: ",distance)
            return distance

    def evaluate(self):
        time.sleep(1)
        try:
            heuristic_module = importlib.import_module("ael_alg")
            eva = importlib.reload(heuristic_module)
            fitness = self.greedy(eva)
            return fitness
        except Exception as e:
            print("Error:", str(e))
            return None
            


