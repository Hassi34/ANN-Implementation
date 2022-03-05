from OneFlow.utils.common import read_config
from OneFlow.utils.data_mgmt import get_data
from OneFlow.utils.model import StepFlow
import argparse, os 

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    sp = StepFlow(config, X_train, y_train, X_valid, y_valid)
    sp.create_model()
    sp.fit_model()
    sp.save_final_model()
    sp.save_plot()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)