import imp

from configs import cfg

def _query_trainer():
    module = cfg.trainer_module
    # replace . with / to create a path and append .py at the end
    trainer_path = module.replace(".", "/") + ".py"
    trainer = imp.load_source(module, trainer_path).Trainer
    # return Trainer class object
    return trainer


def create_trainer(network, optimizer):
    Trainer = _query_trainer()
    # create a trainer class instance 
    # inheriting network and optimizer classes
    return Trainer(network, optimizer)
