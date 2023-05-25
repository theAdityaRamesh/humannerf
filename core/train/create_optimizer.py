import imp

from configs import cfg

def create_optimizer(network):
    module = cfg.optimizer_module
    # replace . with / to create a path and append .py at the end
    optimizer_path = module.replace(".", "/") + ".py"
    # return instance of optimizer class with network parameters set.
    return imp.load_source(module, optimizer_path).get_optimizer(network)
