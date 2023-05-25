import torch.optim as optim

from configs import cfg

_optimizers = {
    'adam': optim.Adam
}

def get_customized_lr_names():
    # list comprehension
    # iterate through the train dict in default.yaml
    # get a list of all keys
    # if keys start with lr_ 
    # store key in list from 4th place onwards
    # ie : (drop lr_)
    # return list with learning rates 
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def get_optimizer(network):
    # optimizer is train['optimizer'] from default.yaml
    optimizer = _optimizers[cfg.train.optimizer]
    # get the learning rates for the various MLPs
    cus_lr_names = get_customized_lr_names()
    params = []
    print('\n\n********** learnable parameters **********\n')
    # sets the learning rates for the learnable parameters
    # for the Network
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue

        is_assigned_lr = False
        for lr_name in cus_lr_names:
            if lr_name in key:
                # store lr value with lr name and network param
                # value in params
                params += [{"params": [value], 
                            "lr": cfg.train[f'lr_{lr_name}'],
                            "name": lr_name}]
                print(f"{key}: lr = {cfg.train[f'lr_{lr_name}']}")
                is_assigned_lr = True

        # if network value requires grad but
        # has no custom lr parameter
        # user train['lr_']
        # set key,value as it is
        if not is_assigned_lr:
            params += [{"params": [value], 
                        "name": key}]
            print(f"{key}: lr = {cfg.train.lr}")

    print('\n******************************************\n\n')

    if cfg.train.optimizer == 'adam':
        # create instance of optimizer class with 
        #   - params
        #   - train lr
        # find out what is beta ?
        optimizer = optimizer(params, lr=cfg.train.lr, betas=(0.9, 0.999))
    else:
        # if optimizer not adam throw error.
        assert False, "Unsupported Optimizer."
    # return optimizer class instance.
    return optimizer
