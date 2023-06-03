import time

###############################################################################
## Misc Functions
###############################################################################

def cpu_data_to_gpu(cpu_data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    # create a dict variable to store gpu data
    gpu_data = {}
    # cpu_data is a dict variable
    for key, val in cpu_data.items():
        if key in exclude_keys:
            # skip the keys in exclude_keys
            continue

        # if val is of type list
        if isinstance(val, list):
            # assert val has non-zero 
            # number of elements
            assert len(val) > 0
            # if val[0] is not a string
            # (ignore string instance - chungYiWeng)
            if not isinstance(val[0], str):
                # create a list comprehension
                # iterate val
                # tensor.cuda() returns a copy of 
                # the object in cuda memory
                gpu_data[key] = [x.cuda() for x in val]
        elif isinstance(val, dict):
            # if val is a dict
            # create a dict 
            # with key : sub_k and val sub_val
            # with key for the dict = key
            gpu_data[key] = {sub_k: sub_val.cuda() for sub_k, sub_val in val.items()}
        else:
            # if not a dict or list
            # add directoly to cuda
            gpu_data[key] = val.cuda()
    # return gpu_data dict
    return gpu_data


###############################################################################
## Timer
###############################################################################

class Timer():
    def __init__(self):
        self.curr_time = 0

    def begin(self):
        self.curr_time = time.time()

    def log(self):
        diff_time = time.time() - self.curr_time
        self.begin()
        return f"{diff_time:.2f} sec"
