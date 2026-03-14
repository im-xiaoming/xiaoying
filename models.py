import torch
from . import net

def get_model(name, checkpoint):
    model = net.build_model(name)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    return model