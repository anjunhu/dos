from ..utils import utils


def get_visuals_dict(input_dict, names, num_visuals):
    return {
        name: utils.tensor_to_image(input_dict[name][:num_visuals])
        for name in names
        if name in input_dict
    }
