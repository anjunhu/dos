from ..utils import utils


def get_visuals_dict(input_dict, names, num_visuals):
    return {
        # is a dictionary comprehension that iterates over the names list and creates key-value pairs in the dictionary.
        name: utils.tensor_to_image(input_dict[name][:num_visuals])
        for name in names
        if name in input_dict
    }
