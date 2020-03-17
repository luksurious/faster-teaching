import json
import numpy as np

from actions import Actions


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Actions):
            return str(obj)
        else:
            return super(CustomEncoder, self).default(obj)
