import numpy as np


def sign_filter(params):
    # return params
    return np.sign(params)


def difference_filter(params):
    return [params[i] - params[i + 1] for i in range(len(params) - 1)]


if __name__ == "__main__":
    a = [1.1, -0.9]
    print(sign_filter(a))
