import numpy as np


# creates a cellular automaton pattern (https://en.wikipedia.org/wiki/Cellular_automaton)
def get_applied_rule(row, rule_dict):
    lcr = np.vstack((np.roll(row, 1), row, np.roll(row, -1)))
    new_row = np.apply_along_axis(lambda x: rule_dict[''.join(x.astype(str))], axis=0, arr=lcr)
    return new_row


def create_automaton(rule=110, size=(238, 238), seed=42):
    np.random.seed(seed)
    bin_rule_list = [int(b) for b in np.binary_repr(rule, 8)]
    pattern_list = [np.binary_repr(pat, 3) for pat in range(7, -1, -1)]
    rule_dict = {key: val for key, val in zip(pattern_list, bin_rule_list)}
    result = np.zeros(size, dtype=np.int)
    result[0, :] = np.random.randint(2, size=size[1])
    for row in range(result.shape[0]-1):
        result[row + 1] = get_applied_rule(result[row], rule_dict)
    return result


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    image = create_automaton()
    plt.imshow(image)
    print('done')
