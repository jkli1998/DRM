import os
import sys
import numpy as np


class GenerateRandom(object):
    def __init__(self):
        path = os.path.join("SEMI_utils", "relation_length.log")
        with open(path, 'r') as fp:
            self.length = int(fp.readlines()[0])

    def run(self, percent, random_seed):
        # percent, random_seed, index.
        np.random.seed(random_seed)
        labeled_tot = int(percent / 100. * self.length)
        labeled_ind = np.random.choice(range(self.length), size=labeled_tot)
        name = os.path.join("SEMI_utils", "tmp_rand.npy")
        if os.path.exists(name):
            data = np.load(name, allow_pickle=True).item()
            if percent in data:
                data[percent][random_seed] = labeled_ind
            else:
                data[percent] = {random_seed: labeled_ind}
        else:
            data = {percent: {random_seed: labeled_ind}}
        np.save(name, data)


def main():
    percent = int(sys.argv[1])
    seed = int(sys.argv[2])
    GenerateRandom().run(percent, seed)


if __name__ == "__main__":
    main()

