import os
import sys


output = []
n_gpu = 0
test_tag = False

if len(sys.argv) >= 5:
    if sys.argv[4] == 'test':
        test_tag = True
        assert len(sys.argv) == 6

with open(sys.argv[1], 'r') as fp:

    for line in fp.readlines():
        line = line.split("\\")[0]
        line = " ".join(line.split())
        if 'train_net.py' in line and test_tag:
            line = line.replace('train_net.py', 'test_net.py')
        if 'nproc_per_node' in line:
            n_gpu = int(line.split('=')[-1])
        if 'OUTPUT_DIR' in line and test_tag:
            out_dir = line.split()[-1]
            line += ' > ' + os.path.join(out_dir, sys.argv[5])
        output.append(line)

n_node = int(sys.argv[2])

mem = int(sys.argv[3])

prefix = "srun -w irip-c1-compute-{} --gres=gpu:a800:{} -c {} --mem {}G nohup ".format(n_node, n_gpu, n_gpu*6, mem)

print(prefix + " ".join(output))




