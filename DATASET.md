## DATASET

If you want to use other directory, please link it in `DATASETS` of `maskrcnn_benchmark/config/paths_catelog.py`. 

### For VG Dataset:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`.
2. Download the [scene graphs](https://1drv.ms/u/s!AjK8-t5JiDT1kxyaarJPzL7KByZs?e=bBffxj) and extract them to `datasets/vg/VG-SGG-with-attri.h5`.
3. The former list of zero-shot triplets have overlap with train triplet types. We carefully remove them and provide the updated list of zero-shot triplets. You can download it from this [link](TODO), and check the details via our repo [T-CAR](https://github.com/jkli1998/T-CAR).

### For GQA Dataset:
1. Download the GQA images [Full (20.3 Gb)](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). Extract these images to the file `datasets/gqa/images`. 
2. In order to achieve a representative split like VG150, we use the protocol provided by [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG). You can download the annotation file from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kwwKFbdBB3ZU3c49?e=06qeZc), and put all three files to  `datasets/gqa/`. 

### For Open Image V6 Dataset:
1. We use the dataset processed by [PySGG](https://github.com/SHTUPLUS/PySGG). You can download the processed dataset [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3). 
2. The dataset dir contains the `images` and `annotations` folder.



