# U2DP: Unlocking Unlabeled Data Potential for Semi-Supervised Remote Sensing Image Captioning

This repository includes the implementation for paper "U2DP: Unlocking Unlabeled Data Potential for Semi-Supervised Remote Sensing Image Captioning".

## Requirements

- Python 3.6
- Java 1.8.0
- PyTorch 1.10
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)

## Training AoANet With U2DP

### Prepare data

See details in `data/README.md`.
You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json ./dataset/dataset_bu/UCM_captions/0.2/data.json --dict_json ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk.json --output_pkl ./dataset/dataset_bu/UCM_captions/0.2/ucm-train --split train
```

Run the following command to extract clip image features:

```bash
$ python scripts/pre_CLIP_feature.py
```

### Start training

```bash
$ CUDA_VISIBLE_DEVICES=0 sh train.sh
```

See `opts.py` for the options.

### Evaluation

```bash
$ CUDA_VISIBLE_DEVICES=0 sh eval.sh
```
## Acknowledgements

This repository is based on [AoANet](https://github.com/husthuaan/AoANet), and you may refer to it for more details about the code.

