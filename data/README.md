# Prepare data

## UCM-Captions

### Download UCM-Captions and preprocess them

Download preprocessed UCM-Captions from [link](https://pan.baidu.com/s/1mjPToHq).

Then  pre-extract the image features refer to [link](https://github.com/peteanderson80/bottom-up-attention).

### Split Unlabeled Data

Run the following command to split the unlabeled data:

```bash
$ python scripts/random_set_unlabel.py ./dataset/UCM_captions/dataset.json ./dataset/dataset_bu/UCM_captions/0.2 0.8
```

Then run:

```bash
$ python scripts/prepro_labels.py \
    --input_json ./dataset/dataset_bu/UCM_captions/0.2/data.json \
    --output_json ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk.json \
    --output_h5 ./dataset/dataset_bu/UCM_captions/0.2/ucmtalk_label.h5 \
    --word_count_threshold 5 \
    --max_length 50
```
