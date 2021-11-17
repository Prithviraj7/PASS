## Introduction

This repository contains the source code for training PASS-g and PASS-s using features from a pre-trained model.

- [Dhar, Prithviraj, et al. "PASS: Protected Attribute Suppression System for Mitigating Bias in Face Recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.](https://openaccess.thecvf.com/content/ICCV2021/papers/Dhar_PASS_Protected_Attribute_Suppression_System_for_Mitigating_Bias_in_Face_ICCV_2021_paper.pdf)

BibTeX:
```
@InProceedings{Dhar_Gleason_2021_ICCV,
    author    = {Dhar, Prithviraj and Gleason, Joshua and Roy, Aniket and Castillo, Carlos D. and Chellappa, Rama},
    title     = {{PASS}: Protected Attribute Suppression System for Mitigating Bias in Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15087-15096}
}
```

## Running The Code

Requirements are defined in `requirements.txt` and may be installed in a new virtual environment using

```
pip install -r requirements.txt
```

An example configuration is defined in [`config/config_template.yaml`](https://github.com/Prithviraj7/PASS/blob/main/config/config_template.yaml).

In the config file set `TYPE:'race'` for PASS-s or `TYPE:'gender'` for PASS-g.

### Required Input Files

#### Training features (`train.py`)

This file should be provided in the `TRAIN_BIN_FEATS` and `VAL_BIN_FEATS` config entries. Must be a binary file. Given a numpy array of `N` 512-dimensional features you can create this file using the following snippet (note we assume binary file created with same byte order as system used to train)

```python
import numpy as np
import struct

# feat = ... (load features into np.ndarray of shape [N, 512])
# ...

with open('input_features.bin', 'wb') as f:
    f.write(struct.pack('i', np.int32(N)))
    f.write(struct.pack('i', np.int32(512)))
    np.ascontiguousarray(feat).astype(np.float32).tofile(f)
```

#### Training metadata (`train.py`)

This file should be provided in the `TRAIN_META` and `VAL_META` config entries. This CSV file must contain information about each training feature (one-to-one corresponding) and must contain the following columns:

```none
SUBJECT_ID,FILENAME,RACE,PR_MALE
```

- `SUBJECT_ID` is an integer corresponding to subject
- `FILENAME` is original filename that feature was extracted from (not used currently)
- `RACE` is an integer representing a BUPT class label between 0 and 3 with {0: asian, 1: caucasian, 2: african, 3: indian}
- `PR_MALE` is a float between 0 and 1 representing probability that subject is male

Note that for PASS-g `RACE` may be omitted and for PASS-s `PR_MALE` may be omitted.

#### Test features (`inference.py`)

CSV file containing features to perform debiasing on after training is finished with following columns:

```none
SUBJECT_ID,FILENAME,DEEPFEATURE_1,...,DEEPFEATURE_512
```

where `DEEPFEATURE_*` contains the value of the input feature at the specified dimension.

---

To run PASS training execute

```
python train.py
```

To generate debiased features, select the desired checkpoint file and update `CHECKPOINT_FILE` in the config then run

```
python inference.py
```
