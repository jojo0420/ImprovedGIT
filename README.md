# ImprovedGIT
To get FID, run below

```
pip install pytorch-fid
python3 -m pytorch_fid --device cuda:0 path/to/origin_image path/to/processed_image
```

To get IS, run below
```
pip install pytorch-gan-metrics
from pytorch_gan_metrics import get_inception_score
IS, IS_std = get_inception_score(images)
```
