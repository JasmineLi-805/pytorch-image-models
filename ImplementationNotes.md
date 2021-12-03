# Saliency Map + Resnet 18 Implementation

## Data Preprocessing
[x] gray scale
[x] cropping and stacking
[x] downsizing

## Implementation Notes
### Define a new model
[x] Add a new module in `./timm/models`
[x] Register new model with `./timm/models/registry.py`
[x] Setup a default config

### Define ImageNet Proprocessing
[x] Add new dataset class to `timm/data/dataset.py`
[x] Add new dataset class to `timm/data/dataset_factory.py`
