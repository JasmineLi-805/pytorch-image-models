# Saliency Map + Resnet 18 Implementation

## Data Preprocessing
- gray scale
- cropping and stacking
- downsizing (?)

## Implementation Notes
### Define a new model
- Add a new module in `./timm/models`
- Register new model with `./timm/models/registry.py`
- Setup a default config

### Define ImageNet Proprocessing
- Add new dataset class to `timm/data/dataset.py`
- Add new dataset class to `timm/data/dataset_factory.py`
