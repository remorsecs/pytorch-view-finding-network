# Configuration

## Supported Format

```yaml

---
checkpoint:
  root_dir: # Path to checkpoint root, type: str
  prefix:   # Prefix for each checkpoint file, type: str

weight:     # Path to trained model for evaluation, type: str 

device:     # Computing devices, follow the `torch.device` argument. type: str
            # See: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

model:
  backbone:
    name:   # Name for class in module `viewfinder_benchmark.network.backbones`, supports 'AlexNet', 'VGG', type: str
    pretrained:   # To load pretrained parameters, type: bool

train:
  num_epochs:     # Number of epochs, type: int   
  viz:            # Determine to launch the visdom server, type: bool

  optimizer:
    name:         # Name for class `torch.optim`, supports 'Adam', type: str
                  # The rest of arguments defined here will pass to `torch.optim`.

  loss:
    name:         # Name for loss function, supports 'hinge', type: str

  dataset:
    name:         # Name for dataset class, supports 'FlickrPro', type: str
    root_dir:     # Path to dataset root, type: str
    gulpio_dir:   # Path to GulpIO dataset root, type: str

  dataloader:
    # The arguments defined here will pass to `torch.utils.data.DataLoader`.

validation:
  viz:            # Determine to launch the visdom server, type: bool

  dataset:
    name:         # Name for dataset class, supports 'FlickrPro', type: str
    root_dir:     # Path to dataset root, type: str
    gulpio_dir:   # Path to GulpIO dataset root, type: str

  dataloader:
    # The arguments defined here will pass to `torch.utils.data.DataLoader`.

evaluate:
  FCDB:
    root_dir: # Path to FCDB dataset, type: str
    download: # Determine to download FCDB dataset, type: bool

  ICDB:
    root_dir: # Path to ICDB dataset, type: str
    download: # Determine to download ICDB dataset, type: bool
...

```