# faster-view-finding-network

## Getting Started

### Prerequisites

- Python 3.6
- PyTorch >= 1.0
    - https://github.com/pytorch/pytorch
    - https://pytorch.org/get-started/locally/
- torchvision >= 0.3.0
    - https://github.com/pytorch/vision
- visdom
    - `pip install visdom`
    - https://github.com/facebookresearch/visdom
- pytorch-ignite
    - `pip install pytorch-ignite`
    - https://github.com/pytorch/ignite
- GulpIO
    - `pip install gulpio`
    - https://github.com/TwentyBN/GulpIO
    - **Note**: GulpIO depends on Python 3.3~3.6, you might encounter an installation error on other Python versions.
- PyYAML
    - `pip install PyYAML`
    - https://github.com/yaml/pyyaml
- opencv-python
    - `pip install opencv-python`

### Installation

```bash
git clone https://github.com/yiling-chen/faster-view-finding-network

cd faster-view-finding-network/

python setup.py build develop
```

### Usage

#### Configuration

The configuration file will be parsed by class `ConfigParser` in `viewfinder_benchmark/config/parser.py`.

You can follow the `configs/prototype.yml` or take the `configs/example.yml` as reference. 

The supported format is visible in `configs/README.md`.  


#### Start Visdom Server

We use Visdom for data visualization.

Start a Visdom server from the command line before training:

```bash
visdom
```

It will launch a Visdom server and output PID. The Visdom server can be accessed by going to 
`http://localhost:8097` (default port) in browser.

You can visit the [official site](https://github.com/facebookresearch/visdom#usage)
for more information.


#### Train

```bash
cd tools/

python train.py -c '../configs/example.yml'
```

#### Evaluate
 
```bash
cd tools/

python evaluate.py -c '../configs/example.yml'
```
