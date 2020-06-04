# pytorch-view-finding-network

This is a PyTorch implementation of the [view finding network](https://github.com/yiling-chen/view-finding-network) method.

## Getting Started

### Installation

```bash
git clone https://github.com/remorsecs/pytorch-view-finding-network

cd pytorch-view-finding-network/

pip install -r requirements.txt

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
