# faster-view-finding-network

## Getting Started

### Prerequisites

- Python 3.7
- PyTorch 1.0

### Installation

```bash
python setup.py build develop
```

### Usage

Before training, start Visdom server for visualization from the command 
line:

```bash
python -m visdom.server &
```

It will output PID. Visdom server can be accessed by going to 
`http://localhost:8097` in browser. 

Train and test:

```bash
python -m vfn.train --config_file='configs/example.yml'
python -m vfn.evaluate --config_file='configs/example.yml'
```

After training, you need to stop the visdom server:

```bash
kill -9 PID
```
