import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from configs.parser import ConfigParser


def train(model,
          criterion,
          optimizer,
          scheduler,
          dataloader,
          num_epochs,
          checkpointer,
          # TODO: logger,
          # TODO: visualizer,
          ):
    pass


def run_train():
    # TODO: parse config.yml model, criterion, optimizer, scheduler, dataloader, num_epochs, ...
    parser = ConfigParser('/path/to/config.yml')
    kwargs = parser.parse()
    train(**kwargs)     # maybe need `functool.partial`
    pass


if __name__ == '__main__':
    run_train()
