import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from tqdm import tqdm

import vfn.networks.backbones as backbones
from configs.parser import ConfigParser
from vfn.networks.models import ViewFindingNet
from vfn.networks.losses import ranknet_loss, svm_loss
from vfn.data.datasets.FlickrPro import FlickrPro


def get_data_loaders(train_batch_size, val_batch_size, input_dim):
    data_transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_dim)
    ])
    data_loaders = dict(
        train=DataLoader(FlickrPro(root_dir='../raw_images',
                                   transforms=data_transfrom),
                         train_batch_size,
                         shuffle=True),
        val=DataLoader(FlickrPro(root_dir='../raw_images',
                                 transforms=data_transfrom),
                       val_batch_size,
                       shuffle=False),
    )
    return data_loaders


def run():
    num_epochs = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = backbones.AlexNet()  # type: backbones.Backbone
    batch = dict(
        train_batch_size=100,
        val_batch_size=100,
    )
    data_loaders = get_data_loaders(**batch, input_dim=backbone.input_dim())
    model = ViewFindingNet(backbone)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = svm_loss
    metric = dict(
        loss=Loss(svm_loss)
    )
    desc = 'EPOCH - loss: {:.4f}'
    pbar = tqdm(initial=0, leave=False, total=len(data_loaders['train']), desc=desc.format(0))
    log_interval = 10

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metric, device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(data_loaders['train']) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(data_loaders['train'])
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        tqdm.write(
            'Training Results - Epoch: {}\n'
            'Avg loss: {:.4f}\n'
            .format(engine.state.epoch, avg_loss))
        pbar.n = pbar.last_print_n = 0

    trainer.run(data_loaders['train'], num_epochs)


if __name__ == '__main__':
    run()
