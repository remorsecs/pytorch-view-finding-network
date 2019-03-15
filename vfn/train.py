import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from tqdm import tqdm

import vfn.networks.backbones as backbones
from configs.parser import ConfigParser
from vfn.networks.models import ViewFindingNet
from vfn.networks.losses import ranknet_loss, svm_loss
from vfn.data.datasets.FlickrPro import FlickrPro


def get_data_loaders(train_batch_size, val_batch_size, input_dim, train_size):
    data_transform = transforms.Compose([
        transforms.Resize(input_dim),
        transforms.ToTensor(),
    ])

    dataset = FlickrPro(root_dir=kwargs['root_dir'], transforms=data_transform)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    data_loaders = dict(
        train=DataLoader(train_dataset, train_batch_size, shuffle=True),
        val=DataLoader(val_dataset, val_batch_size, shuffle=False),
    )
    return data_loaders


def run():
    NUM_EPOCHS = kwargs['num_epochs']
    BATCH_TRAIN = kwargs['batch_train']
    BATCH_VAL = kwargs['batch_val']
    TRAIN_SIZE = kwargs['train_size']
    LR = kwargs['learning_rate']
    MOMENTUM = kwargs['momentum']
    RANKING_LOSS = kwargs['ranking_loss']
    GPU_ID = kwargs['gpu_id']

    device = 'cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else 'cpu'
    backbone = backbones.AlexNet()  # type: backbones.Backbone
    batch = dict(
        train_batch_size=BATCH_TRAIN,
        val_batch_size=BATCH_VAL,
    )
    data_loaders = get_data_loaders(**batch, input_dim=backbone.input_dim(), train_size=TRAIN_SIZE)
    model = ViewFindingNet(backbone).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = None
    if RANKING_LOSS == 'ranknet':
        loss_fn = ranknet_loss
    elif RANKING_LOSS == 'svm':
        loss_fn = svm_loss

    metric = dict(
        loss=Loss(loss_fn)
    )
    desc = 'EPOCH - loss: {:.4f}'
    pbar = tqdm(initial=0, leave=False, total=len(data_loaders['train']), desc=desc.format(0), ascii=True)
    log_interval = 10

    def step(engine, batch):
        # initialize
        model.train()
        optimizer.zero_grad()

        # fetch inputs and transfer to specific device
        image_raw, image_crop = batch
        image_raw, image_crop = image_raw.to(device), image_crop.to(device)

        # forward
        score_I = model(image_raw)
        score_C = model(image_crop)

        # backward
        loss = loss_fn(score_I, score_C)
        loss.backward()
        optimizer.step()

        return dict(
            score_I=score_I,
            score_C=score_C,
            loss=loss.item()
        )

    trainer = Engine(step)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # fetch inputs and transfer to specific device
            image_raw, image_crop = batch
            image_raw, image_crop = image_raw.to(device), image_crop.to(device)

            # forward
            score_I = model(image_raw)
            score_C = model(image_crop)

            loss = loss_fn(score_I, score_C)

            return dict(
                score_I=score_I,
                score_C=score_C,
                loss=loss.item()
            )

    evaluator = Engine(inference)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(data_loaders['train']) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output['loss'])
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        tqdm.write(
            'Training Results - Epoch: {}\n'
            'Loss: {:.4f}\n'.format(engine.state.epoch, engine.state.output['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(data_loaders['val'])
        tqdm.write(
            'Validation Results - Epoch: {}\n'
            'Loss: {:.4f}\n'.format(engine.state.epoch, evaluator.state.output['loss']))

        pbar.n = pbar.last_print_n = 0

    trainer.run(data_loaders['train'], NUM_EPOCHS)


if __name__ == '__main__':
    # TODO: parse YAML by ConfigParser
    kwargs = dict(
        num_epochs=200,
        batch_train=100,
        batch_val=14,
        train_size=17000 * 14,
        learning_rate=0.01,
        momentum=0.9,
        ranking_loss='ranknet',
        root_dir='../raw_images/flickr_pro',
        gpu_id=0,
    )
    run()
