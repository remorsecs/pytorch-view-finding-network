import argparse
import numpy as np
import torch
import torch.cuda

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from tqdm import tqdm
from visdom import Visdom

from configs.parser import ConfigParser
from vfn.utils.visualization import plot_bbox


class Trainer:

    def __init__(self, configs):
        self.configs = configs
        self._init_config_settings()
        self._init_logger()
        self._init_trainer()
        self._init_validator()

    def _init_config_settings(self):
        self.device = self.configs.parse_device()
        self.num_epochs = self.configs.configs['train']['num_epochs']
        self.model_name = self.configs.get_model_name()
        self.model = self.configs.parse_model().to(self.device)
        self.data_loaders = self.configs.parse_dataloader()
        self.optimizer = self.configs.parse_optimizer()
        self.optimizer = self.optimizer(self.model.parameters())
        self.loss_fn = self.configs.parse_loss_function()

    def _init_logger(self):
        self.desc = 'Loss: {:.6f}'
        self.pbar = tqdm(
            initial=0,
            leave=False,
            total=len(self.data_loaders['train']),
            desc=self.desc.format(0),
            ascii=True,
        )
        self.log_interval = 1
        self.vis = Visdom()

    def _init_trainer(self):
        self.trainer = Engine(self._inference)
        self.trainer.add_event_handler(Events.EPOCH_STARTED, self._set_model_stage, is_train=True)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_iteration, is_train=True)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch, is_train=True)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._run_validation)
        ckpt_handler = ModelCheckpoint(
            dirname=self.configs.configs['checkpoint']['root_dir'],
            filename_prefix=self.configs.configs['checkpoint']['prefix'],
            save_interval=1,
            n_saved=self.num_epochs,
            require_empty=False,
        )
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler, {self.model_name: self.model})

    def _init_validator(self):
        self.validator = Engine(self._inference)
        self.validator.add_event_handler(Events.EPOCH_STARTED, self._set_model_stage)
        self.validator.add_event_handler(Events.ITERATION_COMPLETED, self._log_iteration)
        self.validator.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)

    def _set_model_stage(self, engine, is_train=False):
        self.model.train(is_train)
        torch.set_grad_enabled(is_train)
        engine.state.is_train = is_train
        engine.state.cum_average_loss = 0

    def _inference(self, engine, batch):
        # fetch inputs and transfer to specific device
        image_raw, image_crop = batch
        image_raw, image_crop = image_raw.to(self.device), image_crop.to(self.device)

        # forward
        score_I = self.model(image_raw)
        score_C = self.model(image_crop)
        loss = self.loss_fn(score_I, score_C)

        # backward
        if engine.state.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        engine.state.iteration_loss = loss.mean().item()
        engine.state.cum_average_loss += loss.mean().item()

    def _log_iteration(self, engine, is_train=False):
        average_loss = engine.state.cum_average_loss / engine.state.iteration
        self.pbar.desc = self.desc.format(average_loss)
        self.pbar.update(self.log_interval)

        if is_train:
            self.vis.line(
                Y=np.array([self.trainer.state.iteration_loss]),
                X=np.array([self.trainer.state.iteration]),
                win='loss-iteration',
                env=self.configs.configs['checkpoint']['prefix'],
                update='append',
                name='train',
                opts=dict(
                    title='Learning Curve',
                    showlegend=True,
                    xlabel='Iteration',
                    ylabel='Loss',
                )
            )

    def _log_epoch(self, engine, is_train=False):
        stage = 'Training' if is_train else 'Validation'
        self.pbar.refresh()
        if is_train:
            tqdm.write('Epoch {}:'.format(self.trainer.state.epoch))

        average_loss = engine.state.cum_average_loss / engine.state.iteration
        tqdm.write('{} Loss: {:.6f}'.format(stage, average_loss))
        self.pbar.n = self.pbar.last_print_n = 0
        self.vis.line(
            Y=np.array([average_loss]),
            X=np.array([self.trainer.state.epoch]),
            win='loss-epoch',
            env=self.configs.configs['checkpoint']['prefix'],
            update='append',
            name=stage,
            opts=dict(
                title='Learning Curve',
                showlegend=True,
                xlabel='Epoch',
                ylabel='Loss',
            )
        )

    def _run_validation(self, engine):
        self.validator.run(self.data_loaders['val'])

    def run(self):
        self.trainer.run(self.data_loaders['train'], self.num_epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to config file (.yml)', default='../configs/example.yml')
    args = parser.parse_args()

    configs = ConfigParser(args.config_file)
    trainer = Trainer(configs)
    trainer.run()


if __name__ == '__main__':
    main()
