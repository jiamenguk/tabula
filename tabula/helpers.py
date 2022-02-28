import os
from pathlib import Path

import torch


class Helper():

    def epoch_start(self, data, metadata):
        pass

    def epoch_end(self, data, metadata):
        pass

    def iter_start(self, data, metadata):
        pass

    def iter_end(self, data, metadata):
        pass


class CheckpointHelper(Helper):
    save_dir = Path("checkpoints")

    def __init__(self, exp_name, checkpoint_dict, save_epoch=False, save_iters=None, only_save_last=True):

        assert save_iters is None or isinstance(save_iters, int), "save_iters must be None or int"

        self.exp_name = exp_name
        self.checkpoint_dict = checkpoint_dict
        self.save_epoch = save_epoch
        self.save_iters = save_iters
        self.only_save_last = only_save_last

    def epoch_end(self, data, metadata):
        if self.save_epoch:
            if self.only_save_last:
                fname = "lastmodel.pt"
            else:
                fname = f"epoch_{metadata['epoch']}.pt"
            self._save(data, metadata, fname)

    def iter_end(self, data, metadata):
        iteration = metadata['iters']
        if self.save_iters is not None and iteration > 0 and iteration % self.save_iters == 0:
            fname = f'iter_{iteration}.pt'
            self._save(data, metadata, fname)

    def _save(self, data, metadata, fname):
        save_dict = {
            'metadata': metadata,
            'checkpoint': {k: v.state_dict() for k, v in self.checkpoint_dict.items()},
        }
        checkpoint_path = self.checkpoint_dir / fname
        print(f"Saving checkpoint at {checkpoint_path}")
        torch.save(save_dict, checkpoint_path)

    def load(self, checkpoint_path):

        assert os.path.isfile(checkpoint_path)

        loaded_dict = torch.load(checkpoint_path, map_location='cpu')

        for k, v in self.checkpoint_dict.items():
            try:
                v.load_state_dict(loaded_dict['checkpoint'][k])
            except KeyError as e:
                print(e)

        return loaded_dict['metadata']

    @property
    def checkpoint_dir(self):
        checkpoint_dir = self.save_dir / self.exp_name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        return checkpoint_dir


class SubslateHelper(Helper):
    def __init__(self, slate, dataloader, run_epoch=False, run_iters=None):

        assert run_iters is None or isinstance(run_iters, int), "save_iters must be None or int"

        self.slate = slate
        self.dataloader = dataloader
        self.run_epoch = run_epoch
        self.run_iters = run_iters

    def _run(self):
        if not self.slate.train:
            self.slate.model.eval()
        self.slate.run(self.dataloader, max_epochs=1)
        if not self.slate.train:
            self.slate.model.train()

    def epoch_end(self, data, metadata):
        if self.run_epoch:
            self._run()

    def iter_end(self, data, metadata):
        iteration = metadata['iters']
        if self.run_iters is not None and iteration > 0 and iteration % self.run_iters == 0:
            self._run()
