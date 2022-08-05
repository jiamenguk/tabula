from typing import List, Optional

import torch
from tqdm import tqdm


def _to_gpu(data):
    if hasattr(data, 'to_gpu'):
        return data.to_gpu()
    elif isinstance(data, list):
        return [_to_gpu(i) for i in data]
    elif isinstance(data, dict):
        return {k: _to_gpu(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple([_to_gpu(i) for i in data])
    elif torch.is_tensor(data):
        return data.cuda()
    else:
        return data


class Slate():
    epoch = None
    iters = None
    helpers = []
    train = True

    def __init__(self):

        self.epoch = 0
        self.iters = 0
        self._stop = False
        self._data = None

    def step(self, data):

        raise NotImplementedError

    def run(self, dataloader, max_epochs: Optional[int] = None,
            max_iters: Optional[int] = None):
        self._run(dataloader, max_epochs, max_iters)

    def _run(self, dataloader, max_epochs, max_iters):
        self._stop = False

        while not self._stop:
            self.epoch += 1

            if self.helpers is not None:
                for helper in self.helpers:
                    helper.epoch_start(data=self.data, metadata=self.metadata)

            data_enum = tqdm(dataloader)
            for batch_data in data_enum:
                self.iters += 1

                if self.helpers is not None:
                    for helper in self.helpers:
                        helper.iter_start(data=self.data, metadata=self.metadata)

                batch_data = _to_gpu(batch_data)

                loss_dict, self._data = self.step(batch_data)

                message = [f"{k}: {v:.6f}" for k, v in loss_dict.items()]
                message = ", ".join(message)
                message = f"Epoch {self.epoch} Iter {self.iters} " + message
                data_enum.set_description(message)

                if self.helpers is not None:
                    for helper in self.helpers:
                        helper.iter_end(data=self.data, metadata=self.metadata)

                if max_iters is not None and self.iters >= max_iters:
                    self._stop = True
                    break

            if self.helpers is not None:
                for helper in self.helpers:
                    helper.epoch_end(data=self.data, metadata=self.metadata)

            if max_epochs is not None and self.epoch >= max_epochs:
                self._stop = True

        self.shutdown()

    def shutdown(self):
        pass

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return {
            'epoch': self.epoch,
            'iters': self.iters,
        }
