from functools import partial

import torch
from torch.utils import data


class DataFeature():

    def __init__(self):
        pass

    def feat(self, data):

        return data

    def collate(self, data):
        data = torch.tensor(data)
        return data


class Dataset(data.Dataset):

    def __init__(self, dataset, features, proc_fn=None):

        if proc_fn is not None:
            dataset = proc_fn(dataset)
        self.dataset = dataset
        self.features = features


    def __getitem__(self, idx):
        data = {key: feature.feat(self.dataset[idx]) for key, feature in self.features.items()}

        return data

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch, features):
    data = {key: feature.collate([data[key] for data in batch])
            for key, feature in features.items()}

    return data


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):

        _collate_fn = partial(collate_fn, features=dataset.features)
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
