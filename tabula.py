

class Slate():
    epoch = None
    iters = None

    def __init__(self, dataloader, helpers=None):
        self.dataloader = dataloader

        if list is not None:
            assert type(helpers) == list, "Helpers must be in a list"
        self.helpers = helpers

        self.epoch = 0
        self.iters = 0
        self._stop = False

    def step(self, model, data):

        raise NotImplementedError

    def run(self, model, epochs=None, iters=None):
        self._run(model, epochs, iters)

    def _run(self, model, epochs, iters):
        self._stop = False

        while not self._stop:

            if self.helpers is not None:
                for helper in helpers:
                    helper.epoch_start(self.data, model)

            for batch_data in self.dataloader:
                if self.helpers is not None:
                    for helper in helpers:
                        helper.iter_start(self.data, model)

                self._data = self.step(model, batch_data)

                if self.helpers is not None:
                    for helper in helpers:
                        helper.iter_end(self.data, model)

                self.iters += 1

                if iters is not None and self.iters > iters:
                    self._stop = True
                    break

            if self.helpers is not None:
                for helper in helpers:
                    helper.epoch_end(self.data, model)

            epoch += 1
            if epoch is not None and self.epoch > epochs:
                self._stop = True

        self.shutdown()

    def shutdown(self):
        pass

    @property
    def data(self):
        return {
            'info': {
                'epoch': self.epoch,
                'iter': self.iters,
            },
            'data': self._data,
        }
