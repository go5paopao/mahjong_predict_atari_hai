import math


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    assert(len(lr) == 1)
    lr = lr[0]
    return lr


class NullScheduler():
    def __init__(self, lr=0.01):
        super(NullScheduler, self).__init__()
        self.lr = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = "NullScheduler\n" \
            + "lr={0:0.5f}".format(self.lr)
        return string


class ManualScheduler():
    def __init__(self, lr=0.01, lr_decay=0.9):
        super(ManualScheduler, self).__init__()
        self.lr_list = [lr * (lr_decay ** i) for i in range(100)]
        self.cycle = 0

    def __call__(self, time):
        if time < len(self.lr_list):
            return self.lr_list[time]
        else:
            return self.lr_list[-1]

    def __str__(self):
        string = "ManualScheduler\n" \
            + "lr={0:0.5f}".format(self.lr_list[0])
        return string


class CosineAnnealingScheduler():
    def __init__(self, eta_min=0.0001, eta_max=0.002, cycle=100, repeat=False):
        super(CosineAnnealingScheduler, self).__init__()
        self.cycle = cycle
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.lr = self.eta_min + (self.eta_max - self.eta_min)
        self.repeat = repeat

    def __call__(self, epoch):
        if self.repeat:
            self.lr = self.eta_min + \
                (self.eta_max - self.eta_min) \
                * (1 + math.cos(math.pi * epoch / self.cycle)) / 2
        else:
            if epoch <= self.cycle:
                self.lr = self.eta_min + \
                    (self.eta_max - self.eta_min) \
                    * (1 + math.cos(math.pi * epoch / self.cycle)) / 2
            else:
                self.lr = self.eta_min
        return self.lr

    def __str__(self):
        string = 'CosineAnealingScheduler\n' \
                + 'lr=%0.5f ' % (self.lr)
        return string


