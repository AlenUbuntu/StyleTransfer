import torch.optim as optim

def build_optimizer(cfg, model):
    if cfg.OPTIMIZER.NAME == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            rho=cfg.OPTIMIZER.ADADELTA_RHO,
            eps=cfg.OPTIMIZER.ADADELTA_EPS
        )
    elif cfg.OPTIMIZER.NAME == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            initial_accumulator_value=cfg.OPTIMIZER.ADAGRAD_INITIAL_ACCUMULATOR_VALUE
        )
    elif cfg.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=cfg.OPTIMIZER.ADAM_BETAS,
            eps=cfg.OPTIMIZER.ADAM_EPS
        )
    elif cfg.OPTIMIZER.NAME == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.RMSPROP_DECAY,
            momentum=cfg.OPTIMIZER.RMSPROP_MOMENTUM,
            eps=cfg.OPTIMIZER.RMSPROP_EPS
        )
    elif cfg.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            momentum=cfg.OPTIMIZER.SGD_MOMENTUM
        )
    else:
        raise NotImplementedError('Optimizer {} is not supported yet.'.format(cfg.OPTIMIZER.NAME))

    return optimizer


class LambdaLRScheduler(object):
    def __init__(self, lr, lr_decay, optimizer):
        super(LambdaLRScheduler, self).__init__()
        self.optimizer = optimizer
        self.iteration = 0
        self.lr = lr
        self.lr_decay = lr_decay
    
    def step(self):
        self.iteration += 1
        self.adjust_learning_rate()

    def adjust_learning_rate(self):
        """Imitating the original implementation"""
        lr =  self.lr / (1.0 + self.lr_decay * self.iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def build_lr_scheduler(cfg, optimizer):
    if cfg.OPTIMIZER.LR_SCHEDULER.NAME == 'lambda_lr':
        scheduler = LambdaLRScheduler(
            lr=cfg.OPTIMIZER.BASE_LR,
            lr_decay=cfg.OPTIMIZER.LR_SCHEDULER.LR_DECAY,
            optimizer=optimizer
        )
    elif cfg.OPTIMIZER.LR_SCHEDULER.NAME == 'cosine_annealing_warm_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.OPTIMIZER.MAX_ITER,
            T_mult=cfg.OPTIMIZER.LR_SCHEDULER.T_MULT
        )
    else:
        raise NotImplementedError('lr scheduler {} is not supported yet.'.format(cfg.OPTIMIZER.LR_SCHEDULER.NAME))

    return scheduler
