import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# matplotlib.use('Agg')
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
#exponencial
def plot_scheduler2(step, schedulers):
    if not isinstance(schedulers, list):
        schedulers = [schedulers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 3])
    for scheduler in schedulers:
        x = range(step)
        y = [scheduler(i).numpy() for i in x]
        ax1.plot(x, y, label=scheduler.name)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.legend()

        ax2.plot(x, y, label=scheduler.name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend()
    plt.show()  #
#linear
def plot_scheduler(step, schedulers):
    if not isinstance(schedulers, list):
        schedulers = [schedulers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 3])
    for scheduler in schedulers:
        ax1.plot(range(step), scheduler(range(step)), label=scheduler.name)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.legend()

        ax2.plot(range(step), scheduler(range(step)), label=scheduler.name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend()
    plt.show()

def otimizerSelector(Otimizer_type,Lrate):
    initial_learning_rate=0.001
    if (Lrate == 'CosineDecayRestarts'):
        first_decay_steps = 70
        Lrate = (tf.keras.experimental.CosineDecayRestarts(initial_learning_rate,first_decay_steps))
    elif (Lrate == 'CosineDecay'):
        decay_steps = 100
        Lrate = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)
    elif (Lrate == 'CyclicalLearningRate'):
        INIT_LR = 1e-4
        MAX_LR = 1e-2
        Lrate = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,maximal_learning_rate=MAX_LR,scale_fn=lambda x: 1 / (2. ** (x - 1)),step_size=15)
    elif(Lrate == 'WarmUpCosine'): #não funciona rever calculos das variaveis
        ARTIFICIAL_EPOCHS = 1000
        ARTIFICIAL_BATCH_SIZE = 512
        DATASET_NUM_TRAIN_EXAMPLES = 97000
        TOTAL_STEPS = int(
            DATASET_NUM_TRAIN_EXAMPLES / ARTIFICIAL_BATCH_SIZE * ARTIFICIAL_EPOCHS
        )
        Lrate = WarmUpCosine(
            learning_rate_base=INIT_LR,
            total_steps=TOTAL_STEPS,
            warmup_learning_rate=0.0,
            warmup_steps=1500,
        )
        # lrs = [Lrate(step) for step in range(TOTAL_STEPS)]
        # plt.plot(lrs)
        # plt.xlabel("Step", fontsize=14)
        # plt.ylabel("LR", fontsize=14)
        # plt.show()
    else:
        Lrate = 0.001
    # plot_scheduler(200, [Lrate])

    if (Otimizer_type == 'Adam'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=Lrate)
    elif (Otimizer_type == 'RMSprop'):
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=Lrate)
    elif (Otimizer_type == 'SGD'):
        optimizer=tf.keras.optimizers.SGD(learning_rate=Lrate,momentum=0.9,name='SGD',nesterov=True)
    elif (Otimizer_type == 'SGDW'):
        optimizer = tfa.optimizers.SGDW(learning_rate=Lrate, weight_decay=0.001, momentum=0.9,nesterov=True)
    elif (Otimizer_type == 'RectifiedAdam'):
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=Lrate)
    elif (Otimizer_type == 'RectifiedAdamWarmup'):
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=Lrate,total_steps=10000,warmup_proportion=0.1,min_lr=1e-5)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=Lrate)

    PosOtimizer ='xxx'
    if (PosOtimizer== 'SWA'):
        optimizer = tfa.optimizers.SWA(optimizer, 50, 10)
    elif (PosOtimizer== 'Lookahead'):
        optimizer = tfa.optimizers.Lookahead(optimizer)
    elif(PosOtimizer== 'Lookahead2'):
        optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=6, slow_step_size=0.5)


    return optimizer


