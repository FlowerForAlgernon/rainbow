class Config(object):
    def __init__(self):
        self.device = None

        self.c = None
        self.h = None
        self.w = None
        self.n_actions = None

        self.memory_size = 100000
        self.batch_size = 32

        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6

        self.n_step = 3

        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51

        self.epsilon_max = 1
        self.epsilon_min = 0.01
        self.eps_decay = 30000

        self.gamma = 0.99
        self.learning_rate = 2e-4
        self.learning_start = 10000
        self.target_update = 1000
        self.n_episodes = 1000
