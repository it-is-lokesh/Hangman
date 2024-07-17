import os
import torch

class config:
    def __init__(self):
        self.seed = 3407
        self.shuffle = True
        self.pin_memory = True
        self.nworkers = 8
        self.gpu = True
        self.device = 'cuda' if self.gpu and torch.cuda.is_available() else 'cpu'

        self.sequence_length = 20
        self.batch_size = 256
        self.lr = 1e-3
        self.epochs = 200
        self.step_size = 10
        self.gamma = 0.9
        self.test_size = 0.2

        self.input_size = 1
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1

        self.train_path = './storage'
        self.model_file_name = 'bilstm_' + '_'.join([str(self.hidden_size), str(self.num_layers)])
        self.model_path = os.path.join('storage', self.model_file_name)

        if not os.path.isdir('./storage'):
            os.mkdir('./storage')
        