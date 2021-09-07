import numpy as np
import pandas as pd
from parameters import *
from trainer import Trainer

par = Params()
par.data.cs_sample = CSSAMPLE.VILK

# train
trainer = Trainer(par)
trainer.launch_training_expanding_window()
trainer.create_paper()
