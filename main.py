from torch import nn, optim
from torch.optim import lr_scheduler

from TransformerNet import Semi_Transformer
import numpy as np

train = np.array([1,1],[2.7])
model = Semi_Transformer(num_classes=2) # seq_len : frame number

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)