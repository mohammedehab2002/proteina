import torch
from proteinfoundation.proteinflow.proteina import Proteina
from tqdm import tqdm

model = Proteina.load_from_checkpoint("./checkpoints/proteina_v1.1_DFS_200M_tri.ckpt", strict=False)

epochs = 1000

for epoch in tqdm(range(epochs)):
    N = 50
    B = 25