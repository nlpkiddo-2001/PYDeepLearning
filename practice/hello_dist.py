import os

import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

t = torch.tensor([rank], device="cuda")
print(f"rank {rank} before : {t}")

dist.all_reduce(tensor=t)

print(f"rank {rank} after: {t}")

dist.destroy_process_group()
