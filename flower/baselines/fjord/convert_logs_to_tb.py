import re
import os
from torch.utils.tensorboard import SummaryWriter

base_dir = "/Users/zeevweizmann/projects/federated_learning/flower/baselines/fjord/runs"

experiments = {
    "no_KD": "2025-11-12:11-08-51/run.log",
    "with_KD": "2025-11-12:11-23-50/run.log",
}

for name, log_path in experiments.items():
    full_path = os.path.join(base_dir, log_path)
    with open(full_path) as f:
        text = f.read()

    # Извлечём accuracy по каждому p
    pattern = r"p=(\d\.\d):.*accuracy=(\d+\.\d+)"
    matches = re.findall(pattern, text)

    writer = SummaryWriter(log_dir=os.path.join(base_dir, name))
    for p_str, acc_str in matches:
        p = float(p_str)
        acc = float(acc_str)
        writer.add_scalar("Accuracy/final", acc, int(p * 10))  # шаг = p*10
    writer.close()
    print(f" Written TensorBoard logs for {name}")
