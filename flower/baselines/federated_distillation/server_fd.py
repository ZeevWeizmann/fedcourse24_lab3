import flwr as fl
from typing import List, Tuple
import numpy as np

# -------------------------------
# Custom strategy for Federated Distillation
# -------------------------------
class FedDistillationStrategy(fl.server.strategy.FedAvg):
    """Custom strategy for Federated Distillation."""

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation results by averaging logits and accuracy."""

        logits_list = []
        accuracies = []

        for _, evaluate_res in results:
            # check if metrics exist and contain our keys
            metrics = evaluate_res.metrics
            if metrics is not None:
                if "avg_logits" in metrics:
                    logits_list.append(metrics["avg_logits"])
                if "accuracy" in metrics:
                    accuracies.append(metrics["accuracy"])

        if logits_list:
            avg_logits = float(np.mean(logits_list))
            print(f"[Server] Round {rnd}: averaged logits = {avg_logits:.6f}")
        else:
            print(f"[Server] Round {rnd}: No logits received!")

        if accuracies:
            mean_acc = float(np.mean(accuracies))
            print(f"[Server] Round {rnd}: mean accuracy = {mean_acc:.4f}")
        else:
            print(f"[Server] Round {rnd}: No accuracy reported!")

        return super().aggregate_evaluate(rnd, results, failures)
if __name__ == "__main__":
    strategy = FedDistillationStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
