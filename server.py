import numpy as np
import flwr as flwr
from model import TBML_DetectionModel

# --- GLOBAL CONFIGURATION ---
# 20 rounds gives the clients enough epochs to calibrate the embeddings and probabilities
TOTAL_ROUNDS = 20

class modelStrategy(flwr.server.strategy.FedProx):
    def aggregate_fit(self, curr_round, results, failures):
        # Process the FedProx aggregation
        aggregated_weights, metrics = super().aggregate_fit(curr_round, results, failures) 

        # --- SAVE ON THE FINAL ROUND ---
        if aggregated_weights is not None and curr_round == TOTAL_ROUNDS:
            print(f"💾 Saving final global weights for round {curr_round}...")

            # Convert Flower byte stream back to numpy arrays
            aggregated_array = flwr.common.parameters_to_ndarrays(aggregated_weights)

            # Save the numpy array to a file for production inference
            np.savez("global_model_final.npz", *aggregated_array)
            print("✅ Global weights saved successfully to 'global_model_final.npz'")

        return aggregated_weights, metrics


if __name__ == "__main__":
    print(f"🚀 Starting Flower server for {TOTAL_ROUNDS} global rounds...")

    # --- ARCHITECTURE ALIGNMENT ---
    # We drop the old dimension arguments because the new model handles 
    # the 21/75/12 input slice dimensions internally with nn.Embedding.
    model = TBML_DetectionModel(
        hidden_dim=64, 
        lstm_hidden=64, 
        classes=3
    )

    # Extract initial weights and convert to Flower parameters
    ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = flwr.common.ndarrays_to_parameters(ndarrays)

    # --- FEDPROX STABILIZER ---
    strategy = modelStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        proximal_mu=0.01  # The Rubber Band: Keeps Bank C from drifting and washing out Bank A/B
    )
    
    # Start the server
    flwr.server.start_server(
        server_address="0.0.0.0:8085",
        config=flwr.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy
    )