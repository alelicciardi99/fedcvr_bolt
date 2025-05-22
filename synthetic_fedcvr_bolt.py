from sklearn.cluster import KMeans, SpectralClustering
from torch.utils.data import TensorDataset
import random
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
#SEED = 42
set_seed(42)
def generate_synthetic_data(K=20, D=61, num_classes=10, num_samples_per_client=200, alpha=0.5, beta=0.5, iid=False, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = D - 1  
    Sigma_diag = np.array([j**-1.2 for j in range(1, input_dim + 1)])
    Sigma = np.diag(Sigma_diag)

    # Shared W and b for IID case
    shared_W = np.random.normal(0, 1, (num_classes, input_dim))
    shared_b = np.random.normal(0, 1, num_classes)

    client_data = {}

    for k in range(K):
        # Model heterogeneity: generate uk ∼ N(0, alpha)
        uk = np.random.normal(0, np.sqrt(alpha), num_classes)

        if iid:
            Wk = shared_W
            bk = shared_b
        else:
            Wk = np.random.normal(loc=uk[:, np.newaxis], scale=1.0, size=(num_classes, input_dim))
            bk = np.random.normal(loc=uk, scale=1.0)

        # Data heterogeneity: Bk ∼ N(0, beta)
        Bk = np.random.normal(0, np.sqrt(beta), input_dim)
        vk = np.random.normal(loc=Bk, scale=1.0)

        # Sample data points: xk ∼ N(vk, Sigma)
        Xk = np.random.multivariate_normal(mean=vk, cov=Sigma, size=num_samples_per_client).astype(np.float32)

        # Generate softmax labels
        logits = np.dot(Xk, Wk.T) + bk
        probs = torch.softmax(torch.tensor(logits), dim=1)
        yk = torch.argmax(probs, dim=1).numpy()

        client_data[k] = {
            'x': torch.tensor(Xk),            # shape: [num_samples, D-1]
            'y': torch.tensor(yk, dtype=torch.long)  # shape: [num_samples]
        }

    return client_data


K = 100
D = 61
client_data = generate_synthetic_data(K , D , num_classes = 10, num_samples_per_client = 100, alpha = 1., beta = 1., iid = False, seed = 0)



def power_of_choice_policy(global_model, local_data, K, P, M):
    """Select P clients among M randomly sampled, based on highest local loss."""
    candidates = random.sample(range(K), M)
    model_copy = copy.deepcopy(global_model)
    model_copy.eval()
    losses = []

    with torch.no_grad():
        for c in candidates:
            loss = local_eval_loss(model_copy, local_data[c])
            losses.append((loss, c))

    # Select top-P clients with highest local loss
    losses.sort(reverse=True)  # descending order
    selected = [c for (_, c) in losses[:P]]
    return selected
def active_fl_selection(global_model, local_data, K, P, M, temperature=1.0):
    """
    Active Federated Learning client selection based on softmax over local loss.

    Args:
        global_model: current global model.
        local_data: dict of DataLoader objects per client.
        K: total number of clients.
        P: number of clients to select.
        M: number of clients to sample as candidates (M >= P).
        temperature: temperature parameter for softmax (default=1.0).
    
    Returns:
        selected_clients: list of P selected client indices.
    """
    assert M >= P, "Candidate pool size M must be at least as large as P."

    candidates = random.sample(range(K), M)
    model_copy = copy.deepcopy(global_model)
    model_copy.eval()
    losses = []

    with torch.no_grad():
        for c in candidates:
            loss = local_eval_loss(model_copy, local_data[c])  # average loss on client's local_data
            losses.append(loss)

    # Convert losses to probabilities via softmax
    loss_tensor = torch.tensor(losses)
    probs = F.softmax(loss_tensor / temperature, dim=0).numpy()

    # Sample P clients (without replacement) from M candidates using softmax probabilities
    selected_indices = np.random.choice(M, size=P, replace=False, p=probs)
    selected_clients = [candidates[i] for i in selected_indices]

    return selected_clients
def local_eval_loss(model, dataloader, criterion=nn.CrossEntropyLoss(), device="cpu"):
    """Evaluate the classification loss (cross-entropy) of the model on a client's local data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # raw outputs
            loss = criterion(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float("inf")
def uniform_policy(K,P):
    return random.sample(range(K), P)


def cluster_clients(theta_matrix: np.ndarray, P: int):
    """
    Cluster clients using spectral clustering on a correlation matrix.

    Args:
        correlation_matrix (np.ndarray): KxD matrix of client similarities.
        P (int): Number of clusters.

    Returns:
        dict: cluster_assignments[p] = [client indices in cluster p]
    """
    
    clustering = KMeans(n_clusters=P, random_state=0, n_init = 'auto')
    labels = clustering.fit_predict(theta_matrix)

    cluster_assignments = {p: [] for p in range(P)}
    for client_id, cluster_id in enumerate(labels):
        cluster_assignments[cluster_id].append(client_id)

    return cluster_assignments

def spectral_clustering_policy(theta_matrix: np.ndarray, P: int):
    """
    Perform spectral clustering on model parameters.

    Args:
        theta_matrix (np.ndarray): K x D matrix of client model parameters.
        P (int): Number of clusters.

    Returns:
        dict: cluster_assignments[p] = [client indices in cluster p], with spectral clustering.
    """
    clustering = SpectralClustering(n_clusters=P, random_state=0, affinity= 'rbf')
    labels = clustering.fit_predict(theta_matrix)

    cluster_assignments = {p: [] for p in range(P)}
    for client_id, cluster_id in enumerate(labels):
        cluster_assignments[cluster_id].append(client_id)

    return cluster_assignments
def local_train(model, dataloader, epochs, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)  # raw logits for classification
            loss = loss_fn(y_pred, y_batch)  # y_batch should be integer labels
            loss.backward()
            optimizer.step()
    
    return model.state_dict()


def uniform_policy(K,P):
    return random.sample(range(K), P)
# Federated training

def uniform_policy_clusters(cluster_assignments):
    """
    Uniformly sample one client per cluster.

    Args:
        cluster_assignments (dict): Dictionary where each key is a cluster ID
                                    and each value is a list of client IDs in that cluster.

    Returns:
        list: List of selected client IDs, one per cluster.
    """
    selected_clients = []
    for cluster_id, clients in cluster_assignments.items():
        if clients:
            selected = random.choice(clients)
            selected_clients.append(selected)
    return selected_clients

def min_variance_policy(covariance, alpha, cluster_assignments):
    """
    Select one client per cluster that maximizes a criterion based on the covariance matrix and alpha vector.

    Args:
        covariance (Tensor): (K x K) covariance matrix.
        alpha (Tensor): (K x 1) or (K,) vector.
        cluster_assignments (dict): Mapping from cluster index to list of client indices.

    Returns:
        list: Selected client ID per cluster.
    """
    if type(covariance) == list:

        D = len(covariance)
        value_vectors = []
        for d in range(D):
            covariance_d = covariance[d]
            K = covariance_d.shape[0]
        
            alpha = alpha.view(K, 1) if alpha.ndim == 1 else alpha  # Ensure alpha is column vector
            value_vectors.append((torch.matmul(covariance_d, alpha)**2 / torch.diag(covariance_d).view(K, 1)).squeeze())

            value_vector = torch.stack(value_vectors, dim=0).sum(dim=0)  # Sum across dimensions

    else :
    
        if type(covariance) == np.ndarray:
            covariance = torch.from_numpy(covariance)
            
            
        K = covariance.shape[0]
        
        alpha = alpha.view(K, 1) if alpha.ndim == 1 else alpha  # Ensure alpha is column vector
        value_vector = (torch.matmul(covariance, alpha)**2 / torch.diag(covariance).view(K, 1)).squeeze()

    selected_clients = []
    for cluster_id, clients in cluster_assignments.items():
        cluster_values = value_vector[clients]
        max_idx = torch.argmax(cluster_values).item()
        selected_clients.append(clients[max_idx])

    return selected_clients,value_vector
    
def min_variance_boltzmann_policy(covariance, alpha, cluster_assignments, temperature=1.0):
    """
    Select one client per cluster using a Boltzmann (softmax) policy based on the value_vector.

    Args:
        covariance (Tensor): (K x K) covariance matrix.
        alpha (Tensor): (K x 1) or (K,) vector.
        cluster_assignments (dict): Mapping from cluster index to list of client indices.
        temperature (float): Temperature parameter for softmax. Lower = more greedy.

    Returns:
        list: Selected client ID per cluster.
    """
    if type(covariance) == list:

        D = len(covariance)
        value_vectors = []
        for d in range(D):
            covariance_d = covariance[d]
            K = covariance_d.shape[0]
        
            alpha = alpha.view(K, 1) if alpha.ndim == 1 else alpha  # Ensure alpha is column vector
            value_vectors.append((torch.matmul(covariance_d, alpha)**2 / torch.diag(covariance_d).view(K, 1)).squeeze())

            value_vector = torch.stack(value_vectors, dim=0).sum(dim=0)  # Sum across dimensions

    else :
    
        if type(covariance) == np.ndarray:
            covariance = torch.from_numpy(covariance)
            
            
        K = covariance.shape[0]
        
        alpha = alpha.view(K, 1) if alpha.ndim == 1 else alpha  # Ensure alpha is column vector
        value_vector = (torch.matmul(covariance, alpha)**2 / torch.diag(covariance).view(K, 1)).squeeze()

    selected_clients = []
    for cluster_id, clients in cluster_assignments.items():
        cluster_values = value_vector[clients] / temperature
        probs = torch.softmax(cluster_values, dim=0)
        selected_idx = torch.multinomial(probs, num_samples=1).item()
        selected_clients.append(clients[selected_idx])

    return selected_clients
def evaluate_global_model(global_model, test_data):
    global_accuracy = []

    for client_id in range(K):
        data_loader = test_data[client_id]
        for x_batch, y_batch in data_loader:
            x = x_batch
            y = y_batch.long()
            global_model.eval()
            y_pred_global = global_model(x).detach()
            _, predicted_global = torch.max(y_pred_global, 1)
            accuracy_global = (predicted_global == y).float().mean().item()
            global_accuracy.append(accuracy_global)
            break 

    avg_global_accuracy = sum(global_accuracy) / K
    print(f"\n[Evaluation @ Round {t+1}] Average Global Model Accuracy: {avg_global_accuracy:.4f}")
    return avg_global_accuracy


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=60, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)  # output raw logits
    
input_size = D - 1
output_size = 10


def evaluate_global_loss(global_model, test_data):
    global_model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for client_id in range(K):
            data_loader = test_data[client_id]
            for x_batch, y_batch in data_loader:
                x = x_batch
                y = y_batch.long()
                logits = global_model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
                total_samples += len(y)

    avg_loss = total_loss / total_samples
    print(f"\n[Evaluation @ Round {t+1}] Average Global Model Loss: {avg_loss:.4f}")
    return avg_loss

T1 = 30  # Number of initial uniform rounds
T = 100  # Total number of rounds
P = 10    # Number of clients to sample after T1
S = 10
gamma = 1e-2
batch_size = 100
global_model = LogisticRegression(input_size, output_size)
personal_models = {k: copy.deepcopy(global_model) for k in range(K)}
local_models = {k: copy.deepcopy(global_model) for k in range(K)}

#alpha = torch.ones(K) / K  # Uniform distribution over clients
alpha_train = []
N_tot = 0

for k in range(K):
    num_samples = len(local_data[k].dataset)
    alpha_train.append(num_samples)
    N_tot += num_samples

alpha_train = np.array(alpha_train) / N_tot
alpha = torch.tensor(alpha_train).float()
# ----- FedAvg phase (T1 rounds) -----

for t in range(T1):
    print(f"--- FedAvg Round {t+1} ---")
    selected_clients = uniform_policy(K, P)
    local_weights = {}

    for client_id in selected_clients:
        local_model = copy.deepcopy(global_model)
        updated_weights = local_train(local_model, local_data[client_id], S)
        local_model.load_state_dict(updated_weights)

        # Update the state dict of the personal model
        personal_models[client_id].load_state_dict(updated_weights)
        local_models[client_id].load_state_dict(updated_weights)
        # Store updated weights
        local_weights[client_id] = updated_weights

    # Aggregate selected clients' weights
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(local_weights[c][key] for c in selected_clients) / P
    global_model.load_state_dict(global_dict)
    if (t + 1) % 5 == 0:
        acc = evaluate_global_model(global_model, test_data)
        #avg_loss = evaluate_global_loss(global_model, test_data)
        test_losses.append(acc)

print("Initial FedAvg phase complete.")

num_pars = sum(p.numel() for p in global_model.parameters())
D = 610
sampled_weights = random.sample(range(num_pars), D)  # Dimensionality of model vector
covariances = [torch.ones(K,K) for _ in range(D)]            # One KxK matrix per model parameter


# ----- Main clustering-based phase -----
for t in range(T1, T):
    print(f"--- Global Round {t + 1} ---")
    theta_matrix = []
    for k in range(K):
        params = torch.nn.utils.parameters_to_vector(local_models[k].parameters()).detach().numpy()
        theta_matrix.append(params)
    theta_matrix = np.stack(theta_matrix)
    # Normalize rows
    theta_norm = torch.norm(torch.from_numpy(theta_matrix), dim=1, keepdim=True) + 1e-8  # Avoid div by 0
    normalized_theta = theta_matrix / theta_norm

    # Compute cosine similarity matrix
    rho = normalized_theta @ normalized_theta.T  # Shape: (K, K)
    
    
    cluster_assignments = spectral_clustering_policy(normalized_theta, P)
    selected_clients = min_variance_boltzmann_policy(covariances, alpha, cluster_assignments, temperature=1.0)

    print(f"Selected clients: {selected_clients}")

    local_weights = []
    global_state_dict_before_training = global_model.state_dict()

    # Local updates for selected clients
    for client_id in selected_clients:
        local_model = copy.deepcopy(global_model)
        updated_weights = local_train(local_model, local_data[client_id], S)
        local_model.load_state_dict(updated_weights)
        local_weights.append(updated_weights)
        local_models[client_id] = copy.deepcopy(local_model)
   
    # Update personal models in each cluster using the representative
    for p in range(P):
        jp = selected_clients[p]
        rep_state_dict = local_models[jp].state_dict()

        for k in cluster_assignments[p]:
            rho_kjp = rho[k, jp].item()
            old_state = personal_models[k].state_dict()
            new_state = {
                key: rho_kjp * rep_state_dict[key] 
                for key in old_state
            }
            personal_models[k].load_state_dict(new_state)

    #gamma = 1/(t - T1 +2)
    for d in range(D):
        # Build theta_d: the vector of d-th components from representatives (length K, default zeros for non-updated clients)
        theta_d = torch.zeros(K)
        bar_theta_d = torch.zeros(K)

        for p in range(P):
            jp = selected_clients[p]
            #rep_params = torch.nn.utils.parameters_to_vector(local_models[jp].parameters()).detach()

            for k in cluster_assignments[p]:
                bar_params = torch.nn.utils.parameters_to_vector(personal_models[k].parameters()).detach()
                rep_params = torch.nn.utils.parameters_to_vector(local_models[k].parameters()).detach()
                theta_d[k] = rep_params[d]
                bar_theta_d[k] = bar_params[d]

        # Compute difference and update covariance
        diff = (theta_d - bar_theta_d).reshape(-1, 1)  # (K,1)
        delta = (diff @ diff.T)
        covariances[d] = (1 - gamma) * covariances[d] + gamma * delta
    
    # Update global model by averaging selected updates
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(w[key] for w in local_weights) / P
    global_model.load_state_dict(global_dict)


    if (t + 1) % 5 == 0:
        acc =  evaluate_global_model(global_model, test_data)
        #avg_loss = evaluate_global_loss(global_model, test_data)
        test_losses.append(acc)
  

print("Federated training complete.")
