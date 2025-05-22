import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

def download_cifar10(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize RGB
    ])    
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return dataset  

def dirichlet_partition(dataset, num_clients=10, alpha=0.5, seed=42):
    labels = np.array(dataset.targets)
    num_classes = 10
    client_data_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng(seed)

    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha]*num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split_indices = np.split(idxs, proportions)
        for client_id, client_idxs in enumerate(split_indices):
            client_data_indices[client_id].extend(client_idxs)

    return client_data_indices

def stratified_client_split(dataset, client_indices, test_size=0.2, seed=42):
    client_train_indices = {}
    client_test_indices = {}
    labels = np.array(dataset.targets)

    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            client_train_indices[client_id] = []
            client_test_indices[client_id] = []
            continue

        y = labels[indices]
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, stratify=y, random_state=seed
            )
        except ValueError:
            # Fall back to random split when stratification fails (e.g. all same class)
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=seed
            )

        client_train_indices[client_id] = train_idx
        client_test_indices[client_id] = test_idx

    return client_train_indices, client_test_indices

def create_client_loaders(dataset, client_indices, batch_size=32, shuffle=True):
    return {
        client_id: DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle)
        for client_id, indices in client_indices.items()
    }


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

# Fix seeds
SEED = 42
print(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def power_of_choice_policy(global_model, local_data, K, P, M):
    """Select P clients among M randomly sampled, based on highest local loss."""
    candidates = random.sample(range(K), M)
    model_copy = copy.deepcopy(global_model).to(device)
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

def local_eval_loss(model, dataloader, criterion=nn.CrossEntropyLoss()):
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

def cluster_clients_uniform(theta_matrix: np.ndarray, P: int):
    """
    Uniformly cluster clients using KMeans on model parameters.

    Args:
        theta_matrix (np.ndarray): K x D matrix of client model parameters.
        P (int): Number of clusters.

    Returns:
        dict: cluster_assignments[p] = [client indices in cluster p], with uniform sizes.
    """
    K = theta_matrix.shape[0]
    clustering = KMeans(n_clusters=P, random_state=0, n_init='auto')
    labels = clustering.fit_predict(theta_matrix)

    # Collect clients per cluster (may be unbalanced)
    preliminary_assignments = {p: [] for p in range(P)}
    for client_id, cluster_id in enumerate(labels):
        preliminary_assignments[cluster_id].append(client_id)

    # Flatten all client IDs and shuffle
    all_clients = [client for cluster in preliminary_assignments.values() for client in cluster]
    np.random.shuffle(all_clients)

    # Reassign uniformly
    cluster_assignments = {p: [] for p in range(P)}
    for i, client_id in enumerate(all_clients):
        cluster_assignments[i % P].append(client_id)

    return cluster_assignments

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


K = 100  # number of clients
alpha = 0.1
test_size = 0.2

full_dataset = download_cifar10()

client_indices = dirichlet_partition(full_dataset, num_clients=K, alpha=alpha)

client_train_indices, client_test_indices = stratified_client_split(full_dataset, client_indices, test_size=test_size)
train_loaders = create_client_loaders(full_dataset, client_train_indices, batch_size=32, shuffle=True)
test_loaders  = create_client_loaders(full_dataset, client_test_indices,  batch_size=128, shuffle=False)



from torch.utils.data import DataLoader, Subset

# local_data: maps client ID to their train DataLoader
local_data = {
    i: DataLoader(Subset(full_dataset, indices), batch_size=32, shuffle=True)
    for i, indices in client_train_indices.items()
}

# test_data: maps client ID to their test DataLoader
test_data = {
    i: DataLoader(Subset(full_dataset, indices), batch_size=128, shuffle=False)
    for i, indices in client_test_indices.items()
}

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                            # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                            # [B, 64, 8, 8]
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x



# Replace with your model


def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

# --- Local training ---
def local_train(model, dataloader, epochs, lr=0.01, device="cpu"):
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model.state_dict()


T1 = 30  # Number of initial FedAvg rounds
T = 200  # Total number of rounds
P = 10   # Number of clients to sample after T1
S = 10
gamma = 1e-2
batch_size = 100

global_model = SimpleCNN().to(device)
local_models = [copy.deepcopy(global_model).to(device) for _ in range(K)]
personal_models = {k: copy.deepcopy(global_model).to(device) for k in range(K)}

N_tot = 0
alpha_train = []
N_tot = 0

for indices in client_train_indices.values():
    alpha_train.append(len(indices))
    N_tot += len(indices)

alpha_train = np.array(alpha_train) / N_tot
alpha = torch.tensor(alpha_train, device=device).float()

D = 300

# ----- FedAvg phase (T1 rounds) -----
for t in range(T1):
    print(f"--- FedAvg Round {t+1} ---")
    selected_clients = uniform_policy(K, P)
    local_weights = {}

    for client_id in selected_clients:
        local_model = copy.deepcopy(global_model).to(device)
        updated_weights = local_train(local_model, local_data[client_id], S)
        local_model.load_state_dict(updated_weights)

        personal_models[client_id].load_state_dict(updated_weights)
        local_models[client_id].load_state_dict(updated_weights)
        local_weights[client_id] = updated_weights

    client_samples = {c: len(local_data[c].dataset) for c in selected_clients}
    total_samples = sum(client_samples.values())

    # Initialize global_dict
    global_dict = {key: torch.zeros_like(param).to(device) for key, param in global_model.state_dict().items()}

    # Weighted aggregation
    for key in global_dict:
        for c in selected_clients:
            local_param = local_weights[c][key].to(device)
            weight = client_samples[c] / total_samples
            global_dict[key] += weight * local_param

    # Load aggregated weights into global model
    global_model.load_state_dict(global_dict)

    if (t + 1) % 10 == 0:
        accuracies = []
        for k in range(K):
            acc = evaluate_model(global_model, test_data[k])
            accuracies.append(acc)
        avg_acc = sum(accuracies) / K
        print(f"[Eval] Round {t + 1}: Global model avg test accuracy = {avg_acc:.4f}")

print("Initial FedAvg phase complete.")
num_pars = sum(p.numel() for p in global_model.parameters())
sampled_weights = random.sample(range(num_pars - 1280, num_pars), D)

covariances = [torch.ones(K, K, device=device) for _ in range(D)]  # One KxK matrix per model parameter

# ----- Main clustering-based phase -----
for t in range(T1, T):
    print(f"--- Global Round {t + 1} ---")
    theta_matrix = []
    for k in range(K):
        params = torch.nn.utils.parameters_to_vector(local_models[k].parameters()).detach()[sampled_weights]
        theta_matrix.append(params)
    theta_matrix = torch.stack(theta_matrix).to(device)

    # Normalize rows
    theta_norm = torch.norm(theta_matrix, dim=1, keepdim=True) + 1e-8
    normalized_theta = theta_matrix / theta_norm

    rho = normalized_theta @ normalized_theta.T  # Cosine similarity

    cluster_assignments = spectral_clustering_policy(normalized_theta.cpu().numpy(), P)
    selected_clients = min_variance_boltzmann_policy(covariances, alpha, cluster_assignments, temperature=1.0)
    print(f"Selected clients: {selected_clients}")
    local_weights = []

    for client_id in selected_clients:
        local_model = copy.deepcopy(global_model).to(device)
        updated_weights = local_train(local_model, local_data[client_id], S)
        local_model.load_state_dict(updated_weights)
        local_weights.append(updated_weights)
        local_models[client_id] = copy.deepcopy(local_model).to(device)

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

    #gamma = 1 / (t - T1 + 2)

    for d in range(D):
        theta_d = torch.zeros(K, device=device)
        bar_theta_d = torch.zeros(K, device=device)

        for p in range(P):
            jp = selected_clients[p]

            for k in cluster_assignments[p]:
                bar_params = torch.nn.utils.parameters_to_vector(personal_models[k].parameters()).detach()[sampled_weights]
                rep_params = torch.nn.utils.parameters_to_vector(local_models[k].parameters()).detach()[sampled_weights]
                theta_d[k] = rep_params[d]
                bar_theta_d[k] = bar_params[d]

        diff = (theta_d - bar_theta_d).reshape(-1, 1)
        delta = diff @ diff.T
        covariances[d] = (1 - gamma) * covariances[d] + gamma * delta

    client_samples = [len(local_data[c].dataset) for c in selected_clients]
    total_samples = sum(client_samples)

    # Initialize global_dict as a copy of the current global model state
    global_dict = global_model.state_dict()

    # Weighted aggregation
    for key in global_dict:
        weighted_sum = sum(
            (len(local_data[c].dataset) / total_samples) * local_weights[i][key].to(device)
            for i, c in enumerate(selected_clients)
        )
        global_dict[key] = weighted_sum

    # Load updated global weights
    global_model.load_state_dict(global_dict)
    if (t + 1) % 10 == 0:
            accuracies = []
            for k in range(K):
                acc = evaluate_model(global_model, test_data[k])
                accuracies.append(acc)
            avg_acc = sum(accuracies) / K
            print(f"[Eval] Round {t + 1}: Global model avg test accuracy = {avg_acc:.4f}")
            
print("Federated training complete.")



