import torch
import numpy as np
import random
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random
import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# generate the datasets from normal distributions
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed()
# number of mixture components J 
def sample_x_dataset(mixture_components , mu , sigma,K, n, assignments ):
    # mu is a list of means, sigma is a list of standard deviations
    # mixture_components is the number of mixture components
    # mu and sigma are lists of length mixture_components
    # generate the dataset
    X_data = []

    for k in range(K):
        X_data.append(torch.normal(mu[assignments[k]], sigma[assignments[k]], size= (n,1)))        
    
    X_data = torch.stack(X_data)
    return X_data

#J = 1 for IID
J  = 3
K = 100
N = 100
D = 2
assignments = np.random.choice(J, K)

assignments = torch.from_numpy(assignments)

from collections import defaultdict

exact_cluster_assignments = defaultdict(list)

for client_id, cluster_id in enumerate(assignments):
    exact_cluster_assignments[cluster_id].append(client_id)



mu = np.random.uniform(-1, 1, J)
sigma = [.5]*J

def sample_true_regression_coefficient(D, K, mean, std, n, assignments):
    theta_data = []

    for k in range(K):
        m = mean[assignments[k]]
        s = std[assignments[k]]
        cov_matrix = (s ** 2) * torch.eye(D)
        dist = torch.distributions.MultivariateNormal(m, covariance_matrix=cov_matrix)
        samples = dist.sample((n,))  # n samples of D-dimensional vector
        theta_data.append(samples)

    theta_data = torch.stack(theta_data)  # Shape: (K, n, D)
    return theta_data



mean_theta = 5 * torch.rand(J, D, ) - 5

sigma_theta = [.1] * J


x_data = sample_x_dataset(J, mu, sigma, K, N, assignments)

theta_samples = sample_true_regression_coefficient(D, K, mean_theta, sigma_theta, N, assignments)

bias = torch.ones(K, N, 1)

x_data = torch.cat([bias, x_data], dim=-1)

x_test = sample_x_dataset(J, mu, sigma, K, N, assignments)
x_test = torch.cat([bias, x_test], dim=-1)

y_data = torch.sum(x_data * theta_samples, dim=-1)
y_test = torch.sum(x_test * theta_samples, dim=-1)


##Baseline


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
    losses.sort(reverse=True)  
    selected = [c for (_, c) in losses[:P]]
    return selected
def local_eval_loss(model, dataloader, criterion=nn.MSELoss(), device="cpu"):
    """Evaluate the MSE loss of the model on a client's local data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
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
    #clustering = SpectralClustering(n_clusters=P, random_state=0, affinity= 'rbf')
    labels = clustering.fit_predict(theta_matrix)

    cluster_assignments = {p: [] for p in range(P)}
    for client_id, cluster_id in enumerate(labels):
        cluster_assignments[cluster_id].append(client_id)

    return cluster_assignments

def local_train(model, dataloader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
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
    eps = 1e-8
    candidates = random.sample(range(K), M)
    model_copy = copy.deepcopy(global_model)
    model_copy.eval()
    losses = []

    with torch.no_grad():
        for c in candidates:
            loss = local_eval_loss(model_copy, local_data[c])  # average loss on client's local_data
            losses.append(loss)

    loss_tensor = torch.tensor(losses)
    probs = F.softmax(loss_tensor / temperature, dim=0).numpy()
    probs = probs + eps
    probs = probs / probs.sum()
    selected_indices = np.random.choice(M, size=P, replace=False, p=probs)
    selected_clients = [candidates[i] for i in selected_indices]

    return selected_clients

T1 = 30  # Number of initial uniform rounds
T = 200  # Total number of rounds
P = 10     # Number of clients to sample after T1
S = 10
gamma = 0.01
batch_size = 100
global_model = LinearRegression(input_size)
personal_models = {k: copy.deepcopy(global_model) for k in range(K)}
local_models = {k: copy.deepcopy(global_model) for k in range(K)}

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
    M = 2 * P
    #selected_clients = power_of_choice_policy(global_model, local_data, K, P, M)
    #selected_clients = active_fl_selection(global_model, local_data, K, P, M, )
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
   

print("Initial FedAvg phase complete.")

num_pars = sum(p.numel() for p in global_model.parameters())
D = 2
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
    theta_norm = torch.norm(torch.from_numpy(theta_matrix), dim=1, keepdim=True) + 1e-8 
    normalized_theta = theta_matrix / theta_norm

    # Compute cosine similarity matrix
    rho = normalized_theta @ normalized_theta.T  # Shape: (K, K)
    
    
    cluster_assignments = cluster_clients(theta_matrix, P)


    #selected_clients = uniform_policy_clusters(cluster_assignments)
    
    #selected_clients, value_vect = min_variance_policy(covariances, alpha, cluster_assignments)
    #selected_clients = uniform_policy_clusters(cluster_assignments)
    selected_clients = min_variance_boltzmann_policy(covariances, alpha, cluster_assignments, temperature=1.0)

    print(f"Selected clients: {selected_clients}")
    # local_models = {}
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

    gamma = 1/(t - T1 +2)
    for d in range(D):
        
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


  

print("Federated training complete.")
