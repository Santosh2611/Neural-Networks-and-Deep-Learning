import numpy as np
import numba

@numba.njit(fastmath=True)
def forward_backward_numba(initial_probs, transition_probs, observation_probs, observations, 
                            max_iterations=1000, tol=1e-6):
    """Runs the forward-backward algorithm to estimate a hidden Markov model from observations.

    Args:
        initial_probs (array): array of shape (num_states,) representing the initial probabilities of each state
        transition_probs (array): array of shape (num_states, num_states) representing the transition probabilities between states
        observation_probs (array): array of shape (num_states, num_observations) representing the observation probabilities for each state
        observations (array): array of shape (num_observations,) representing the observed sequence of events
        
        max_iterations (int): maximum number of iterations for the EM algorithm
        tol (float): tolerance for convergence criteria

    Returns:
        new_initial_probs (array): array of shape (num_states,) representing the estimated initial probabilities of each state
        new_transition_probs (array): array of shape (num_states, num_states) representing the estimated transition probabilities between states
        new_observation_probs (array): array of shape (num_states, num_observations) representing the estimated observation probabilities for each state
    """

    # Define variables to hold HMM model parameters
    num_states = initial_probs.shape[0]
    num_observations = observation_probs.shape[1]

    # Initialize message vectors
    alpha = np.zeros((num_states, len(observations)))
    beta = np.zeros((num_states, len(observations)))

    # Compute forward messages
    alpha[:, 0] = initial_probs * observation_probs[:, observations[0]]
    for t in range(1, len(observations)):
        alpha[:, t] = observation_probs[:, observations[t]] * np.dot(np.ascontiguousarray(transition_probs).T, alpha[:, t-1])

    # Compute backward messages
    beta[:, -1] = 1
    for t in range(len(observations)-2, -1, -1):
        beta[:, t] = np.dot(np.ascontiguousarray(transition_probs), observation_probs[:, observations[t+1]] * beta[:, t+1])

    # Compute state marginals and pairwise marginals
    gamma = alpha * beta / np.sum(alpha * beta, axis=0)
    xi = np.zeros((num_states, num_states, len(observations)-1))
    sum_alphabeta = np.sum(alpha * beta, axis=0)  # Cache common calculation
    for t in range(len(observations)-1):
        xi[:,:,t] = alpha[:,t][:,np.newaxis] * observation_probs[:,observations[t+1]] * beta[:,t+1] * transition_probs / sum_alphabeta[t+1]

    # Update parameters
    new_initial_probs = gamma[:, 0]
    new_transition_probs = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, np.newaxis]
    
    new_observation_probs = np.zeros((num_states, num_observations))
    for k in range(num_observations):
        match = (observations == k)
        new_observation_probs[:,k] = np.sum(gamma[:,match], axis=1) / np.sum(gamma, axis=1)

    # Check for convergence
    iters = 0
    while ((np.abs(new_initial_probs - initial_probs) > tol).any() or
           (np.abs(new_transition_probs - transition_probs) > tol).any() or
           (np.abs(new_observation_probs - observation_probs) > tol).any()) and (iters < max_iterations):
        # Update parameters
        initial_probs = new_initial_probs
        transition_probs = new_transition_probs
        observation_probs = new_observation_probs

        # Compute forward messages
        alpha[:, 0] = initial_probs * observation_probs[:, observations[0]]
        for t in range(1, len(observations)):
            alpha[:, t] = observation_probs[:, observations[t]] * np.dot(np.ascontiguousarray(transition_probs).T, alpha[:, t-1])

        # Compute backward messages
        beta[:, -1] = 1
        for t in range(len(observations)-2, -1, -1):
            beta[:, t] = np.dot(np.ascontiguousarray(transition_probs), observation_probs[:, observations[t+1]] * beta[:, t+1])

        # Compute state marginals and pairwise marginals
        gamma = alpha * beta / sum_alphabeta
        xi = np.zeros((num_states, num_states, len(observations)-1))
        for t in range(len(observations)-1):
            xi[:,:,t] = alpha[:,t][:,np.newaxis] * observation_probs[:,observations[t+1]] * beta[:,t+1] * transition_probs / sum_alphabeta[t+1]

        new_initial_probs = gamma[:, 0]
        new_transition_probs = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, np.newaxis]
        new_observation_probs = np.zeros((num_states, num_observations))
        for k in range(num_observations):
            match = (observations == k)
            new_observation_probs[:,k] = np.sum(gamma[:,match], axis=1) / np.sum(gamma, axis=1)

        iters += 1

    return new_initial_probs, new_transition_probs, new_observation_probs

# Define the HMM model parameters
initial_probs = np.array([0.5, 0.5])
transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
observation_probs = np.array([[0.9, 0.1], [0.2, 0.8]])

# Generate some data to fit the model
observations = np.random.choice([0, 1], size=100, p=[0.6, 0.4])

# Use the forward-backward algorithm to estimate the model parameters
new_initial_probs, new_transition_probs, new_observation_probs = forward_backward_numba(initial_probs, 
                                                                                         transition_probs, 
                                                                                         observation_probs, 
                                                                                         observations)

# Print the estimated model parameters
print("Estimated initial probabilities:", new_initial_probs)
print("Estimated transition probabilities:", new_transition_probs)
print("Estimated observation probabilities:", new_observation_probs)
