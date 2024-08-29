# Reinforcement-Learning with Soft Actor-Critic (SAC)

This project explores the use of the Soft Actor-Critic (SAC) reinforcement learning algorithm in the Bipedal Walker v3 environment. SAC is an off-policy actor-critic method for continuous action spaces, including entropy regularization for balancing exploration and exploitation. The goal is to enable the agent to learn stable walking behavior through continuous learning.

## Project Overview

### Methodology

#### 1. Soft Actor-Critic Overview

SAC aims to maximize a trade-off between expected return and entropy, leading to a more exploratory policy. The objective function is formulated as:

\[
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
\]

Where:
- \( r(s_t, a_t) \) is the reward received after taking action \( a_t \) in state \( s_t \).
- \( H(\pi(\cdot|s_t)) = -\mathbb{E}_{a_t \sim \pi(\cdot|s_t)} \left[ \log \pi(a_t|s_t) \right] \) is the entropy of the policy.
- \( \alpha \) is the temperature parameter that determines the trade-off between exploration (entropy) and exploitation (reward).

#### 2. Policy and Value Functions

SAC employs three neural networks: the actor (policy) network and two critic (Q-value) networks.

- **Actor Network**: The policy \( \pi(a|s) \) is parameterized by a neural network and outputs a mean and standard deviation for a Gaussian distribution over actions. The actions are then sampled from this distribution:

\[
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
\]

- **Critic Networks**: There are two Q-value networks \( Q_{\phi_1}(s, a) \) and \( Q_{\phi_2}(s, a) \), which are used to mitigate overestimation bias.
  
\[
Q_{\phi_1}(s, a), Q_{\phi_2}(s, a)
\]

- **Target Networks**: Target networks \( Q_{\phi'_1}(s, a) \) and \( Q_{\phi'_2}(s, a) \) are used for stable training, updated using a soft update mechanism.

#### 3. Loss Functions

- **Critic Loss**: The critic networks are optimized by minimizing the mean squared error between the predicted Q-values and a target value:

\[
L(\phi_i) = \mathbb{E}_{(s,a,r,s')} \left[ (Q_{\phi_i}(s, a) - y)^2 \right]
\]

Where the target value \( y \) is computed as:

\[
y = r + \gamma(1 - d) \left( \min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a'|s') \right)
\]

- **Actor Loss**: The policy is updated by maximizing the expected Q-value, with an entropy term added to encourage exploration:

\[
L(\theta) = \mathbb{E}_{s \sim D} \left[ \alpha \log \pi_\theta(a|s) - Q_{\phi_1}(s, a) \right]
\]

#### 4. Soft Updates

The target networks are updated via a soft update mechanism:

\[
\phi'_i \leftarrow \tau \phi_i + (1 - \tau) \phi'_i
\]

Where \( \tau \) is the soft update parameter.

### Implementation Details

- **Framework**: PyTorch is used for constructing the neural networks and optimizing them. The actor and critic networks are multi-layer perceptrons with ReLU activations and dropout layers to improve generalization.
- **Replay Buffer**: A replay buffer is used to store transitions and sample mini-batches for training.
- **Hyperparameters**:
  - Learning rate: \( 3 \times 10^{-4} \)
  - Discount factor \( \gamma \): 0.99
  - Soft update parameter \( \tau \): 0.01
  - Replay buffer capacity: 10,000
  - Batch size: 64
  - Maximum episodes: 2000
  - Maximum timesteps per episode: 2000

### Convergence Results

The SAC agent's performance is measured over 1000 episodes, with convergence behaviors analyzed. The results show that the agent initially learns to improve performance with progress seen around 200 episodes. However, performance plateaus around 400 episodes, indicating potential challenges in the agent’s learning process.

### Limitations

- **Exploration vs. Exploitation Trade-off**: The balance between exploration and exploitation might not be optimal, leading to inconsistent performance.
- **Network Architecture and Hyperparameters**: The architecture and hyperparameters may not be ideal, contributing to instability in learning.
- **Reward Shaping**: The reward function may focus on immediate rewards rather than long-term efficiency.

### Future Work

1. **Hyperparameter Tuning**: Conduct a detailed hyperparameter search to find optimal values.
2. **Adaptive Entropy Adjustment**: Implement an adaptive mechanism for the temperature parameter \( \alpha \).
3. **Network Architecture Optimization**: Experiment with different network architectures, including more advanced options like recurrent neural networks (RNNs).
4. **Reward Shaping**: Modify the reward structure to incentivize more efficient walking patterns.
5. **Longer Training Duration**: Extend the training duration beyond 1000 episodes.
6. **Experience Replay Enhancements**: Implement prioritized experience replay for more stable learning.

## Running the Model

To run the reinforcement learning model, follow these steps:

1. **Install Required Libraries**:
   ```bash
   pip install torch torchvision
