def epsilon_decay(epsilon, decay_rate, epsilon_final):
    """
    decay the epsilon factor in greedy action selection with
    decay_rate until epsilon_final reached
    """
    if epsilon <= epsilon_final:
        return epsilon_final

    return epsilon * decay_rate
