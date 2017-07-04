def epsilon_decay(epsilon, decay_rate, epsilon_final):
    """
    decay the epsilon factor in greedy action selection with
    decay_rate until epsilon_final reached
    """
    if epsilon <= epsilon_final:
        epsilon = epsilon_final
    else:
        epsilon *= decay_rate

    return epsilon
