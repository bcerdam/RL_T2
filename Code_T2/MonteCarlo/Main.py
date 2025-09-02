from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
import random
import numpy as np
import matplotlib.pyplot as plt
import time


def sample_action(action_value_dict, state, possible_actions, epsilon):
    '''
    Funcion para pregunta j, basicamente samplea una accion de manera e-greedy.
    '''
    if random.uniform(0, 1) < epsilon:
        return random.choice(possible_actions)
    if state not in action_value_dict:
        return random.choice(possible_actions)

    state_actions = action_value_dict[state]
    best_action = None
    max_value = -float('inf')
    for action, value in state_actions.items():
        if value > max_value:
            max_value = value
            best_action = action
    return best_action


def episode(game_environment, action_value_dict, epsilon):
    '''
        Funcion para pregunta j, basicamente genera un episodio y retorna SAR,SAR,.....
    '''
    memory = []
    new_episode_initial_state = game_environment.reset()
    terminal_state = False

    while terminal_state == False:
        action = sample_action(action_value_dict, new_episode_initial_state, game_environment.action_space, epsilon)
        resulting_state, reward, terminal_state = game_environment.step(action)
        memory.append((new_episode_initial_state, action, reward))
        new_episode_initial_state = resulting_state

    return memory



def update(memory, action_value_dict, action_state_counter_dict, gamma, constant_step_size):
    '''
        Funcion para pregunta j, recibe memoria de un episodio, y actualiza
    '''
    G = 0
    for t in reversed(range(len(memory))):
        state, action, reward = memory[t]
        G = gamma * G + reward

        if constant_step_size:
            step_size = 0.1
        else:
            if state not in action_state_counter_dict:
                action_state_counter_dict[state] = {}
            if action not in action_state_counter_dict[state]:
                action_state_counter_dict[state][action] = 0
            action_state_counter_dict[state][action] += 1
            step_size = 1 / action_state_counter_dict[state][action]

        if state not in action_value_dict:
            action_value_dict[state] = {}
        if action not in action_value_dict[state]:
            action_value_dict[state][action] = 0

        Q_s_a = action_value_dict[state][action]
        action_value_dict[state][action] = Q_s_a + step_size * (G - Q_s_a)


def simulation(game_environment, action_value_dict, n_episodes):
    '''
        Funcion para pregunta j, recibe memoria de un episodio, y actualiza
    '''
    rewards = []

    for episode in range(n_episodes):
        state = game_environment.reset()
        terminal_state = False
        episode_reward = 0

        while terminal_state == False:
            action = sample_action(action_value_dict, state, game_environment.action_space, 0)
            state, reward, terminal_state = game_environment.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)


def plot_policy_performance(results, title):
    '''
        Funcion para pregunta j, plottea resultados que provienen de la pipeline general (train())
    '''
    plt.figure(figsize=(12, 8))

    for i, run_results in enumerate(results):
        episodes, rewards = zip(*run_results)
        plt.plot(episodes, rewards, label=f"Run {i + 1}")

    plt.title(title, fontsize=16)
    plt.xlabel("Episodios", fontsize=12)
    plt.ylabel("Retorno Promedio", fontsize=12)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/brunocerdamardini/Desktop/RL_T2/resultados/pregunta_j_blackjack.jpeg', dpi=500)
    plt.show()


def train(game, game_environment, gamma, epsilon, n_episodes, n_check, n_test_simulations, n_runs, cte_step_size):
    '''
        Funcion para pregunta j, entrena agente segun numero de runs y episodios, luego plottea resultados.
    '''
    all_runs_results = []

    for run in range(n_runs):
        action_value_dict = {}
        action_state_counter_dict = {}
        run_results = []

        episode_memory = episode(game_environment, action_value_dict, epsilon)
        first_return = sum(reward for state, action, reward in episode_memory)
        print(f"Run {run+1}, Episode 1: Return = {first_return}")
        run_results.append((1, first_return))
        update(episode_memory, action_value_dict, action_state_counter_dict, gamma, cte_step_size)
        for i in range(2, n_episodes + 1):
            episode_memory = episode(game_environment, action_value_dict, epsilon)
            update(episode_memory, action_value_dict, action_state_counter_dict, gamma, cte_step_size)
            if i % n_check == 0:
                avg_reward = simulation(game_environment, action_value_dict, n_test_simulations)
                run_results.append((i, avg_reward))
                print(f"Run {run+1}, Episode {i}: Avg Return = {avg_reward:.4f}")
        all_runs_results.append(run_results)
    plot_policy_performance(all_runs_results, f"Resultados {game}: On-Policy every-visit MC Control")


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")

def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


if __name__ == '__main__':

    '''
    Pregunta j
    '''
    # GAME_BLACKJACK = "Blackjack"
    # GAME_ENVIRONMENT_BLACKJACK = BlackjackEnv()
    # GAMMA_BLACKJACK = 1
    # EPSILON_BLACKJACK = 0.01
    # N_EPISODES_BLACKJACK = 10**7
    # N_CHECK_BLACKJACK = 5*10**5
    # N_TEST_SIMULATIONS_BLACKJACK = 10**5
    # N_RUNS_BLACKJACK = 5
    # CTE_STEP_SIZE_BLACKJACK = False
    #
    # train(GAME_BLACKJACK, GAME_ENVIRONMENT_BLACKJACK, GAMMA_BLACKJACK, EPSILON_BLACKJACK, N_EPISODES_BLACKJACK,
    #       N_CHECK_BLACKJACK, N_TEST_SIMULATIONS_BLACKJACK, N_RUNS_BLACKJACK, CTE_STEP_SIZE_BLACKJACK)

    # GAME_CLIFF = "Cliff Walking"
    # GAME_ENVIRONMENT_CLIFF = CliffEnv(width=12)
    # GAMMA_CLIFF = 1
    # EPSILON_CLIFF = 0.1
    # N_EPISODES_CLIFF = 2*10**5
    # N_CHECK_CLIFF = 10**3
    # N_TEST_SIMULATIONS_CLIFF = 1
    # N_RUNS_CLIFF = 5
    # CTE_STEP_SIZE_CLIFF = False
    #
    # train(GAME_CLIFF, GAME_ENVIRONMENT_CLIFF, GAMMA_CLIFF, EPSILON_CLIFF, N_EPISODES_CLIFF,
    #       N_CHECK_CLIFF, N_TEST_SIMULATIONS_CLIFF, N_RUNS_CLIFF, CTE_STEP_SIZE_CLIFF)


    '''
    Pregunta j
    '''

