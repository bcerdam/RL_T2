import random
import time
import numpy as np
import matplotlib.pyplot as plt

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem


def plot_convergence(plotting_data, labels, title):
    plt.figure(figsize=(12, 8))

    for i, deltas in enumerate(plotting_data):
        plt.plot(deltas, label=labels[i])

    plt.title(title, fontsize=16)
    plt.xlabel("Iteration until convergence (While loop, delta < theta)", fontsize=12)
    plt.ylabel("Delta", fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    # plt.savefig('pregunta_e_cookie.jpeg', dpi=500)
    plt.show()


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def bellman(actions, problem, state, V_s, gamma):
    pi = 1 / len(actions)
    double_sum = 0

    for action in actions:
        action_value = 0
        transitions = problem.get_transitions(state, action)

        for transition in transitions:
            p, next_state, r = transition

            if next_state in V_s.keys():
                v_s_p = V_s[next_state]
            else:
                v_s_p = 0

            action_value += p * (r + gamma*v_s_p)

        double_sum += pi * action_value

    return double_sum


def play(problem, theta, gamma):
    done = False
    states = problem.states
    V_s = {}

    for state in states:
        if state not in V_s.keys():
            V_s[state] = 0

    deltas = []

    while not done:
        delta = 0

        for state in states:

            if problem.is_terminal(state):
                continue
            else:
                actions = problem.get_available_actions(state)
                v_value = V_s[state]
                V_s[state] = bellman(actions, problem, state, V_s, gamma)
                delta = max(delta, np.abs(v_value - V_s[state]))

        deltas.append(delta)
        if delta < theta:
            done = True

    return V_s, deltas


def play_gambler_problem(p, theta, gamma):
    problem = GamblerProblem(p)
    V_s, deltas = play(problem, theta, gamma)
    return V_s, deltas, problem


def play_grid_problem(size, theta, gamma):
    problem = GridProblem(size)
    V_s, deltas = play(problem, theta, gamma)
    return V_s, deltas, problem


def play_cookie_problem(size, theta, gamma):
    problem = CookieProblem(size)
    V_s, deltas = play(problem, theta, gamma)
    return V_s, deltas, problem


if __name__ == '__main__':

    '''
    Pregunta d)
    '''
    THETA = 0.0000000001

    # print("--- GridProblem START ---")
    #
    # GAMMA = 1
    # plotting_info = []
    # for size in range(3, 11):
    #     start_time = time.time()
    #     V_s, deltas, problem = play_grid_problem(size, THETA, GAMMA)
    #     duration = time.time() - start_time
    #
    #     plotting_info.append(deltas)
    #     initial_state = problem.get_initial_state()
    #     initial_state_value = V_s[initial_state]
    #     print(f"Tamaño grilla: {size}; Valor estado inicial: {initial_state_value:.3f}; Tiempo Ejecucion: {duration:.3f}s")
    #
    # plot_labels = [f"Tamaño grilla {size}x{size}" for size in range(3, 11)]
    # plot_convergence(plotting_info, plot_labels, "GridProblem: Iterative Policy Evaluation Convergence")
    #
    # print("--- GridProblem END ---")

    # print("--- CookieProblem START ---")
    #
    # GAMMA = 0.99
    # plotting_info = []
    # for size in range(3, 11):
    #     start_time = time.time()
    #     V_s, deltas, problem = play_cookie_problem(size, THETA, GAMMA)
    #     duration = time.time() - start_time
    #
    #     plotting_info.append(deltas)
    #     initial_state = problem.get_initial_state()
    #     initial_state_value = V_s[initial_state]
    #     print(
    #         f"Tamaño grilla: {size}; Valor estado inicial: {initial_state_value:.3f}; Tiempo Ejecucion: {duration:.3f}s")
    #
    # plot_labels = [f"Tamaño grilla {size}x{size}" for size in range(3, 11)]
    # plot_convergence(plotting_info, plot_labels, "CookieProblem: Iterative Policy Evaluation Convergence")
    #
    # print("--- CookieProblem END ---")

    # print("--- GamblerProblem START ---")
    #
    # GAMMA = 1
    # p_head = [0.25, 0.4, 0.55]
    # plotting_info = []
    # for p in p_head:
    #
    #
    #     start_time = time.time()
    #     V_s, deltas, problem = play_gambler_problem(p, THETA, GAMMA)
    #     duration = time.time() - start_time
    #
    #     plotting_info.append(deltas)
    #     initial_state = problem.get_initial_state()
    #     initial_value = V_s[initial_state]
    #
    #     print(f"Probabilidad cara = {p}; Valor estado inicial: {initial_value:.3f}; Tiempo Ejecucion: {duration:.3f}s")
    #
    # plot_labels = [f"p={p}" for p in p_head]
    # plot_convergence(plotting_info, plot_labels, "GamblerProblem: Iterative Policy Evaluation Convergence")
    #
    # print("--- GamblerProblem END ---")

    '''
    Pregunta d)
    '''
