import random
import numpy as np
import matplotlib.pyplot as plt

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem


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

    return deltas


def play_gambler_problem(theta, gamma):
    p = 0.4
    problem = GamblerProblem(p)
    return play(problem, theta, gamma)


def play_grid_problem(theta, gamma):
    size = 4
    problem = GridProblem(size)
    return play(problem, theta, gamma)


def play_cookie_problem(theta, gamma):
    size = 3
    problem = CookieProblem(size)
    return play(problem, theta, gamma)


if __name__ == '__main__':

    THETA = 0.0000000001
    GAMMA = 0.99

    # deltas = play_cookie_problem(THETA, GAMMA)
    # deltas = play_grid_problem(THETA, GAMMA)
    deltas = play_gambler_problem(THETA, GAMMA)

    plt.plot(deltas)
    plt.show()
