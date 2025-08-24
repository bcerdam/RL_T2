import random
import numpy as np

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

'''
CODIGO ORIGINAL

def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")
'''

def bellman(actions, problem, state, V_s, gamma):
    pi = 1 / len(actions)
    transitions = [problem.get_transitions(state, action) for action in actions] # [(p, (x+1, y+1), r), ...] # Problema esta en "_get_outcomes_when_agent_reaches_the_cookie", si llega a la galleta pone una lista nueva.
    print(transitions)

    double_sum = 0
    for transition in transitions:
        if transition[0][1] not in V_s.keys():
            double_sum += transition[0][0] * (transition[0][-1] + gamma*0)
        else:
            double_sum += transition[0][0] * (transition[0][-1] + gamma*V_s[transition[0][1]])

    return pi*len(actions)*double_sum

def play(problem, theta, gamma):
    # state = problem.get_initial_state()
    done = False
    total_reward = 0

    states = problem.states
    # V_s = [0.0 for i in states]
    V_s = {}
    for state in states:
        if state not in V_s.keys():
            V_s[state] = 0

    print(V_s)

    while not done:
        delta = 0
        for idx, state in enumerate(states):
            problem.show(state)

            actions = problem.get_available_actions(state)

            v_value = V_s[state]
            V_s[state] = bellman(actions, problem, state, V_s, gamma)
            delta = max(delta, np.abs(v_value - V_s[state]))



            # s_next, reward = sample_transition(transitions)
            # done = problem.is_terminal(s_next)
            # state = s_next
            # total_reward += reward
        if delta < theta:
            done = True

    # print("Done.")
    # print(f"Total reward: {total_reward}")



def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(size)
    play(problem)


'''
CODIGO ORIGINAL

def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)
'''

def play_cookie_problem(theta, gamma):
    size = 3
    problem = CookieProblem(size)
    play(problem, theta, gamma)


if __name__ == '__main__':
    '''
    CODIGO ORIGINAL
    
    # play_grid_problem()
    play_cookie_problem()
    # play_gambler_problem()
    '''

    THETA = 0.0000000001
    GAMMA = 0.99
    play_cookie_problem(THETA, GAMMA)
