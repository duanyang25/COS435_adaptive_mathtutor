import pandas as pd
import numpy as np
import gymnasium as gym

class simulator:
    def __init__(self, reward_func):
        
        # Normal Distribution, data within (0, 1)
        # self.intelligence_level = max(0, min(1, np.random.normal(loc=0.4, scale=0.2)))
        self.intelligence_level = np.random.choice([0.5,0.6],p=[0.6,0.4])

        # 5 types
        self.misconception_type = np.random.randint(low=1, high=6)

        # self.dataset_path = dataset_path
        # self.dataset = pd.read_csv(self.dataset_path, sep="\t")
        # self.dataset = self.dataset[self.dataset["misconception_type"] == self.misconception_type]

        # Conversation Length Distribution
        # self.len_prob = np.array(self.dataset["convo_turn"].value_counts() / np.sum(self.dataset["done"] == 1))

        
        self.convo_turn = 0
        self.done = 0
        self.listen_to_feedback = 0
        self.problem_progress = 0
        self.progress_delta = 0
        self.correct_solution = 0

        self.listen_to_feedback_count = 0

        # "Focus": 0,
        # "Probing": 1,
        # "Telling": 2,
        # "Generic": 3,
        # self.action = [0,1,2,3]

        self.observation_space = np.array([self.misconception_type, self.convo_turn, self.done, 
                int(self.listen_to_feedback),int(self.problem_progress),
                int(self.progress_delta), int(self.correct_solution)])
        self.action_space = np.array([0,1,2,3])

        self._max_episode_steps = 118

        self.prob_count = 0

        self.reward_func = reward_func
        
        
    def step(self, action):
        # State
        # self.misconception_type

        # action affects probabilities
        factor = 0
        if action == 0:
            factor = 0.6
        elif action == 1:
            factor = 0.8
        elif action == 2:
            factor = 0.9
        else:
            factor = 0.5
            
        # one more response
        # condition = np.random.choice([0,1], p=[1-min(1.0,self.len_prob[self.convo_turn]), min(1.0, self.len_prob[self.convo_turn])])
        # prob = min(1, self.intelligence_level + factor)
        # condition = np.random.choice([0,1], p=[1-factor, factor])

        condition = np.random.choice([0,1], p=[1-factor, factor])
        if action == 2:
            condition = np.random.choice([0,1], p=[factor, 1- factor])
        if condition == 1:
            self.convo_turn = self.convo_turn + 1
            self.done = 0
        else:
            self.convo_turn = self.convo_turn + 1
            self.done = 1
        
        # feedback
        # prob = min(1, self.intelligence_level + factor)
        self.listen_to_feedback = np.random.choice([0,1], p=[1-factor * 0.9 ** self.listen_to_feedback_count, factor * 0.9 ** self.listen_to_feedback_count])
        self.listen_to_feedback_count += self.listen_to_feedback 
        # self.listen_to_feedback = condition
        
        # progress
        # prob = min(1, self.intelligence_level + factor)
        # condition = np.random.choice([0,1], p=[1-prob, prob])
        if action == 0:
            progress = np.random.choice([-5,0,5], p=[0.2,1-factor-0.1, factor-0.1])
        elif action == 1:
            progress = np.random.choice([-5,5,10], p=[0.2,1-factor-0.1, factor-0.1])
        elif action == 2:
            progress = 25
        else:
            progress = np.random.choice([-5,0], p=[1-factor, factor])

        old_problem_progress = self.problem_progress
        self.problem_progress = max(0, min(100, self.problem_progress + progress))
        
        self.progress_delta = self.problem_progress - old_problem_progress

        # correct solution
        # prob = max(0,(min(1, self.intelligence_level + factor -0.5)) * (self.problem_progress / 100))
        if action == 0:
            self.correct_solution = np.random.choice([0,1], p=[1-factor, factor])
        elif action == 1:
            self.correct_solution = np.random.choice([0,1], p=[1-factor, factor])
        elif action == 2:
            self.correct_solution = np.random.choice([0,1], p=[1-factor, factor])
        else:
            # not change
            self.correct_solution = self.correct_solution    

        if self.problem_progress == 100:
            self.correct_solution = 1

        if self.correct_solution == 1:
            self.done = 1
            self.problem_progress = 100
            self.progress_delta = 100 - old_problem_progress

        # Reward Function, maybe consider the conversation length?: + (-0.1 * self.convo_turn)
        
        # "Focus": 0,
        # "Probing": 1,
        # "Telling": 2,
        # "Generic": 3,
        # self.action = [0,1,2,3]
        # action penalty
        penalty = 0
        # Focus
        if action == 0:
            penalty = 1
            penalty = penalty * (0.99 ** self.prob_count)
            # self.prob_count += 1
        # Probing
        elif action == 1:
            penalty = 2
            penalty = penalty * (0.99 ** self.prob_count)
            self.prob_count += 1
            
        # telling
        elif action == 2:
            penalty = -5
            penalty = penalty * (0.99 ** self.prob_count)
            self.prob_count += 2
        # Generic
        elif action == 3:
            penalty = 0
        
        # reward = self.correct_solution * 10 + self.done * 5 + self.listen_to_feedback * 2 + np.sign(self.progress_delta) * 1 + penalty
        # reward = self.correct_solution * 5 + penalty + (1 + self.done) # TD3 starts to learn something
        # reward = np.sign(self.progress_delta) * 5 + self.correct_solution * 200 + penalty + (10 + self.done) 
        # reward = self.correct_solution * 5 + penalty + (self.convo_turn + self.done)
        reward = self.reward_func(self, penalty, action)
        
        if self.done == 0:
            done_ = False
        else:
            done_ = True
        
        return [self.misconception_type, self.convo_turn, self.done, 
                int(self.listen_to_feedback),int(self.problem_progress),
                int(self.progress_delta), int(self.correct_solution)], int(reward), done_, "" , ""

    def reset(self):
        # Copy from __init__

        # should not change to a different student,
        # otherwise, unstable learning and cannot converge due to stochasticity
        
        # Normal Distribution, data within (0, 1)
        # self.intelligence_level = max(0, min(1, np.random.normal(loc=0.4, scale=0.2)))

        # 5 types
        self.misconception_type = np.random.randint(low=1, high=6)

        # self.dataset = pd.read_csv(self.dataset_path, sep="\t")
        # self.dataset = self.dataset[self.dataset["misconception_type"] == self.misconception_type]

        # # Conversation Length Distribution
        # self.len_prob = np.array(self.dataset["convo_turn"].value_counts() / np.sum(self.dataset["done"] == 1))

        
        self.convo_turn = 0
        self.done = 0
        self.listen_to_feedback = 0
        self.problem_progress = 0
        self.progress_delta = 0
        self.correct_solution = 0

        self.listen_to_feedback_count = 0

        self.prob_count = 0
        
        return [self.misconception_type, self.convo_turn, self.done, 
                int(self.listen_to_feedback),int(self.problem_progress),
                int(self.progress_delta), int(self.correct_solution)], ""
    def render(self):
        pass
