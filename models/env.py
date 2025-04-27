import pandas as pd
import numpy as np

class simulator:
    def __init__(self, dataset_path):
        
        # Normal Distribution, data within (0, 1)
        self.intelligence_level = max(0, min(1, np.random.normal(loc=0.4, scale=0.2)))

        # 5 types
        self.misconception_type = np.random.randint(low=1, high=6)

        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(self.dataset_path, sep="\t")
        self.dataset = self.dataset[self.dataset["misconception_type"] == self.misconception_type]

        # Conversation Length Distribution
        self.len_prob = np.array(self.dataset["convo_turn"].value_counts() / np.sum(self.dataset["done"] == 1))

        
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
        
        
    def step(self, action):
        # State
        # self.misconception_type

        # action affects probabilities
        factor = 0
        if action == 0:
            factor = 0.05
        elif action == 1:
            factor = 0.1
        elif action == 2:
            factor = 0.2
        else:
            factor = 0.01
            
        # one more response
        condition = np.random.choice([0,1], p=[1-min(1.0,self.len_prob[self.convo_turn]), min(1.0, self.len_prob[self.convo_turn])])
        if condition == 1:
            self.convo_turn = self.convo_turn + 1
            self.done = 0
        else:
            self.convo_turn = self.convo_turn + 1
            self.done = 1

        # feedback
        prob = min(1, self.intelligence_level + factor)
        self.listen_to_feedback = np.random.choice([0,1], p=[1-prob, prob])
        
        # progress
        prob = min(1, self.intelligence_level + factor)
        condition = np.random.choice([0,1], p=[1-prob, prob])
        if condition == 1:
            progress = np.random.choice([5,10], p=[1-self.intelligence_level, self.intelligence_level])
        else:
            progress = np.random.choice([-5,0], p=[1-self.intelligence_level, self.intelligence_level])

        old_problem_progress = self.problem_progress
        self.problem_progress = max(0, min(100, self.problem_progress + progress))
        
        self.progress_delta = self.problem_progress - old_problem_progress

        # correct solution
        prob = (min(1, self.intelligence_level + factor)) * (self.problem_progress / 100)
        self.correct_solution = np.random.choice([0,1], p=[1-prob, prob])

        if self.correct_solution == 1:
            self.done = 1
            self.problem_progress = 100
            self.progress_delta = 100 - old_problem_progress

        # Reward Function
        reward = self.correct_solution * 10 + self.done * 5 + self.listen_to_feedback * 2 + np.sign(self.progress_delta) * 1

        if self.done == 0:
            done_ = False
        else:
            done_ = True
        
        return (self.misconception_type, self.convo_turn, self.done, 
                int(self.listen_to_feedback),int(self.problem_progress),
                int(self.progress_delta), int(self.correct_solution)), int(reward), done_, "" , ""

    def reset(self):
        # Copy from __init__
        
        # Normal Distribution, data within (0, 1)
        self.intelligence_level = max(0, min(1, np.random.normal(loc=0.4, scale=0.2)))

        # 5 types
        self.misconception_type = np.random.randint(low=1, high=6)

        self.dataset = pd.read_csv(self.dataset_path, sep="\t")
        self.dataset = self.dataset[self.dataset["misconception_type"] == self.misconception_type]

        # Conversation Length Distribution
        self.len_prob = np.array(self.dataset["convo_turn"].value_counts() / np.sum(self.dataset["done"] == 1))

        
        self.convo_turn = 0
        self.done = 0
        self.listen_to_feedback = 0
        self.problem_progress = 0
        self.progress_delta = 0
        self.correct_solution = 0

        self.listen_to_feedback_count = 0
        return (self.misconception_type, self.convo_turn, self.done, 
                int(self.listen_to_feedback),int(self.problem_progress),
                int(self.progress_delta), int(self.correct_solution)), ""
    def render(self):
        pass
