import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

class MNR:
    def __init__(self, C : int, D : int):
        self.C = C
        self.D = D
        self.W_tilde = np.random.rand(C, D+1)
        print(f"W_init is of shape (C, D+1) = {self.W_tilde.shape}")

    model_name = "MNR"
    
    def get_learning_rate(self, epoch : int, prev_learning_rate : float) -> float:
        pass
    
    def measure_ace(self, s : np.array, t : np.array) -> float:
        """Returns Average Cross-Entropy Loss

        Args:
            s (np.array): array that represents the softmax output of the model
            t (np.array): array that represents the true labels

        Returns:
            float: average cross-entropy loss
        """
        N = len(s)
        assert s.shape == (N, self.C), f"s.shape is {s.shape}"
        assert t.shape == (N, self.C), f"y.shape is {t.shape}"
        dots = []
        for i in range(N):
            dots.append(-1 * t[i].T @ np.log(s[i]))
        
        dots = np.array(dots)
        ones = np.ones((N, 1))

        ACE = (1/N) * dots @ ones

        return ACE
    def measure_zero_one_loss(s : np.array, t : np.array) -> float:
        N = len(s)
        dots = []
        for i in range(N):
            dots.append(int(np.argmax(t[i]) != np.argmax(s[i])))
        
        dots = np.array(dots)
        ones = np.ones((N, 1))

        avg_01_loss = (1/N) * dots @ ones
        return avg_01_loss
    
    def train(self, X_tilde : np.array, t : np.array, epochs : int, initial_lr : float, mod : int):
        W_tild_curr = self.W_tilde
        assert X_tilde.shape[0] == self.D + 1
        N = X_tilde.shape[1]

        ace_list : List[float] = []
        weights : List[np.array] = []

        for i in tqdm(range(epochs)):
            s = MNR.predict(X_tilde, W_tild_curr)
            assert t.shape == s.shape
            grad =  (1/N) * (s - t) @ X_tilde.T
            lr = self.get_learning_rate(i, initial_lr)
            W_tild_curr = W_tild_curr - lr * grad
            if i == 0 or i % mod == 0:
                ace_list.append(self.measure_ace(s.T, t.T))
                weights.append(W_tild_curr)
        
        return weights, ace_list

    def predict(X_tilde, W_tild_curr):
        h = W_tild_curr @ X_tilde
            # using the log sum exp trick
        h_max = np.max(h)
        h = h - h_max

        denom = np.sum(np.exp(h), axis=0)
            
        s = np.exp(h) / denom
        return s

class ConstLrMNR(MNR):

    model_name = "Constant LR MNR"

    def get_learning_rate(self, epoch : int, prev_learning_rate : float) -> float:
        return prev_learning_rate