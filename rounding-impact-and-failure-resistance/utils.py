from random import random, seed, randint, choices

def argmax(lst):
    if not lst:
        raise ValueError("La liste ne doit pas être vide.")
    
    # search the best value that is not None
    max_value = None
    max_index = None
    for i in range(len(lst)):
        if lst[i] is not None:
            max_index = i
            max_value = lst[max_index]
            break
    
    if max_index is None: return max_index

    for i in range(max_index + 1, len(lst)):
        if lst[i] is not None and lst[i] > max_value:
            max_value = lst[i]
            max_index = i
    
    return max_index

class IsolatedRandomGenerator:
    """
    Random number generator able to produces random float and int, with respect to a given seed.
    """
    def __init__(self, seed):
        self.seed = seed

    def random(self, t) -> float:
        """Returns a random float between 0 and 1 includes given a time step."""
        seed(self.seed + t)
        value = random()

        return value

    def randint(self, t, min, max) -> int:
        """
        Returns a random int between min and max includes given a time step.
        """
        seed(self.seed + t)
        value = randint(min, max)
        return value

    def choice(self, items, weights, t):
        """Returns randomly a item contained in a given list of items, with a given probability weights."""
        seed(self.seed + t)
        value = choices( items, weights=weights, k=1 )[0]
        return value


class BernoulliArm:
    """Simulates a bandit arm which returns a reward with a given probability, with respect to a reward seed."""
    def __init__(self, p, seed):
        self.p = p
        self.random_generator = IsolatedRandomGenerator(seed=seed)

    def pull(self, t) -> int:
        """Returns a reward 0 or 1 randomly with respect to a time step t."""
        random_value = self.random_generator.random(t)
        reward = int(random_value < self.p)
        return reward

    def __str__(self):
        return str(self.p)
    
