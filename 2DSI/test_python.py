from Experience import Experience
import numpy as np


experiences = []
exp = Experience([1, 2, 3], [2], [1, 2, 3], 3)
experiences.append(exp)

x_ = np.array([exp.previous_state for exp in experiences])
a_ = np.array([exp.action for exp in experiences])
r_ = np.array([exp.reward for exp in experiences])

print(x_)
print(a_)
print(r_)
