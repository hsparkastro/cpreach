import pickle

import matplotlib.pyplot as plt

# Load the pickled file
with open('examples/vanderpol/data/randomtraj_p0.pkl.train', 'rb') as f:
    data = pickle.load(f)

# Extract the trajectory array
trajectories = data[0]
points = trajectories.reshape(-1, 2)
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], c='r', s=2, alpha=0.01)

# # Plot the trajectories in the x-y plane
# for traj in trajectories:
#     plt.plot(traj[:, 0], traj[:, 1], 'k', alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectories in the x-y plane')
plt.axis('equal')
plt.show()
