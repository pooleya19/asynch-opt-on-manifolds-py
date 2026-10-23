import numpy as np

def generateFormationHelix(N):
    helix_rotations = 3

    formation_positions = []
    for i in range(N):
        t = i/(N-1)
        position = np.reshape([0.5*np.cos(helix_rotations*2*np.pi*t),
                               0.5*np.sin(helix_rotations*2*np.pi*t),
                               t - 0.5], (3,1))
        formation_positions.append(position)

    formation_edges = []
    formation_offsets = []
    for i in range(N-1):
        for j in range(i+1,min(N,i+3)):
            formation_edges.append((i,j))
            formation_offsets.append(formation_positions[j] - formation_positions[i])
            formation_edges.append((j,i))
            formation_offsets.append(formation_positions[i] - formation_positions[j])

    return formation_positions, formation_edges, formation_offsets