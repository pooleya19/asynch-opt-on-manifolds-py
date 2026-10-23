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

def generateFormationLattice(N):
    height = 2
    ratio = 1 # length / width

    # Calculate length and width
    assert N%height == 0, "N must be divisible by height (N=%g, height=%g)" % (N, height)
    n = N//height
    lengths = np.array([i for i in range(1,n+1) if n % i == 0])
    ratio_errors = (ratio - np.power(lengths,2)/n)**2
    length = int(lengths[np.argmin(ratio_errors)])
    width = n//length
    assert height*length*width == N, "Failed to calculate height x length x width = N (height=%g, length=%g, width=%g, N=%g)" % (height, length, width, N)

    max_dim = max(height, max(length, width))
    gap = 1/(max_dim-1) if max_dim>1 else 1

    # Generate positions
    formation_positions = []
    for i in range(length):
        x = gap * (i - (length-1)/2)
        for j in range(width):
            y = gap * (j - (width-1)/2)
            for k in range(height):
                z = gap * (k - (height-1)/2)
                position = np.reshape([x,y,z], (3,1))
                formation_positions.append(position)

    # Generate edges
    max_edge_distance = gap * np.sqrt(3)
    formation_edges = []
    formation_offsets = []
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            ti = formation_positions[i]
            tj = formation_positions[j]
            if np.linalg.norm(ti-tj) <= max_edge_distance:
                formation_edges.append((i,j))
                formation_offsets.append(tj - ti)

    return formation_positions, formation_edges, formation_offsets