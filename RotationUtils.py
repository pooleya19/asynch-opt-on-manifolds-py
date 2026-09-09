import numpy as np

def cross(pv,qv):
    return np.reshape(np.cross(np.reshape(pv,(3)),np.reshape(qv,(3))),(3,1))

def rotate_by_axis_angle(v, axis, angle):
    v = np.reshape(v, (3,1))
    axis = np.reshape(axis, (3,1)) / np.linalg.norm(axis)
    return v*np.cos(angle) + cross(axis, v)*np.sin(angle) + axis*(axis.T @ v)*(1-np.cos(angle)) # Rodrigues' rotation formula

def quat_conj(q):
    q = np.reshape(q,(4,1))
    return np.vstack((q[0,0], -q[1:4]))

def quat_prod(p,q):
    p = np.reshape(p,(4,1))
    q = np.reshape(q,(4,1))
    return np.vstack((p.T @ quat_conj(q), p[0,0]*q[1:4] + q[0,0]*p[1:4] + cross(p[1:4],q[1:4])))

def rotate_vector_by_quat(v, q):
    v = np.reshape(v,(3,1))
    q = np.reshape(q,(4,1)) / np.linalg.norm(q)
    vq = np.vstack((0,v))
    vqp = quat_prod(quat_prod(q, vq), quat_conj(q))
    return vqp[1:4]

def quat_to_rot_mat(q):
    qw, qx, qy, qz = q[:,0]
    Rq = np.array([[1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                   [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                   [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]])
    return Rq

def axis_angle_to_quat(axis, angle):
    axis = np.reshape(axis,(3,1))/np.linalg.norm(axis)
    return np.vstack((np.cos(angle/2), np.sin(angle/2)*axis))

def axis_angle_to_rot_mat(axis, angle):
    return quat_to_rot_mat(axis_angle_to_quat(axis, angle))

def partial_rot_mat_partial_quat(q):
    qw, qx, qy, qz = np.reshape(q,(4))
    partial_Rq_partial_qw = np.array([[0, -2*qz, 2*qy],
                                      [2*qz, 0, -2*qx],
                                      [-2*qy, 2*qx, 0]])
    partial_Rq_partial_qx = np.array([[0, 2*qy, 2*qz],
                                      [2*qy, -4*qx, -2*qw],
                                      [2*qz, 2*qw, -4*qx]])
    partial_Rq_partial_qy = np.array([[-4*qy, 2*qx, 2*qw],
                                      [2*qx, 0, 2*qz],
                                      [-2*qw, 2*qz, -4*qy]])
    partial_Rq_partial_qz = np.array([[-4*qz, -2*qw, 2*qx],
                                      [2*qw, -4*qz, 2*qy],
                                      [2*qx, 2*qy, 0]])
    return [partial_Rq_partial_qw, partial_Rq_partial_qx, partial_Rq_partial_qy, partial_Rq_partial_qz]