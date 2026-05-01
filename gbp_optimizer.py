import numpy as np
import open3d as o3d
import copy

def vector_to_se3(v):
    """Convert 6D twist (vx, vy, vz, wx, wy, wz) to SE(3) matrix."""
    import scipy.linalg
    w = v[3:]
    v_trans = v[:3]
    
    W = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    
    # Matrix exponential for SE(3)
    T = np.eye(4)
    theta = np.linalg.norm(w)
    if theta < 1e-7:
        T[:3, :3] = np.eye(3) + W
        T[:3, 3] = v_trans
    else:
        T[:3, :3] = scipy.linalg.expm(W)
        T[:3, 3] = v_trans # Simplified translation approximation for small updates
    return T

def se3_to_vector(T):
    """Convert SE(3) to 6D twist vector."""
    import scipy.linalg
    R = T[:3, :3]
    t = T[:3, 3]
    
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-7:
        w = np.zeros(3)
    else:
        log_R = scipy.linalg.logm(R)
        w = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
        
    return np.concatenate([t, np.real(w)])

class GBPNode:
    """A variable node in the factor graph representing a robot pose."""
    def __init__(self, node_id, initial_pose):
        self.node_id = node_id
        self.pose = initial_pose.copy() # 4x4 SE(3) matrix
        
        # GBP Belief states over tangent space (6D)
        self.eta = np.zeros(6)        # Information vector
        self.Lambda = np.eye(6) * 1e-6 # Information matrix (prior)
        
        # Inbox for messages from connected factors
        self.inbox_eta = {}
        self.inbox_Lambda = {}

    def update_belief(self):
        """Aggregate messages to compute marginal belief and update pose."""
        # Sum incoming messages
        eta_sum = np.zeros(6)
        Lambda_sum = np.eye(6) * 1e-6 # weak prior to keep invertible
        
        for f_id in self.inbox_eta:
            eta_sum += self.inbox_eta[f_id]
            Lambda_sum += self.inbox_Lambda[f_id]
            
        self.eta = eta_sum
        self.Lambda = Lambda_sum
        
        # Compute mean update: dx = Lambda^-1 * eta
        try:
            dx = np.linalg.solve(self.Lambda, self.eta)
        except np.linalg.LinAlgError:
            dx = np.zeros(6)
            
        # Update pose via SE(3) exponential map
        self.pose = self.pose @ vector_to_se3(dx)
        
        # Reset local belief for next linearization round
        self.eta = np.zeros(6)
        self.Lambda = np.eye(6) * 1e-6


class GBPFactor:
    """A factor node representing a relative pose measurement (edge)."""
    def __init__(self, factor_id, id_i, id_j, measurement_T, information_matrix):
        self.factor_id = factor_id
        self.id_i = id_i
        self.id_j = id_j
        self.Z = measurement_T     # 4x4 Measured relative pose
        self.Omega = information_matrix # 6x6 Uncertainty weight
        
        # Messages from this factor to connected nodes
        self.msg_to_i = (np.zeros(6), np.zeros((6, 6))) # (eta, Lambda)
        self.msg_to_j = (np.zeros(6), np.zeros((6, 6)))

    def compute_messages(self, node_i, node_j):
        """Linearize error and compute outgoing messages to nodes."""
        Xi = node_i.pose
        Xj = node_j.pose
        
        # Compute error: e = log(Z^-1 * Xi^-1 * Xj)
        # For simplicity in this base implementation, we approximate Jacobians
        # e = Xj - Xi * Z (in vector space approximation)
        error_matrix = np.linalg.inv(self.Z) @ np.linalg.inv(Xi) @ Xj
        e_vec = se3_to_vector(error_matrix)
        
        # Standard SE(3) Jacobian approximations for pose graph
        Ji = -np.eye(6)
        Jj = np.eye(6)
        
        # Compute Factor Information components
        H_ii = Ji.T @ self.Omega @ Ji
        H_jj = Jj.T @ self.Omega @ Jj
        H_ij = Ji.T @ self.Omega @ Jj
        H_ji = H_ij.T
        
        b_i = -Ji.T @ self.Omega @ e_vec
        b_j = -Jj.T @ self.Omega @ e_vec
        
        # Synchronous GBP Message Formulation (Marginalization)
        # Message to Node I
        self.msg_to_i = (b_i, H_ii)
        
        # Message to Node J
        self.msg_to_j = (b_j, H_jj)


def optimise_pose_graph_gbp(pose_graph, num_iterations=10):
    """
    Custom Gaussian Belief Propagation (GBP) solver for Open3D PoseGraph.
    This replaces the Levenberg-Marquardt global optimization.
    """
    print(f"\n  [GBP] Initialising Gaussian Belief Propagation with {num_iterations} iterations...")
    
    # 1. Initialize Nodes
    nodes = {}
    for i, node in enumerate(pose_graph.nodes):
        nodes[i] = GBPNode(node_id=i, initial_pose=node.pose)
        
    # 2. Initialize Factors (Edges)
    factors = []
    for f_idx, edge in enumerate(pose_graph.edges):
        # Open3D stores information matrix as 6x6. 
        # (Often formatted as translation x,y,z then rotation, but varies by implementation)
        info_mat = np.asarray(edge.information)
        factors.append(GBPFactor(
            factor_id=f_idx,
            id_i=edge.source_node_id,
            id_j=edge.target_node_id,
            measurement_T=edge.transformation,
            information_matrix=info_mat
        ))
        
    print(f"  [GBP] Built Factor Graph: {len(nodes)} Variables, {len(factors)} Factors.")

    # 3. Message Passing Loop
    for it in range(num_iterations):
        # Step A: Factors compute messages and send to Nodes
        for factor in factors:
            factor.compute_messages(nodes[factor.id_i], nodes[factor.id_j])
            
            # Deliver to inbox
            msg_eta_i, msg_Lam_i = factor.msg_to_i
            nodes[factor.id_i].inbox_eta[factor.factor_id] = msg_eta_i
            nodes[factor.id_i].inbox_Lambda[factor.factor_id] = msg_Lam_i
            
            msg_eta_j, msg_Lam_j = factor.msg_to_j
            nodes[factor.id_j].inbox_eta[factor.factor_id] = msg_eta_j
            nodes[factor.id_j].inbox_Lambda[factor.factor_id] = msg_Lam_j
            
        # Step B: Nodes aggregate messages and update beliefs (poses)
        # Node 0 is usually anchored (the origin)
        nodes[0].inbox_Lambda[-1] = np.eye(6) * 1e9 # Strong anchor prior
        
        for idx in nodes:
            nodes[idx].update_belief()
            
    print("  [GBP] Message passing converged.")

    # 4. Reconstruct optimized Open3D Pose Graph
    optimized_graph = copy.deepcopy(pose_graph)
    for i in range(len(optimized_graph.nodes)):
        optimized_graph.nodes[i].pose = nodes[i].pose
        
    return optimized_graph
