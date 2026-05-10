import numpy as np
import open3d as o3d
import copy
import scipy.linalg

def vector_to_se3(v):
    """Convert 6D twist (w, rho) to SE(3) matrix."""
    w = v[:3]
    rho = v[3:]
    Omega = np.array([
        [0, -w[2], w[1], rho[0]],
        [w[2], 0, -w[0], rho[1]],
        [-w[1], w[0], 0, rho[2]],
        [0, 0, 0, 0]
    ])
    return np.real(scipy.linalg.expm(Omega))

def se3_to_vector(T):
    """Convert SE(3) to 6D twist vector (w, rho)."""
    log_T = scipy.linalg.logm(T)
    rho = np.array([log_T[0,3], log_T[1,3], log_T[2,3]])
    w = np.array([log_T[2,1], log_T[0,2], log_T[1,0]])
    return np.real(np.concatenate([w, rho]))

def adjoint(T):
    """Compute 6x6 Adjoint matrix of SE(3) for (w, rho) order."""
    R = T[:3, :3]
    t = T[:3, 3]
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    Adj = np.zeros((6, 6))
    Adj[:3, :3] = R           # w from w
    Adj[:3, 3:] = 0           # w from rho
    Adj[3:, :3] = tx @ R      # rho from w
    Adj[3:, 3:] = R           # rho from rho
    return Adj

class GBPNode:
    """A variable node in the factor graph representing a robot pose."""
    def __init__(self, node_id, initial_pose):
        self.node_id = node_id
        self.pose = initial_pose.copy() # 4x4 SE(3) matrix
        
        # Belief over dx (tangent space update)
        self.eta = np.zeros(6)
        self.Lambda = np.eye(6) * 1e-6
        
        # Inbox for incoming messages
        self.inbox_eta = {}
        self.inbox_Lambda = {}

    def aggregate_messages(self):
        """Sum incoming messages to compute the marginal belief."""
        eta_sum = np.zeros(6)
        Lambda_sum = np.eye(6) * 1e-6 # Weak prior to keep it invertible
        
        for f_id in self.inbox_eta:
            eta_sum += self.inbox_eta[f_id]
            Lambda_sum += self.inbox_Lambda[f_id]
            
        self.eta = eta_sum
        self.Lambda = Lambda_sum


class GBPFactor:
    """A factor node representing a relative pose measurement (edge)."""
    def __init__(self, factor_id, id_i, id_j, measurement_T, information_matrix):
        self.factor_id = factor_id
        self.id_i = id_i
        self.id_j = id_j
        self.Z = measurement_T          # 4x4 Measured relative pose
        self.Omega = information_matrix # 6x6 Uncertainty weight
        
        # Linearized components
        self.H_ii = np.zeros((6,6))
        self.H_jj = np.zeros((6,6))
        self.H_ij = np.zeros((6,6))
        self.H_ji = np.zeros((6,6))
        self.b_i = np.zeros(6)
        self.b_j = np.zeros(6)
        
        # Messages from this factor to connected nodes
        self.msg_to_i = (np.zeros(6), np.zeros((6, 6))) # (eta, Lambda)
        self.msg_to_j = (np.zeros(6), np.zeros((6, 6)))

    def linearize(self, node_i, node_j):
        """Outer loop: evaluate Jacobians and residuals at the current linearization point."""
        Xi = node_i.pose
        Xj = node_j.pose
        
        # Error: e = log(Z^-1 * Xi^-1 * Xj)
        error_matrix = np.linalg.inv(self.Z) @ np.linalg.inv(Xi) @ Xj
        e_vec = se3_to_vector(error_matrix)
        
        # Exact SE(3) Jacobians
        Ji = -adjoint(np.linalg.inv(Xj) @ Xi)
        Jj = np.eye(6)
        
        self.H_ii = Ji.T @ self.Omega @ Ji
        self.H_jj = Jj.T @ self.Omega @ Jj
        self.H_ij = Ji.T @ self.Omega @ Jj
        self.H_ji = self.H_ij.T
        
        self.b_i = -Ji.T @ self.Omega @ e_vec
        self.b_j = -Jj.T @ self.Omega @ e_vec
        
        # Reset messages for the new linear system
        self.msg_to_i = (np.zeros(6), np.zeros((6, 6)))
        self.msg_to_j = (np.zeros(6), np.zeros((6, 6)))

    def compute_messages(self, node_i, node_j):
        """Inner loop: Marginalize using the Schur complement for GBP."""
        # --- Message to Node I ---
        # 1. Get Node J's belief, discounting the message from this factor
        Lam_j_to_f = node_j.Lambda - self.msg_to_j[1]
        eta_j_to_f = node_j.eta - self.msg_to_j[0]
        
        # 2. Augment J's block with incoming belief
        H_jj_aug = self.H_jj + Lam_j_to_f
        b_j_aug = self.b_j + eta_j_to_f
        
        # 3. Marginalize J out
        try:
            # Slight damping for invertibility
            H_jj_inv = np.linalg.inv(H_jj_aug + np.eye(6)*1e-5) 
        except np.linalg.LinAlgError:
            H_jj_inv = np.zeros((6,6))
            
        Lam_f_to_i = self.H_ii - self.H_ij @ H_jj_inv @ self.H_ji
        eta_f_to_i = self.b_i - self.H_ij @ H_jj_inv @ b_j_aug
        
        # --- Message to Node J ---
        # 1. Get Node I's belief, discounting the message from this factor
        Lam_i_to_f = node_i.Lambda - self.msg_to_i[1]
        eta_i_to_f = node_i.eta - self.msg_to_i[0]
        
        # 2. Augment I's block with incoming belief
        H_ii_aug = self.H_ii + Lam_i_to_f
        b_i_aug = self.b_i + eta_i_to_f
        
        # 3. Marginalize I out
        try:
            H_ii_inv = np.linalg.inv(H_ii_aug + np.eye(6)*1e-5)
        except np.linalg.LinAlgError:
            H_ii_inv = np.zeros((6,6))
            
        Lam_f_to_j = self.H_jj - self.H_ji @ H_ii_inv @ self.H_ij
        eta_f_to_j = self.b_j - self.H_ji @ H_ii_inv @ b_i_aug
        
        # Save outgoing messages
        self.msg_to_i = (eta_f_to_i, Lam_f_to_i)
        self.msg_to_j = (eta_f_to_j, Lam_f_to_j)


def optimise_pose_graph_gbp(pose_graph, num_iterations=10):
    """
    Custom Gaussian Belief Propagation (GBP) solver for Open3D PoseGraph.
    Uses exact SE(3) logarithms/exponentials and Schur complement marginalization.
    """
    print(f"\n  [GBP] Initialising true Gaussian Belief Propagation with {num_iterations} iterations...")
    
    # 1. Initialize Nodes
    nodes = {}
    for i, node in enumerate(pose_graph.nodes):
        nodes[i] = GBPNode(node_id=i, initial_pose=node.pose)
        
    # 2. Initialize Factors (Edges)
    factors = []
    for f_idx, edge in enumerate(pose_graph.edges):
        info_mat = np.asarray(edge.information)
        factors.append(GBPFactor(
            factor_id=f_idx,
            id_i=edge.source_node_id,
            id_j=edge.target_node_id,
            measurement_T=edge.transformation,
            information_matrix=info_mat
        ))
        
    print(f"  [GBP] Built Factor Graph: {len(nodes)} Variables, {len(factors)} Factors.")

    # 3. Outer Relinearization Loop
    for outer_it in range(num_iterations):
        # Step A: Evaluate Jacobians at current linearization point
        for factor in factors:
            factor.linearize(nodes[factor.id_i], nodes[factor.id_j])
            
        # Step B: Inner GBP Message Passing Loop (Solve H dx = b)
        # Reset beliefs for the new tangent space
        for idx in nodes:
            nodes[idx].eta = np.zeros(6)
            nodes[idx].Lambda = np.eye(6) * 1e-6
            
        inner_iterations = 5 # Passing messages back and forth 5 times per linearisation
        for inner_it in range(inner_iterations):
            for factor in factors:
                factor.compute_messages(nodes[factor.id_i], nodes[factor.id_j])
                
                # Deliver messages
                nodes[factor.id_i].inbox_eta[factor.factor_id] = factor.msg_to_i[0]
                nodes[factor.id_i].inbox_Lambda[factor.factor_id] = factor.msg_to_i[1]
                
                nodes[factor.id_j].inbox_eta[factor.factor_id] = factor.msg_to_j[0]
                nodes[factor.id_j].inbox_Lambda[factor.factor_id] = factor.msg_to_j[1]
                
            # Nodes aggregate messages
            for idx in nodes:
                nodes[idx].aggregate_messages()
            
            # Apply strict anchor prior to Node 0 to fix global frame
            nodes[0].Lambda += np.eye(6) * 1e9
            
        # Step C: Update Poses
        total_dx = 0.0
        for idx in nodes:
            try:
                # Add Levenberg-Marquardt style damping for robust updates
                Lam_damped = nodes[idx].Lambda + np.eye(6) * 1e-3
                dx = np.linalg.solve(Lam_damped, nodes[idx].eta)
                
                # Sanity check: Clamp massive updates
                dx_norm = np.linalg.norm(dx)
                if dx_norm > 1.0: 
                    dx = (dx / dx_norm) * 1.0
                    dx_norm = 1.0
                    
                total_dx += dx_norm
                
                # Apply exponential map to update pose
                nodes[idx].pose = nodes[idx].pose @ vector_to_se3(dx)
            except np.linalg.LinAlgError:
                pass
                
        print(f"  [GBP] Iteration {outer_it+1}/{num_iterations} | Avg update: {total_dx/len(nodes):.5f}")

    # 4. Reconstruct optimized Open3D Pose Graph
    optimized_graph = copy.deepcopy(pose_graph)
    for i in range(len(optimized_graph.nodes)):
        optimized_graph.nodes[i].pose = nodes[i].pose
        
    return optimized_graph
