import numpy as np
import random
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
plt.style.use(['science','grid'])


def new_particle():

    # Vector basis: [HH, HV, VH, VV]
    return np.array([0.0, 1/np.sqrt(2), 1/np.sqrt(2), 0.0])

def apply_joint_analyzers(joint_state, alpha_deg, beta_deg):
    """Projects the joint state using tensor products (Kronecker)."""
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    
    # Local hardware analyzer bases (Transmitted and Blocked)
    u_A_trans = np.array([np.cos(alpha), np.sin(alpha)])
    u_A_block = np.array([-np.sin(alpha), np.cos(alpha)])
    
    u_B_trans = np.array([np.cos(beta), np.sin(beta)])
    u_B_block = np.array([-np.sin(beta), np.cos(beta)])
    
    # 4 Possible Joint Outcomes (Probability = Amplitude^2)
    # np.kron creates the joint measurement basis for the two detectors
    p_HH = np.dot(joint_state, np.kron(u_A_trans, u_B_trans))**2
    p_HV = np.dot(joint_state, np.kron(u_A_trans, u_B_block))**2
    p_VH = np.dot(joint_state, np.kron(u_A_block, u_B_trans))**2
    p_VV = np.dot(joint_state, np.kron(u_A_block, u_B_block))**2
    
    return [p_HH, p_HV, p_VH, p_VV]

def add_noise(probs, noise_std=0.05):
    """Adds Gaussian noise directly to the theoretical probabilities."""
    return [max(0.0, p + random.gauss(0, noise_std)) for p in probs]
    

def collapse_quantum(noisy_probs):
    """Quantum measurement collapse into a single discrete joint event."""
    # Normalize probabilities to ensure they sum to exactly 1.0 after noise
    total = sum(noisy_probs)
    norm = [p / total for p in noisy_probs]
    
    roll = random.random()
    if roll < norm[0]:
        return (1, 0, 0, 0)  # Both Transmitted (HH / DD)
    elif roll < norm[0] + norm[1]:
        return (0, 1, 0, 0)  # Alice Transmitted, Bob Blocked (HV / DA)
    elif roll < sum(norm[:3]):
        return (0, 0, 1, 0)  # Alice Blocked, Bob Transmitted (VH / AD)
    else:
        return (0, 0, 0, 1)  # Both Blocked (VV / AA)

def simulate_single_pulse(alpha_deg, beta_deg, noise_std=0.02):
    """End-to-end quantum pulse simulation."""
    state = new_particle()
    # The beam splitter is no longer needed; entanglement spans the spatial gap
    probs = apply_joint_analyzers(state, alpha_deg, beta_deg)
    noisy_probs = add_noise(probs, noise_std=noise_std)
    event = collapse_quantum(noisy_probs)
    return event

def run_simulation(N, alpha_deg, beta_deg):
    """Runs N pulses and returns normalized coincidence probabilities."""
    counts = np.array([0, 0, 0, 0], dtype=float)
    for _ in range(N):
        event = simulate_single_pulse(alpha_deg, beta_deg)
        counts += np.array(event)
    return counts / N


def reconstruct_density_matrix(hv_prob, da_prob):
    """
    Reconstructs the 4x4 density matrix.
    Targets the INNER corners for an Anti-Correlated (|HV> / |VH>) state.
    """
    rho = np.zeros((4, 4), dtype=float)

    rho[0, 0] = hv_prob[0]  # P(HH)
    rho[1, 1] = hv_prob[1]  # P(HV)
    rho[2, 2] = hv_prob[2]  # P(VH)
    rho[3, 3] = hv_prob[3]  # P(VV)

    P_DD = da_prob[0]
    
    # Inner-corner coherence equation
    coherence = 2.0 * P_DD - 0.5 * (rho[1, 1] + rho[2, 2])
    
    rho[1, 2] = coherence
    rho[2, 1] = coherence

    return rho

def make_physical(rho):
    evals, evecs = np.linalg.eigh(rho)
    evals = np.maximum(evals, 0)
    trace = np.sum(evals)
    if trace > 0:
        evals = evals / trace
    return evecs @ np.diag(evals) @ evecs.conj().T

def main():
    results = []
    for bits in range(1, 101):

        # 1. Measure in Rectilinear Basis (H/V) 
        hv_prob = run_simulation(bits, alpha_deg=0, beta_deg=0)

        # 2. Measure in Diagonal Basis (D/A) 
        da_prob = run_simulation(bits, alpha_deg=45, beta_deg=45)

        # 3. Reconstruct Density Matrix
        rho = reconstruct_density_matrix(hv_prob, da_prob)
        rho = make_physical(rho)
        
        sigma = [[0,0,0,0],[0,1/2,1/2,0],[0,1/2,1/2,0],[0,0,0,0]]
        #fidelity = np.sqrt(np.real(np.trace(rho @ sigma)))
        #print(np.round(rho, 4))
        fidelity = np.sqrt(np.trace(rho @ sigma))
        results.append([bits, fidelity])
    
    bits = 1000
        # 1. Measure in Rectilinear Basis (H/V) 
    hv_prob = run_simulation(bits, alpha_deg=0, beta_deg=0)
    # 2. Measure in Diagonal Basis (D/A) 
    da_prob = run_simulation(bits, alpha_deg=45, beta_deg=45)
    # 3. Reconstruct Density Matrix
    rho = reconstruct_density_matrix(hv_prob, da_prob)
    rho = make_physical(rho)
    print(rho, np.sqrt(np.trace(rho @ sigma)))
    
    #print(np.round(rho, 4))
    

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.plot([r[0] for r in results], [r[1] for r in results], 'o', color="red")
    plt.xlabel('Number of Pulses')
    plt.ylabel('Fidelity')
    plt.title('Quantum State Fidelity vs. Number of Pulses')
    plt.grid(True)
    #plt.savefig(ROOT_DIR / "data" / "sim.png", dpi=300)

if __name__ == "__main__":
    main()