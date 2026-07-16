from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
THRESHOLD_FACTOR = 0.1
MAXIMA_COUNT = 6

def _robust_peak(values):
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if values.size <= MAXIMA_COUNT:
        return float(np.max(values))
    return float(np.mean(np.sort(values)[-MAXIMA_COUNT:]))

def set_threshold(data):
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        return np.array([_robust_peak(values) * THRESHOLD_FACTOR], dtype=float)

    if values.ndim == 2 and values.shape[0] == 3:
        series = values[1:3]
    else:
        series = values[:, 1:3].T

    thresholds = np.array(
        [_robust_peak(series[0]), _robust_peak(series[1])],
        dtype=float,
    )
    return thresholds * THRESHOLD_FACTOR

def calculate_n(data, threshold, total_pulses=20):
    # 1. Reconstruct expected max intensity (since threshold = max * factor)
    # We use 0.15 matching your global THRESHOLD_FACTOR
    max_a = max(THRESHOLD_FACTOR / 0.15, 1e-9) 
    max_b = max(THRESHOLD_FACTOR / 0.15, 1e-9)

    # 2. Use the low threshold to detect when ANY pulse is active
    active_any = (data[1] > THRESHOLD_FACTOR) | (data[2] > THRESHOLD_FACTOR)

    counts = np.zeros(4, dtype=float) # Uses floats for continuous addition
    in_event = False
    peak_a = 0.0
    peak_b = 0.0

    for a_val, b_val, active in zip(data[1], data[2], active_any):
        if active:
            in_event = True
            # Track the maximum analog intensity seen during this pulse window
            if a_val > peak_a: peak_a = a_val
            if b_val > peak_b: peak_b = b_val
            continue

        if in_event:
            # 3. Window closed: apply dark-level subtraction and normalize
            # This zeroes out ambient noise so it doesn't create fake HH coincidences
            clean_peak_a = max(peak_a - THRESHOLD_FACTOR, 0.0)
            clean_peak_b = max(peak_b - THRESHOLD_FACTOR, 0.0)
            
            clean_max_a = max(max_a - THRESHOLD_FACTOR, 1e-9)
            clean_max_b = max(max_b - THRESHOLD_FACTOR, 1e-9)

            n_a = min(clean_peak_a / clean_max_a, 1.0)
            n_b = min(clean_peak_b / clean_max_b, 1.0)
            
            # 4. Distribute the cleaned analog intensities cross-proportionally
            counts[0] += n_a * n_b           # HH
            counts[1] += n_a * (1.0 - n_b)   # HV
            counts[2] += (1.0 - n_a) * n_b   # VH
            
            in_event = False
            peak_a = 0.0
            peak_b = 0.0

    # Catch the final pulse if the array ended while still active
    if in_event:
        clean_peak_a = max(peak_a - THRESHOLD_FACTOR, 0.0)
        clean_peak_b = max(peak_b - THRESHOLD_FACTOR, 0.0)
        

        n_a = min(clean_peak_a, 1.0)
        n_b = min(clean_peak_b, 1.0)
        
        counts[0] += n_a * n_b
        counts[1] += n_a * (1.0 - n_b)
        counts[2] += (1.0 - n_a) * n_b

    # Infer the missing dark/vacuum pulses as VV (Index 3)
    active_sum = counts[0] + counts[1] + counts[2]
    if total_pulses > active_sum:
        counts[3] = total_pulses - active_sum
    else:
        counts[3] = 0.0

    # Return as floats so load_basis_probabilities can average the fractions
    return tuple(float(value) for value in counts)

def _record_event(has_a, has_b):
    if has_a and has_b:
        return (1, 0, 0, 0) # HH / DD
    if has_a:
        return (0, 1, 0, 0) # HV / DA
    if has_b:
        return (0, 0, 1, 0) # VH / AD
    return (0, 0, 0, 0)     # Decoupled baseline zero-fill

def load_basis_probabilities(intensities, basis_type='HV', bits=10):
    """
    Loads raw intensity data, extracts valid detection events using the thresholding mechanism,
    and returns normalized probabilities parsed contextually for the given measurement basis.
    """
    # Extract the expected pulse total from the filename (e.g., '10_intensities.csv' -> 10)
    if intensities.size == 0:
        return {}

    if intensities.ndim == 1 or intensities.shape[0] < 3:
        raise ValueError("Expected at least three columns: timestamp, detector A, detector B")

    thresholds = set_threshold(intensities)
    counts = calculate_n(intensities, thresholds, total_pulses=bits)
    total = float(sum(counts))
    if total <= 0:
        return {}

    normalized = [c / total for c in counts]
    
    if basis_type == 'HV':
        return {
            'HH': normalized[0],
            'HV': normalized[1],
            'VH': normalized[2],
            'VV': normalized[3]
        }
    elif basis_type == 'DA':
        return {
            'DD': normalized[0],
            'DA': normalized[1],
            'AD': normalized[2],
            'AA': normalized[3]
        }
    return {}

def reconstruct_density_matrix(hv_prob, da_prob):
    """
    Reconstructs the 4x4 density matrix \rho using distinct, decoupled measurement inputs
    from both the Rectilinear (H/V) and Diagonal (D/A) bases.
    """
    rho = np.zeros((4, 4), dtype=float)
    
    # 1. Populate the diagonal elements directly from the H/V basis probabilities
    rho[0, 0] = hv_prob.get('HH', 0.0)
    rho[1, 1] = hv_prob.get('HV', 0.0)
    rho[2, 2] = hv_prob.get('VH', 0.0)
    rho[3, 3] = hv_prob.get('VV', 0.0)
    
    # 2. Extract off-diagonal coherence terms using the D/A basis correlations
    if rho[0, 0] > 0.2 and rho[3, 3] > 0.2:
        # Aligned configuration (Bell state \psi_1)
        real_rho_14 = 2.0 * da_prob.get('DD', 0.0) - 0.5 * (rho[0, 0] + rho[3, 3])
        rho[0, 3] = real_rho_14
        rho[3, 0] = real_rho_14
        
    elif rho[1, 1] > 0.2 and rho[2, 2] > 0.2:
        # Complementary configuration (Bell state \psi_2)
        real_rho_23 = 0.5 * (rho[1, 1] + rho[2, 2]) - 2.0 * da_prob.get('DA', 0.0)
        rho[1, 2] = real_rho_23
        rho[2, 1] = real_rho_23

    return rho

def main():
    bits = [10,20,50]
    for b in bits:
        main2(b)

def main2(bits):
    
    data_dir = ROOT_DIR / "data" / "p1"

    if not data_dir.exists():
        print(f"Data directory '{data_dir}' not found. Please verify the file path.")
        return

    csv_file = data_dir / f"{bits}_intensities.csv"
    if not csv_file.exists():
        print(f"No CSV files found in '{data_dir}'.")
        return

    print("Quantum State Tomography at {} bits".format(bits))
    print("======================================================")    

    # Process coupled files
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1, unpack=True)
    data[3], data[4] = data[4], data[3] # Switching up columns since we classified them backwards in the CSV file
    for column in data[1:]:
        max=np.max(column)
        column /= max
    try:
        hv_probabilities = load_basis_probabilities(data[[0, 1, 2]], basis_type='HV', bits=bits)
        da_probabilities = load_basis_probabilities(data[[0, 3, 4]], basis_type='DA', bits=bits)
        
        rho = reconstruct_density_matrix(hv_probabilities, da_probabilities)
        print("Reconstructed Density Matrix (\\rho):")
        print(np.round(rho, decimals=4))
        
        sigma = np.array([[0, 0, 0, 0],
                          [0, 1/2, 1/2, 0],
                          [0, 1/2, 1/2, 0],
                          [0, 0, 0, 0]], dtype=float)
        fidelity = np.sqrt(np.trace(rho @ sigma))
        print(f"Fidelity with target Bell state: {fidelity:.4f}")
    except Exception as e:
        print(f"Error processing run at {bits} bits")
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()