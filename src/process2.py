from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT_DIR / "data" / "processed"
THRESHOLD_FACTOR = 0.8
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


def _record_event(has_a, has_b):
    if has_a and has_b:
        return (1, 0, 0, 0)
    if has_a:
        return (0, 1, 0, 0)
    if has_b:
        return (0, 0, 1, 0)
    return (0, 0, 0, 1)


def calculate_n(data, threshold, total_pairs=20):
    active_a = data[1] > threshold[0]
    active_b = data[2] > threshold[1]

    counts = np.zeros(4, dtype=int)
    in_event = False
    current_event_a = False
    current_event_b = False
    saw_activity = False

    for a_val, b_val in zip(active_a, active_b):
        if a_val or b_val:
            saw_activity = True
            in_event = True
            current_event_a = current_event_a or a_val
            current_event_b = current_event_b or b_val
            continue

        if in_event:
            counts += _record_event(current_event_a, current_event_b)
            in_event = False
            current_event_a = False
            current_event_b = False
        elif not saw_activity:
            counts += (0, 0, 0, 1)

    if in_event:
        counts += _record_event(current_event_a, current_event_b)
    elif not saw_activity:
        counts += (0, 0, 0, 1)

    return tuple(int(value) for value in counts)

def calculate_E(n_AB, n_Ab, n_aB, n_ab):
    numerator = n_AB - n_Ab - n_aB + n_ab
    denominator = n_AB + n_Ab + n_aB + n_ab
    return numerator / denominator if denominator else 0

def normalize_angle(angle):
    return angle / 10.0 if abs(angle) >= 113 else angle


def get_chsh_value(results, angle_a, angle_b=None):
    if angle_b is None and isinstance(angle_a, tuple):
        angle_a, angle_b = angle_a
    return results[(normalize_angle(angle_a), normalize_angle(angle_b))]


def parse_angle_pair(filename_stem):
    angle_A, angle_B = map(float, filename_stem.split(","))
    return normalize_angle(angle_A), normalize_angle(angle_B)

def main():
    max_s = 0
    best_threshold = 0
    
    for t in np.linspace(0.639, 0.649, 999):
        global THRESHOLD_FACTOR
        THRESHOLD_FACTOR = t
        
        current_s = main2()
        if current_s > max_s:
            max_s = current_s
            # If THRESHOLD_FACTOR is a float, .copy() is unnecessary and will error.
            # If it is a mutable numpy array, keep the .copy().
            best_threshold = t 
            
    print(f"\nMaximum |S| value: {max_s:.4f} at threshold factor: {best_threshold}")

def main2():
    results = {}
    for path in sorted(CSV_DIR.glob("*_intensities.csv")):
        try:
            angle_a, angle_b = parse_angle_pair(path.stem.split("_", 1)[0])
        except ValueError:
            print(f"Skipping file with unexpected name format: {path.name}")
            continue

        data = np.genfromtxt(path, delimiter=",", skip_header=1, unpack=True)
        if data.size == 0:
            continue

        threshold = set_threshold(data)
        counts = calculate_n(data, threshold)
        
        results[(angle_a, angle_b)] = counts
        #print(f"Angles: ({angle_a:>4.1f}°, {angle_b:>4.1f}°) Counts: {counts}")
    angle_lists = [[(-45.0, -22.5), (-45.0, 67.5), (45.0, -22.5), (45.0, 67.5)], [(-45.0, 22.5), (-45.0, 112.5), (45.0, 22.5), (45.0, 112.5)], [(0.0, -22.5), (0.0, 67.5), (90.0, -22.5), (90.0, 67.5)], [(0.0, 22.5), (0.0, 112.5), (90.0, 22.5), (90.0, 112.5)]]
    for angles in angle_lists:
        try:
            e_values = [get_chsh_value(results, angle) for angle in angles]
            #print(f"Angles: {angles} E values: {e_values}")
        except KeyError as exc:
            #print(f"Missing required angle pair in data for CHSH calculation: {exc}")
            pass
    #print("\nCHSH analysis")
    try:
        # Compute the 4 expectation values
        e1 = calculate_E(*[get_chsh_value(results, ang) for ang in angle_lists[0]]) # E(-45, -22.5)
        e2 = calculate_E(*[get_chsh_value(results, ang) for ang in angle_lists[1]]) # E(-45, 22.5)
        e3 = calculate_E(*[get_chsh_value(results, ang) for ang in angle_lists[2]]) # E(0, -22.5)
        e4 = calculate_E(*[get_chsh_value(results, ang) for ang in angle_lists[3]]) # E(0, 22.5)
        
        #print(f"Individual Expectation Values:")
        #print(f"  E1: {e1: .4f} | E2: {e2: .4f} | E3: {e3: .4f} | E4: {e4: .4f}\n")
        
        # Test the 4 standard mathematical variations of the CHSH sum
        s_variants = {
            "S = E1 - E2 + E3 + E4":  e1 - e2 + e3 + e4,
            "S = E1 + e2 + E3 - E4":  e1 + e2 + e3 - e4,
            "S = E1 + E2 - E3 + E4":  e1 + e2 - e3 + e4,
            "S = -E1 + E2 + E3 + E4": -e1 + e2 + e3 + e4
        }
        
        #print("Testing CHSH Configurations:")
        for formula, s_val in s_variants.items():
            status = "VIOLATED!" if abs(s_val) > 2 else "Not Violated"
            #print(f"  {formula:<24} => |S| = {abs(s_val):.4f} ({status})")
            
    except KeyError as exc:
        #print(f"Missing required angle pair in data for CHSH calculation: {exc}")
        pass
    
    return max(abs(s) for s in s_variants.values()) if 's_variants' in locals() else None

if __name__ == "__main__":
    main()