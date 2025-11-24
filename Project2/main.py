import numpy as np
import pandas as pd
from data import *

def N(a, b, phi):
    return a**2/(np.sqrt(a**2*(np.cos(phi)**2) + b**2*(np.sin(phi)**2)))

def geodeticToCartesian(phi_r, lambda_r, h_r):

    N_ = N(a, b, np.radians(phi_r))
    X = (N_ + h_r) * np.cos(np.radians(phi_r)) * np.cos(np.radians(lambda_r))
    Y = (N_ + h_r) * np.cos(np.radians(phi_r)) * np.sin(np.radians(lambda_r))
    Z = (((b**2)/(a**2))*N_ + h_r) * np.sin(np.radians(phi_r))

    return float(X), float(Y), float(Z)

# Task 1: Transform the receiver coordinates to Cartesian coordinates

base_A_cartesian = geodeticToCartesian(*base_A_geodetic)
rover_B_cartesian = geodeticToCartesian(*rover_B_geodetic)

print("Base Station A Cartesian Coordinates:")
print(f"X: {base_A_cartesian[0]} m")
print(f"Y: {base_A_cartesian[1]} m")
print(f"Z: {base_A_cartesian[2]} m")

print("Rover B Cartesian Coordinates:")
print(f"X: {rover_B_cartesian[0]} m")
print(f"Y: {rover_B_cartesian[1]} m")
print(f"Z: {rover_B_cartesian[2]} m")

# Design the observation equation and estimate the receiver position B in the Cartesian
# coordinates together with 4 unknown double difference phase ambiguities. The satellite
# coordinates together with the approximate receiver position are used to compute the
# design matrix A. Estimate the variance covariance matrix of the unknown position and
# ambiguities. See the Appendix. This is a float solution. Also show the variance
# covariance matrix of the float solution in your report.

# Distance between single satellite and receiver
def rho(Xr, Yr, Zr, Xs, Ys, Zs):
    return float(np.sqrt((Xs - Xr)**2 + (Ys - Yr)**2 + (Zs - Zr)**2))

# Design matrix entry
def a_AB_ij(XYZ_AB, XYZ_i, XYZ_j, rho_i, rho_j):
    return -((XYZ_j - XYZ_AB)/rho_j - (XYZ_i - XYZ_AB)/rho_i)

wavelength = c / l1_frequency

def A_row(XYZ_AB, sat_i, sat_j):
    rho_i = rho(*XYZ_AB, sat_i["X"], sat_i["Y"], sat_i["Z"])
    rho_j = rho(*XYZ_AB, sat_j["X"], sat_j["Y"], sat_j["Z"])
    a_x = -((sat_j["X"] - XYZ_AB[0]) / rho_j - (sat_i["X"] - XYZ_AB[0]) / rho_i)
    a_y = -((sat_j["Y"] - XYZ_AB[1]) / rho_j - (sat_i["Y"] - XYZ_AB[1]) / rho_i)
    a_z = -((sat_j["Z"] - XYZ_AB[2]) / rho_j - (sat_i["Z"] - XYZ_AB[2]) / rho_i)
    return [a_x, a_y, a_z]

def A(rover_B, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities=None):
    if ambiguities is None:
        return np.array([
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[1]) + [wavelength, 0, 0, 0],
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[2]) + [0, wavelength, 0, 0],
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[3]) + [0, 0, wavelength, 0],
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[4]) + [0, 0, 0, wavelength],
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[1]) + [wavelength, 0, 0, 0],
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[2]) + [0, wavelength, 0, 0],
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[3]) + [0, 0, wavelength, 0],
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[4]) + [0, 0, 0, wavelength],
        ])
    else:
        return np.array([
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[1]),
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[2]),
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[3]),
            A_row(rover_B, sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[4]),
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[1]),
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[2]),
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[3]),
            A_row(rover_B, sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[4]),
        ])



def phi_AB_ij(Li_A, Lj_A, Li_B, Lj_B):
    return ((Lj_B - Li_B) - (Lj_A - Li_A)) * wavelength

def l_row(sat_from_baseA_i, sat_from_baseA_j, sat_from_RoverB_i, sat_from_roverB_j, rover_B, base_A):
    val=(
        phi_AB_ij(
            sat_from_baseA_i["L1"],
            sat_from_baseA_j["L1"],
            sat_from_RoverB_i["L1"],
            sat_from_roverB_j["L1"],
        ) - (rho(*rover_B, sat_from_roverB_j["X"], sat_from_roverB_j["Y"], sat_from_roverB_j["Z"]))
            + (rho(*rover_B, sat_from_RoverB_i["X"], sat_from_RoverB_i["Y"], sat_from_RoverB_i["Z"]))  
            + (rho(*base_A, sat_from_baseA_j["X"], sat_from_baseA_j["Y"], sat_from_baseA_j["Z"]))
            - (rho(*base_A, sat_from_baseA_i["X"], sat_from_baseA_i["Y"], sat_from_baseA_i["Z"]))
    )
    return float(val)

def L(rover_B, base_A, sat_from_baseA_t1, sat_from_baseA_t2, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities=None):
    terms = [
        l_row(sat_from_baseA_t1.iloc[0], sat_from_baseA_t1.iloc[1], sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[1], rover_B, base_A) - (wavelength * ambiguities[0] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t1.iloc[0], sat_from_baseA_t1.iloc[2], sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[2], rover_B, base_A) - (wavelength * ambiguities[1] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t1.iloc[0], sat_from_baseA_t1.iloc[3], sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[3], rover_B, base_A) - (wavelength * ambiguities[2] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t1.iloc[0], sat_from_baseA_t1.iloc[4], sat_from_roverB_t1.iloc[0], sat_from_roverB_t1.iloc[4], rover_B, base_A) - (wavelength * ambiguities[3] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t2.iloc[0], sat_from_baseA_t2.iloc[1], sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[1], rover_B, base_A) - (wavelength * ambiguities[0] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t2.iloc[0], sat_from_baseA_t2.iloc[2], sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[2], rover_B, base_A) - (wavelength * ambiguities[1] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t2.iloc[0], sat_from_baseA_t2.iloc[3], sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[3], rover_B, base_A) - (wavelength * ambiguities[2] if ambiguities is not None else 0.0),
        l_row(sat_from_baseA_t2.iloc[0], sat_from_baseA_t2.iloc[4], sat_from_roverB_t2.iloc[0], sat_from_roverB_t2.iloc[4], rover_B, base_A) - (wavelength * ambiguities[3] if ambiguities is not None else 0.0),
    ]
    return np.array(terms, dtype=float).reshape(-1, 1)  


# Weight matrix of double differences
P = (1 / (2* sigma**2)) * (1/5) * np.array([
    [4, -1, -1, -1, 0, 0, 0, 0],
    [-1, 4, -1, -1, 0, 0, 0, 0],
    [-1, -1, 4, -1, 0, 0, 0, 0],
    [-1, -1, -1, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, -1, -1, -1],
    [0, 0, 0, 0, -1, 4, -1, -1],
    [0, 0, 0, 0, -1, -1, 4, -1],
    [0, 0, 0, 0, -1, -1, -1, 4]
])

def LS_estimate(A, L, P):
    At_P_A_inv = np.linalg.inv(A.T @ P @ A)
    x_hat = At_P_A_inv @ A.T @ P @ L
    C_x = At_P_A_inv
    return x_hat, C_x

# LS estimation with iteration
rover_B = rover_B_cartesian
base_A = base_A_cartesian

def LS_iteration(rover_B, base_A, sat_from_baseA_t1, sat_from_baseA_t2, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities=None, tol=1e-6, max_iterations=10):

    for i in range(max_iterations):
        A_matrix = A(rover_B, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities)
        L_matrix = L(rover_B, base_A, sat_from_baseA_t1, sat_from_baseA_t2, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities)
        x_hat, C_x = LS_estimate(A_matrix, L_matrix, P)
        
        rover_B = rover_B + x_hat[:3].flatten()
        
        if(max(abs(x_hat[:3].flatten())) < tol):
            print("Convergence achieved.")
            break
    else:
        print("Maximum iterations reached without convergence.")
    amb = x_hat[3:].flatten() if ambiguities is None else ambiguities
    return rover_B, amb, C_x

final_position, final_ambiguities, final_covariance = LS_iteration(
    rover_B, base_A,
    sat_from_baseA_t1, sat_from_baseA_t2,
    sat_from_roverB_t1, sat_from_roverB_t2
)

print("Final Estimated Receiver Position (X, Y, Z):", final_position)
print("Estimated Double Difference Ambiguities:", final_ambiguities)


C_x = final_covariance
print("Variance-Covariance Matrix of the Float Solution:")
print(C_x)
print("Standard Deviations of the Estimated Parameters:")
std_devs = np.sqrt(np.diag(C_x))
pos_std_devs = std_devs[:3]
amb_std_devs = std_devs[3:]
print("Position Standard Deviations (X, Y, Z):", pos_std_devs)
print("Ambiguities Standard Deviations:", amb_std_devs)


# Task 3b): Fix ambiguities to nearest integer
fixed_ambiguities_int = np.round(final_ambiguities).astype(int)
pos_with_fixed_ambiguities, _, cov_with_fixed_ambiguities = LS_iteration(
    rover_B, base_A,
    sat_from_baseA_t1, sat_from_baseA_t2,
    sat_from_roverB_t1, sat_from_roverB_t2,
    ambiguities=fixed_ambiguities_int
)

print("Receiver Position with Fixed Ambiguities (X, Y, Z):", pos_with_fixed_ambiguities)

#Convert to geodetic
def cartesianToGeodetic(X, Y, Z, tol=1e-12, max_iter=100):
    # Longitude
    lambda_ = np.rad2deg(np.atan2(Y, X))
    e2 = (a**2 - b**2)/a**2
    p = np.sqrt(X**2 + Y**2)
    phi_0 = np.arctan((Z/p) * (1-e2)**-1)
    i = 0
    while i < max_iter:
        i += 1
        N_0 = N(a, b, phi_0)
        h = p/np.cos(phi_0) - N_0
        phi = np.arctan((Z/p) * (1-(e2 * N_0/(N_0 + h)))**-1)
        if abs(phi - phi_0) < tol:
            break
        phi_0 = phi

    print(f"Converged after {i} iterations")
    # Final height with correct phi
    Nf = N(a, b, phi)
    h = p/np.cos(phi) - Nf

    return np.rad2deg(phi), lambda_, h

receiver_geodetic = cartesianToGeodetic(*pos_with_fixed_ambiguities)
print("The receiver coordinates in geodetic are:")
print(f"Latitude: {receiver_geodetic[0]} degrees")
print(f"Longitude: {receiver_geodetic[1]} degrees")
print(f"Height: {receiver_geodetic[2]} m")

# Task 3c): Fix ambiguites considering their standard deviations

# Test 3 standard deviations around the float solution
from itertools import product
def generate_ambiguity_combinations(float_amb, amb_std, k=3):
    # float_amb, amb_std are in cycles
    ranges = []
    for a, s in zip(float_amb, amb_std):
        lo = int(np.floor(a - k*s))
        hi = int(np.ceil(a + k*s))
        ranges.append(range(lo, hi+1))
    return list(product(*ranges))  # list of 4-tuples

def weighted_ssr(A_fixed, L_fixed, P):
    # residuals at LS solution: v = A x_hat - L
    x_hat_xyz, _ = LS_estimate(A_fixed, L_fixed, P)  # 3×1
    v = A_fixed @ x_hat_xyz - L_fixed
    return float(v.T @ P @ v), x_hat_xyz

def evaluate_candidate(rover_B_seed, base_A, amb_tuple):
    # 1) iterate to a position with these fixed ambiguities
    pos_xyz, _, _ = LS_iteration(
        rover_B_seed, base_A,
        sat_from_baseA_t1, sat_from_baseA_t2,
        sat_from_roverB_t1, sat_from_roverB_t2,
        ambiguities=np.array(amb_tuple, dtype=float)
    )
    # 2) form A and L for final position & fixed ambiguities (3 unknowns only)
    A_fixed = A(pos_xyz, sat_from_roverB_t1, sat_from_roverB_t2, ambiguities=np.array(amb_tuple, dtype=float))
    L_fixed = L(pos_xyz, base_A, sat_from_baseA_t1, sat_from_baseA_t2,
                sat_from_roverB_t1, sat_from_roverB_t2, ambiguities=np.array(amb_tuple, dtype=float))
    # 3) weighted SSR
    ssr, x_hat_xyz = weighted_ssr(A_fixed, L_fixed, P)
    return ssr, pos_xyz, A_fixed, L_fixed

# Build candidates using +-3 sigma (scenario c)
cands = generate_ambiguity_combinations(final_ambiguities, amb_std_devs, k=3)
print(f"Testing {len(cands)} ambiguity combinations...")

results = []
for ambs in cands:
    ssr, pos_xyz, A_fixed, L_fixed = evaluate_candidate(rover_B_cartesian, base_A_cartesian, ambs)
    results.append((ssr, ambs, pos_xyz))

# Sort by weighted SSR (v^T P v)
results.sort(key=lambda t: t[0])
best_ssr, best_ambs, best_pos = results[0]
second_ssr = results[1][0] if len(results) > 1 else np.inf
ratio = second_ssr / best_ssr

print("Best ambiguities (cycles):", best_ambs)
print("Best weighted SSR v^T P v:", best_ssr)
print("Second/best ratio:", ratio)


chosen_pos = best_pos  # if ratio >= 2 or 3

# Convert to geodetic
chosen_geodetic = cartesianToGeodetic(*chosen_pos)
print("Chosen Receiver Position in Geodetic Coordinates:")
print(f"Latitude: {chosen_geodetic[0]} degrees")
print(f"Longitude: {chosen_geodetic[1]} degrees")
print(f"Height: {chosen_geodetic[2]} m")

