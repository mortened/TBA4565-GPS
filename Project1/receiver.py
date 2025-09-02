import numpy as np
from satellites import satellite_positions
from satellites import obs_data

phi_r = 63.2 #deg
lambda_r = 10.2 #deg
h_r = 100 #m

a = 6378137.0 # Semimajor axis [m]
b = 6356752.314245 # Semiminor axis [m]

c = 299792458 # Speed of light [m/s]

def N(a, b, phi):
    return a**2/(np.sqrt(a**2*(np.cos(phi)**2) + b**2*(np.sin(phi)**2)))


def geodeticToCartesian(phi_r, lambda_r, h_r):

    N_ = N(a, b, np.radians(phi_r))
    X = (N_ + h_r) * np.cos(np.radians(phi_r)) * np.cos(np.radians(lambda_r))
    Y = (N_ + h_r) * np.cos(np.radians(phi_r)) * np.sin(np.radians(lambda_r))
    Z = (((b**2)/(a**2))*N_ + h_r) * np.sin(np.radians(phi_r))

    return X, Y, Z

X, Y, Z = geodeticToCartesian(phi_r, lambda_r, h_r)
print("The approximate receiver coordinates in cartesian are:")
print(f"X: {X} m")
print(f"Y: {Y} m")
print(f"Z: {Z} m")

# Distance between single satellite and receiver
def rho(Xr, Yr, Zr, Xs, Ys, Zs):
    return float(np.sqrt((Xs - Xr)**2 + (Ys - Yr)**2 + (Zs - Zr)**2))

def ls(Xr, Yr, Zr, satellite_positions, obs_data, tol=1e-6):
    print(obs_data)
    done = False
    while not done:
        A = []
        L = []
        for sv in satellite_positions:
            Xs, Ys, Zs = satellite_positions[sv]
            rho_ = rho(Xr, Yr, Zr, Xs, Ys, Zs)
            # Build design matrix
            a_row = np.array([-(Xs - Xr)/rho_, -(Ys - Yr)/rho_, -(Zs - Zr)/rho_, -c])
            A.append(a_row)

            P = float(obs_data[obs_data["sv"] == sv]["P"].values[0])
            dt = float(obs_data[obs_data["sv"] == sv]["dt"].values[0])
            dion = float(obs_data[obs_data["sv"] == sv]["dion"].values[0])
            dtrop = float(obs_data[obs_data["sv"] == sv]["dtrop"].values[0])

            # Compute residual
            l = P - rho_ - c*dt - dion - dtrop
            L.append(l)
        L = np.array(L)

        # Compute ls solution
        A = np.array(A)
        L = np.array(L)
        delta = np.linalg.inv(A.T @ A) @ A.T @ L

        delta_X, delta_Y, delta_Z, dTr = delta

        # Update receiver position 
        Xr += delta_X
        Yr += delta_Y
        Zr += delta_Z

        # Check convergence
        if max(abs(delta_X), abs(delta_Y), abs(delta_Z), abs(dTr)) < tol:
            done = True
    
    print("Converged")
    print(f"Receiver position (X, Y, Z): ({Xr}, {Yr}, {Zr})")
    print(f"Clock bias (dTr): {dTr}")

    # Variance-covariance matrix
    Qx = np.linalg.inv(A.T @ A)

    PDOP = np.sqrt(Qx[0, 0] + Qx[1, 1] + Qx[2, 2])
    print(f"Position Dilution of Precision (PDOP): {PDOP}")

    return Xr, Yr, Zr, dTr

X_ls, Y_ls, Z_ls, dTr = ls(X, Y, Z, satellite_positions, obs_data)

def cartesianToGeodetic(X, Y, Z, tol=1e-12, max_iter=100):
    # Longitude
    lambda_ = np.rad2deg(np.arctan(Y / X))
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



receiver_geodetic = cartesianToGeodetic(X_ls, Y_ls, Z_ls)
print("The receiver coordinates in geodetic are:")
print(f"Latitude: {receiver_geodetic[0]} deg")
print(f"Longitude: {receiver_geodetic[1]} deg")
print(f"Height: {receiver_geodetic[2]} m")