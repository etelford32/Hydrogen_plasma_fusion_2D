import numpy as np
import pygame
from numba import njit, prange
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

# ------------------------
# Physical Constants
# ------------------------
E_CHARGE = 1.60217662e-19       # Elementary charge (C)
E_MASS = 9.10938356e-31         # Electron mass (kg)
D_MASS = 3.343583719e-27        # Deuteron mass (kg)
T_MASS = 5.0073567446e-27       # Triton mass (kg)
HE_MASS = 6.644657230e-27       # Alpha particle mass (kg)
NEUTRON_MASS = 1.674927471e-27  # Neutron mass (kg)
EPS_0 = 8.854187817e-12         # Vacuum permittivity (F/m)
MU_0 = 4e-7 * np.pi             # Vacuum permeability (H/m)
K_B = 1.380649e-23              # Boltzmann constant (J/K)
EV_TO_J = 1.602176634e-19       # Electronvolt to joules conversion

# ------------------------
# Simulation Parameters
# ------------------------
INCLUDE_ELECTRONS = True

# Particle counts
NUM_DEUTERIUM = 50
NUM_TRITIUM = 50
NUM_ELECTRONS = 100 if INCLUDE_ELECTRONS else 0
NUM_PARTICLES = NUM_DEUTERIUM + NUM_TRITIUM + NUM_ELECTRONS

# Domain and grid
DOMAIN_SIZE = 1e-4             # Domain size (m)
GRID_SIZE = 100                # Number of grid cells in each dimension
DX = DOMAIN_SIZE / GRID_SIZE   # Grid spacing (m)

# Time parameters
DT = 1e-10                     # Time step (s)
NUM_STEPS = 5000               # Number of simulation steps

# Plasma parameters
TEMPERATURE = 1e6              # Initial plasma temperature (K)

# Magnetic field
B0 = 5.0                       # Magnetic field strength (T)

# Fusion parameters
FUSION_SCALING_FACTOR = 1e30    # Increased scaling factor for fusion probability

# Collision parameters
R_COLL = DOMAIN_SIZE / GRID_SIZE * 2.0  # Collision radius (m)
R_MAG = DOMAIN_SIZE / GRID_SIZE * 1.0   # Magnetic interaction radius (m)

# Visualization parameters
SCREEN_SIZE = 800              # Screen size in pixels
PARTICLE_SIZE = 4              # Particle radius in pixels
FPS = 60                       # Frames per second

# Runtime Warning Fix Parameters
MIN_DISTANCE = 1e-12           # Minimum distance to prevent division by zero
MAX_FORCE = 1e5                # Maximum allowed force magnitude

# ------------------------
# Fusion Cross-Section Data (D-T Fusion)
# ------------------------
# Synthetic Gaussian approximation; replace with actual data for higher accuracy.

energy_keV = np.linspace(1, 100, 100)
sigma_barns = 1000 * np.exp(-((energy_keV - 20)**2) / (2 * 10**2))  # Peak around 20 keV
sigma_barns[sigma_barns < 1] = 1  # Minimum cross-section
sigma_m2 = sigma_barns * 1e-28  # Convert barns to m^2

# Create interpolation function
fusion_cross_section_interp = interp1d(energy_keV, sigma_m2, kind='cubic', bounds_error=False, fill_value=1e-28)

def get_fusion_cross_section(energy_J):
    """
    Get the fusion cross-section for a given relative kinetic energy (J).
    """
    energy_keV_real = energy_J / EV_TO_J / 1000  # Convert J to keV
    return fusion_cross_section_interp(energy_keV_real)

# ------------------------
# Initialize Pygame
# ------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Enhanced Hydrogen Plasma Fusion Simulation")
clock = pygame.time.Clock()

# ------------------------
# Initialize Particle Arrays
# ------------------------
species = np.zeros(NUM_PARTICLES, dtype=np.int32)
charges = np.zeros(NUM_PARTICLES, dtype=np.float64)
masses = np.zeros(NUM_PARTICLES, dtype=np.float64)

# Species Encoding:
# 0: Deuterium (D⁺)
# 1: Tritium (T⁺)
# 2: Electrons (e⁻)
# 3: Alpha Particles (He²⁺)
# 4: Neutrons (n⁰)

# Assign Deuterium
species[:NUM_DEUTERIUM] = 0
charges[:NUM_DEUTERIUM] = E_CHARGE
masses[:NUM_DEUTERIUM] = D_MASS

# Assign Tritium
species[NUM_DEUTERIUM:NUM_DEUTERIUM + NUM_TRITIUM] = 1
charges[NUM_DEUTERIUM:NUM_DEUTERIUM + NUM_TRITIUM] = E_CHARGE
masses[NUM_DEUTERIUM:NUM_DEUTERIUM + NUM_TRITIUM] = T_MASS

# Assign Electrons
if INCLUDE_ELECTRONS:
    species[NUM_DEUTERIUM + NUM_TRITIUM:] = 2
    charges[NUM_DEUTERIUM + NUM_TRITIUM:] = -E_CHARGE
    masses[NUM_DEUTERIUM + NUM_TRITIUM:] = E_MASS

# Initialize positions uniformly in the domain
positions = np.random.rand(NUM_PARTICLES, 2) * DOMAIN_SIZE

# Initialize velocities with Maxwellian distribution
velocities = np.random.normal(0, np.sqrt(K_B * TEMPERATURE / masses[:, np.newaxis]), (NUM_PARTICLES, 2))

# ------------------------
# Initialize Fields
# ------------------------
Ex = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
Ey = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
Bz = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float64) * B0  # Uniform external magnetic field

# ------------------------
# Initialize Particle Trails
# ------------------------
trail_length = 10  # Number of previous positions to store for each particle
particle_trails = [ [] for _ in range(NUM_PARTICLES) ]

# ------------------------
# Define Simulation Functions
# ------------------------

@njit(parallel=True)
def deposit_charge(positions, charges, grid_size, domain_size):
    rho = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in prange(positions.shape[0]):
        x_idx = int(positions[i, 0] / domain_size * grid_size)
        y_idx = int(positions[i, 1] / domain_size * grid_size)
        
        # Clamp indices to grid boundaries
        if x_idx >= grid_size:
            x_idx = grid_size - 1
        elif x_idx < 0:
            x_idx = 0
        if y_idx >= grid_size:
            y_idx = grid_size - 1
        elif y_idx < 0:
            y_idx = 0
        
        rho[x_idx, y_idx] += charges[i]
    return rho

@njit(parallel=True)
def calculate_phi(rho, dx, eps_0, tol=1e-5, max_iters=1000):
    # Solve Poisson's equation: ∇²φ = -rho / eps_0 using Gauss-Seidel iteration
    phi = np.zeros_like(rho)
    for it in range(max_iters):
        phi_old = phi.copy()
        for i in prange(1, rho.shape[0] - 1):
            for j in prange(1, rho.shape[1] - 1):
                phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] +
                                     phi[i, j+1] + phi[i, j-1] +
                                     (rho[i, j] * dx * dx) / eps_0)
        # Calculate the maximum difference to check convergence
        diff = 0.0
        for i in prange(rho.shape[0]):
            for j in prange(rho.shape[1]):
                d = abs(phi[i, j] - phi_old[i, j])
                if d > diff:
                    diff = d
        if diff < tol:
            break
    return phi

def calculate_E_fields(phi, dx):
    # Calculate electric fields from potential
    Ex = -(phi[1:, :] - phi[:-1, :]) / dx
    Ey = -(phi[:, 1:] - phi[:, :-1]) / dx
    # Pad Ex and Ey to match the grid size for particle access
    Ex_padded = np.pad(Ex, ((0,1),(0,0)), 'constant')
    Ey_padded = np.pad(Ey, ((0,0),(0,1)), 'constant')
    return Ex_padded, Ey_padded

@njit(parallel=True)
def boris_push(positions, velocities, charges, masses, Ex, Ey, Bz, dt, grid_size, domain_size):
    num_particles = positions.shape[0]
    for i in prange(num_particles):
        # Get particle's grid indices
        x_idx = int(positions[i, 0] / domain_size * grid_size)
        y_idx = int(positions[i, 1] / domain_size * grid_size)
        
        # Clamp indices to grid boundaries
        if x_idx >= grid_size:
            x_idx = grid_size - 1
        elif x_idx < 0:
            x_idx = 0
        if y_idx >= grid_size:
            y_idx = grid_size - 1
        elif y_idx < 0:
            y_idx = 0
        
        # Get electric and magnetic fields at particle's position
        E_particle = np.array([Ex[x_idx, y_idx], Ey[x_idx, y_idx]])
        B_particle_z = Bz[x_idx, y_idx]  # Magnetic field in z-direction
        
        # Charge-to-mass ratio
        qm = charges[i] / masses[i]
        
        # Half acceleration due to electric field
        velocities[i] += qm * E_particle * (dt / 2.0)
        
        # Rotation due to magnetic field (B only in z)
        t = qm * B_particle_z * (dt / 2.0)
        s = 2.0 * t / (1.0 + t * t)
        v_minus = velocities[i]
        
        # Cross product in 2D (magnetic field only in z, velocities in x and y)
        v_prime_x = v_minus[0] + v_minus[1] * t
        v_prime_y = v_minus[1] - v_minus[0] * t
        v_prime = np.array([v_prime_x, v_prime_y])
        
        velocities[i] = v_minus + s * v_prime
        
        # Half acceleration due to electric field
        velocities[i] += qm * E_particle * (dt / 2.0)
        
        # Update positions
        positions[i] += velocities[i] * dt
        
        # Reflective boundary conditions with clamping
        for dim in range(2):  # x and y
            if positions[i, dim] < 0:
                positions[i, dim] = -positions[i, dim]
                velocities[i, dim] = -velocities[i, dim]
            elif positions[i, dim] > domain_size:
                positions[i, dim] = 2 * domain_size - positions[i, dim]
                velocities[i, dim] = -velocities[i, dim]
        
        # **Explicit Clamping (Additional Safeguard)**
        positions[i, 0] = min(max(positions[i, 0], 0.0), domain_size)
        positions[i, 1] = min(max(positions[i, 1], 0.0), domain_size)
        
    return positions, velocities

@njit
def compute_relative_velocity(v1, v2):
    return np.linalg.norm(v1 - v2)

@njit(parallel=True)
def compute_pairwise_magnetic_forces(positions, velocities, charges, masses, R_mag, mu_0, dt):
    """
    Compute pairwise magnetic forces between charged particles.
    Simplified 2D magnetic force calculation.
    Returns:
        F_mag: Array of shape (num_particles, 2) representing magnetic forces.
    """
    num_particles = positions.shape[0]
    F_mag = np.zeros((num_particles, 2), dtype=np.float64)
    for i in prange(num_particles):
        for j in range(i + 1, num_particles):
            r_vec = positions[j] - positions[i]
            distance = np.linalg.norm(r_vec)
            if distance < R_mag and distance > MIN_DISTANCE:
                # Simplified magnetic force calculation
                v1 = velocities[i]
                v2 = velocities[j]
                v_rel = v1 - v2
                r_hat = r_vec / distance
                # Relative velocity perpendicular component
                v_rel_perp = v_rel - np.dot(v_rel, r_hat) * r_hat
                # Force magnitude
                F = (mu_0 / (4 * np.pi)) * (charges[i] * charges[j] * np.linalg.norm(v_rel_perp)) / (distance ** 2)
                
                # Clamp force to prevent overflow
                if F > MAX_FORCE:
                    F = MAX_FORCE
                elif F < -MAX_FORCE:
                    F = -MAX_FORCE
                
                # Direction: perpendicular to r_vec
                perp_dir = np.array([-r_hat[1], r_hat[0]])
                F_vector = F * perp_dir * dt
                F_mag[i] += F_vector
                F_mag[j] -= F_vector  # Newton's third law
    return F_mag

def identify_significant_pairs(F_mag, positions, velocities, charges, species, threshold=1e-25):
    """
    Identify particle pairs experiencing significant magnetic forces.
    Args:
        F_mag (np.ndarray): Array of shape (num_particles, 2) representing magnetic forces.
        positions (np.ndarray): Array of shape (num_particles, 2) representing positions.
        velocities (np.ndarray): Array of shape (num_particles, 2) representing velocities.
        charges (np.ndarray): Array of shape (num_particles,) representing charges.
        species (np.ndarray): Array of shape (num_particles,) representing species.
        threshold (float): Threshold for force magnitude to consider significant.
    Returns:
        significant_pairs (list of tuples): List of particle index pairs with significant forces.
    """
    significant_pairs = []
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Compute force magnitude between particles i and j
            r_vec = positions[j] - positions[i]
            distance = np.linalg.norm(r_vec)
            if distance < R_MAG and distance > MIN_DISTANCE:
                # Simplified magnetic force calculation (same as in Numba function)
                v_rel = velocities[i] - velocities[j]
                if distance > 0:
                    r_hat = r_vec / distance
                else:
                    r_hat = np.array([0.0, 0.0])
                v_rel_perp = v_rel - np.dot(v_rel, r_hat) * r_hat
                F = (MU_0 / (4 * np.pi)) * (charges[i] * charges[j] * np.linalg.norm(v_rel_perp)) / (distance ** 2)
                if F > threshold:
                    significant_pairs.append((i, j))
    return significant_pairs

def attempt_fusion_spatial(positions, velocities, charges, masses, species, dt, grid_size, domain_size):
    fusion_events = []
    
    # Calculate cell indices for all particles
    cell_indices_x = (positions[:, 0] / domain_size * grid_size).astype(np.int32)
    cell_indices_y = (positions[:, 1] / domain_size * grid_size).astype(np.int32)
    
    # Clamp indices to grid boundaries
    cell_indices_x = np.clip(cell_indices_x, 0, grid_size - 1)
    cell_indices_y = np.clip(cell_indices_y, 0, grid_size - 1)
    
    # Spatial grid for efficient collision detection
    grid = {}
    for idx in range(len(positions)):
        key = (cell_indices_x[idx], cell_indices_y[idx])
        if key in grid:
            grid[key].append(idx)
        else:
            grid[key] = [idx]
    
    # Iterate through grid cells and their neighbors for potential fusion
    for key, particle_indices in grid.items():
        neighbors = []
        x, y = key
        for dx_cell in [-1, 0, 1]:
            for dy_cell in [-1, 0, 1]:
                neighbor_key = (x + dx_cell, y + dy_cell)
                if neighbor_key in grid:
                    neighbors.extend(grid[neighbor_key])
        for i in particle_indices:
            for j in neighbors:
                if j <= i:
                    continue  # Avoid duplicate checks
                # Check if one is deuterium and the other is tritium
                if ((species[i] == 0 and species[j] == 1) or
                    (species[i] == 1 and species[j] == 0)):
                    # Calculate distance
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance > R_COLL:
                        continue  # Beyond collision radius
                    # Calculate relative kinetic energy (J)
                    relative_velocity = compute_relative_velocity(velocities[i], velocities[j])
                    kinetic_energy_J = 0.5 * masses[i] * relative_velocity**2  # Approximation
                    
                    # Get fusion cross-section (m^2)
                    sigma = get_fusion_cross_section(kinetic_energy_J)
                    
                    # Calculate fusion probability
                    P_fusion = sigma * relative_velocity * dt
                    P_fusion *= FUSION_SCALING_FACTOR  # Artificial scaling
                    
                    # Clamp probability to [0,1]
                    P_fusion = min(P_fusion, 1.0)
                    
                    # Determine if fusion occurs
                    if np.random.rand() < P_fusion:
                        fusion_events.append((i, j))
    
    print(f"Fusion Events This Step: {len(fusion_events)}")
    return fusion_events

def process_fusion_events(fusion_events, positions, velocities, charges, masses, species, particle_trails):
    if not fusion_events:
        return positions, velocities, charges, masses, species, particle_trails, 0

    fusion_count = 0
    removed_indices = set()
    new_positions = []
    new_velocities = []
    new_charges = []
    new_masses = []
    new_species = []
    new_trails = []

    for (i, j) in fusion_events:
        if i in removed_indices or j in removed_indices:
            continue  # Skip if already processed

        # Mark reactants for removal
        removed_indices.add(i)
        removed_indices.add(j)

        # Calculate the midpoint position for fusion products
        mid_pos = (positions[i] + positions[j]) / 2.0

        # Calculate total momentum before fusion
        total_momentum = masses[i] * velocities[i] + masses[j] * velocities[j]

        # Energy distribution (simplified)
        energy_alpha = 8.8e6 * EV_TO_J    # Energy to alpha particle
        energy_neutron = 8.8e6 * EV_TO_J  # Energy to neutron

        # Velocity calculations based on kinetic energy
        v_alpha = np.sqrt(2 * energy_alpha / HE_MASS)
        v_neutron = np.sqrt(2 * energy_neutron / NEUTRON_MASS)

        # Assign random direction for alpha particle
        theta = np.random.uniform(0, 2 * np.pi)
        alpha_velocity = v_alpha * np.array([np.cos(theta), np.sin(theta)])

        # Neutron velocity ensuring momentum conservation
        neutron_velocity = (total_momentum - HE_MASS * alpha_velocity) / NEUTRON_MASS

        # Ensure mid_pos is within domain
        mid_pos = np.clip(mid_pos, 0.0, DOMAIN_SIZE)

        # Append new fusion products
        # Alpha Particle: Cyan (0,255,255)
        new_positions.append(mid_pos)
        new_velocities.append(alpha_velocity)
        new_charges.append(2 * E_CHARGE)      # Alpha particle charge
        new_masses.append(HE_MASS)
        new_species.append(3)                  # Alpha Particle
        new_trails.append([])                  # Initialize trail for new particle

        # Neutron: Magenta (255,0,255)
        new_positions.append(mid_pos)
        new_velocities.append(neutron_velocity)
        new_charges.append(0.0)               # Neutron charge
        new_masses.append(NEUTRON_MASS)
        new_species.append(4)                  # Neutron
        new_trails.append([])                  # Initialize trail for new particle

        fusion_count += 1

    # Create a boolean mask to keep particles not in removed_indices
    if removed_indices:
        mask = np.ones(len(positions), dtype=bool)
        mask[list(removed_indices)] = False

        # Apply mask to all relevant arrays
        positions = positions[mask]
        velocities = velocities[mask]
        charges = charges[mask]
        masses = masses[mask]
        species = species[mask]

        # Update particle_trails by removing trails of fused particles
        particle_trails = [trail for idx, trail in enumerate(particle_trails) if mask[idx]]

    # Add new fusion products to arrays
    if new_positions:
        # Convert lists to NumPy arrays
        new_positions = np.array(new_positions)
        new_velocities = np.array(new_velocities)
        new_charges = np.array(new_charges)
        new_masses = np.array(new_masses)
        new_species = np.array(new_species)

        # Append to existing arrays
        positions = np.vstack([positions, new_positions])
        velocities = np.vstack([velocities, new_velocities])
        charges = np.hstack([charges, new_charges])
        masses = np.hstack([masses, new_masses])
        species = np.hstack([species, new_species])

        # Append new trails
        particle_trails.extend(new_trails)

    return positions, velocities, charges, masses, species, particle_trails, fusion_count

# ------------------------
# Machine Learning with PyTorch
# ------------------------

class FusionPredictor(nn.Module):
    def __init__(self):
        super(FusionPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

model = FusionPredictor()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data collection placeholders
features = []
labels = []

# ------------------------
# Diagnostic Functions
# ------------------------

def compute_total_charge(rho):
    return np.sum(rho) * (DX ** 2)  # Multiply by area element

def compute_total_energy(rho, phi, velocities, masses, Bz):
    """
    Compute the total energy of the system.
    
    Args:
        rho (np.ndarray): Charge density grid (shape: GRID_SIZE x GRID_SIZE).
        phi (np.ndarray): Electric potential grid (shape: GRID_SIZE x GRID_SIZE).
        velocities (np.ndarray): Particle velocities (shape: num_particles x 2).
        masses (np.ndarray): Particle masses (shape: num_particles,).
        Bz (np.ndarray): Magnetic field grid (shape: GRID_SIZE x GRID_SIZE).
    
    Returns:
        float: Total energy of the system.
    """
    # Kinetic Energy: 0.5 * m * v^2
    kinetic_energy = 0.5 * masses * np.sum(velocities**2, axis=1)
    total_kinetic_energy = np.sum(kinetic_energy)
    
    # Potential Energy: 0.5 * sum(rho * phi) * (dx^2)
    potential_energy = 0.5 * np.sum(rho * phi) * (DX ** 2)
    
    # Magnetic Energy: 0.5 * mu_0 * B^2 * (dx^2)
    magnetic_energy = 0.5 * MU_0 * np.sum(Bz**2) * (DX ** 2)
    
    total_energy = total_kinetic_energy + potential_energy + magnetic_energy
    return total_energy

def validate_particles(charges, velocities):
    valid_mask = ~(np.isnan(charges) | np.isinf(charges) |
                  np.isnan(velocities).any(axis=1) | np.isinf(velocities).any(axis=1))
    return valid_mask

def enforce_charge_neutrality(charges, tolerance=1e-12):
    total_charge = np.sum(charges)
    if abs(total_charge) > tolerance:
        correction = total_charge / len(charges)
        charges -= correction  # Distribute the charge imbalance uniformly
        print(f"Enforced charge neutrality by adjusting charges by {correction:.2e} C each.")
    return charges

def monitor_velocities(velocities, threshold=1e6):
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    if max_velocity > threshold:
        print(f"Warning: High velocity detected! Max Velocity = {max_velocity:.2e} m/s")
    return max_velocity

# ------------------------
# UI Functions
# ------------------------

def add_particle(mouse_pos, particle_type, positions, velocities, charges, masses, species, particle_trails):
    # Convert mouse position to simulation coordinates
    x = mouse_pos[0] / SCREEN_SIZE * DOMAIN_SIZE
    y = mouse_pos[1] / SCREEN_SIZE * DOMAIN_SIZE
    new_pos = np.array([x, y])
    
    # Assign properties based on particle_type
    if particle_type == 'deuterium':
        charge = E_CHARGE
        mass = D_MASS
        specie = 0
        velocity = np.random.normal(0, np.sqrt(K_B * TEMPERATURE / mass), 2)
    elif particle_type == 'tritium':
        charge = E_CHARGE
        mass = T_MASS
        specie = 1
        velocity = np.random.normal(0, np.sqrt(K_B * TEMPERATURE / mass), 2)
    elif particle_type == 'electron':
        charge = -E_CHARGE
        mass = E_MASS
        specie = 2
        velocity = np.random.normal(0, np.sqrt(K_B * TEMPERATURE / mass), 2)
    else:
        # Undefined particle type
        return positions, velocities, charges, masses, species, particle_trails
    
    # Append the new particle
    positions = np.vstack([positions, new_pos])
    velocities = np.vstack([velocities, velocity])
    charges = np.hstack([charges, charge])
    masses = np.hstack([masses, mass])
    species = np.hstack([species, specie])
    particle_trails.append([])  # Initialize empty trail for the new particle
    
    return positions, velocities, charges, masses, species, particle_trails

# ------------------------
# Main Simulation Loop
# ------------------------
running_simulation = True
step = 0
fusion_count_total = 0  # To keep track of total fusion events

# UI Variables
current_particle_type = 'deuterium'  # Default particle type

# End State Variables
MAX_FUSION_EVENTS = 100
neutrality_steps_required = 100
neutrality_counter = 0
previous_energy = None  # To be initialized after first energy computation

while running_simulation and step < NUM_STEPS:
    # Event handling for Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running_simulation = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running_simulation = False
                print("Simulation terminated by user.")
                break
            elif event.key == pygame.K_d:
                current_particle_type = 'deuterium'
                print("Selected particle type: Deuterium")
            elif event.key == pygame.K_t:
                current_particle_type = 'tritium'
                print("Selected particle type: Tritium")
            elif event.key == pygame.K_e:
                current_particle_type = 'electron'
                print("Selected particle type: Electron")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if event.button == 1:  # Left-click
                positions, velocities, charges, masses, species, particle_trails = add_particle(
                    mouse_pos, 'deuterium', positions, velocities, charges, masses, species, particle_trails
                )
                print(f"Added Deuterium at {mouse_pos}")
            elif event.button == 3:  # Right-click
                positions, velocities, charges, masses, species, particle_trails = add_particle(
                    mouse_pos, 'tritium', positions, velocities, charges, masses, species, particle_trails
                )
                print(f"Added Tritium at {mouse_pos}")
            elif event.button == 2:  # Middle-click
                positions, velocities, charges, masses, species, particle_trails = add_particle(
                    mouse_pos, 'electron', positions, velocities, charges, masses, species, particle_trails
                )
                print(f"Added Electron at {mouse_pos}")
    
    # Deposit charge
    rho = deposit_charge(positions, charges, GRID_SIZE, DOMAIN_SIZE)

    # Calculate potential phi with convergence
    phi = calculate_phi(rho, DX, EPS_0)

    # Calculate electric fields Ex and Ey outside Numba
    Ex_padded, Ey_padded = calculate_E_fields(phi, DX)

    # Push particles
    positions, velocities = boris_push(positions, velocities, charges, masses, Ex_padded, Ey_padded, Bz, DT, GRID_SIZE, DOMAIN_SIZE)

    # Compute and apply pairwise magnetic forces
    F_mag = compute_pairwise_magnetic_forces(positions, velocities, charges, masses, R_mag=R_MAG, mu_0=MU_0, dt=DT)
    velocities += F_mag  # Update velocities based on magnetic forces

    # Check for invalid values in positions and velocities
    valid_mask = validate_particles(charges, velocities)
    if not np.all(valid_mask):
        num_invalid = np.sum(~valid_mask)
        print(f"Step {step}: Invalid Particles Detected = {num_invalid}. Removing them.")
        positions = positions[valid_mask]
        velocities = velocities[valid_mask]
        charges = charges[valid_mask]
        masses = masses[valid_mask]
        species = species[valid_mask]
        # Remove trails for invalid particles
        particle_trails = [trail for idx, trail in enumerate(particle_trails) if valid_mask[idx]]

    # Identify significant_pairs after ensuring all positions are valid
    significant_pairs = identify_significant_pairs(F_mag, positions, velocities, charges, species, threshold=1e-25)

    # Attempt fusion with spatial collision detection
    fusion_events = attempt_fusion_spatial(positions, velocities, charges, masses, species, DT, GRID_SIZE, DOMAIN_SIZE)

    # Process fusion events
    if fusion_events:
        positions, velocities, charges, masses, species, particle_trails, fusion_count = process_fusion_events(
            fusion_events, positions, velocities, charges, masses, species, particle_trails
        )
        fusion_count_total += fusion_count

        # Collect data for machine learning (optional)
        for (i, j) in fusion_events:
            # Ensure indices are still valid after fusion processing
            if i < len(positions) and j < len(positions):
                # Features: [x_i, y_i, vx_i, vy_i, x_j, y_j, vx_j, vy_j]
                feature = np.concatenate([positions[i], velocities[i], positions[j], velocities[j]])
                features.append(feature)
                labels.append(1)  # Fusion occurred

    # Update particle trails
    for i in range(len(positions)):
        if len(particle_trails[i]) >= trail_length:
            particle_trails[i].pop(0)  # Remove the oldest position
        trail_pos = (
            int(positions[i, 0] / DOMAIN_SIZE * SCREEN_SIZE),
            int(positions[i, 1] / DOMAIN_SIZE * SCREEN_SIZE)
        )
        particle_trails[i].append(trail_pos)

    # Compute total charge
    total_charge = compute_total_charge(rho)

    # Compute total energy and check for equilibrium
    current_energy = compute_total_energy(rho, phi, velocities, masses, Bz)
    if previous_energy is not None:
        energy_change = abs(current_energy - previous_energy)
        if energy_change < 1e-20:
            print("Energy equilibrium reached. Stopping simulation.")
            running_simulation = False
    previous_energy = current_energy

    # Check charge neutrality and monitor velocities every 100 steps
    if step % 100 == 0:
        if abs(total_charge) < 1e-12:
            neutrality_counter += 1
            if neutrality_counter >= neutrality_steps_required:
                print("Charge neutrality achieved and stable. Stopping simulation.")
                running_simulation = False
        else:
            neutrality_counter = 0
            # Optionally enforce charge neutrality
            charges = enforce_charge_neutrality(charges)
        
        avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
        max_velocity = monitor_velocities(velocities)
        print(f"Step {step}: Avg Velocity = {avg_velocity:.2e} m/s, Max Velocity = {max_velocity:.2e} m/s, Total Charge = {total_charge:.2e} C")

    # Visualization
    screen.fill((0, 0, 0))  # Clear screen with black

    # Draw boundary walls
    boundary_color = (255, 255, 255)  # White
    boundary_thickness = 5
    pygame.draw.rect(screen, boundary_color, pygame.Rect(0, 0, SCREEN_SIZE, SCREEN_SIZE), boundary_thickness)

    # Optional: Draw magnetic force lines
    for (i, j) in significant_pairs:
        x1 = int(positions[i, 0] / DOMAIN_SIZE * SCREEN_SIZE)
        y1 = int(positions[i, 1] / DOMAIN_SIZE * SCREEN_SIZE)
        x2 = int(positions[j, 0] / DOMAIN_SIZE * SCREEN_SIZE)
        y2 = int(positions[j, 1] / DOMAIN_SIZE * SCREEN_SIZE)
        pygame.draw.line(screen, (0, 0, 255), (x1, y1), (x2, y2), 1)  # Blue lines for magnetic forces

    # Draw particles and their trails
    for i in range(len(positions)):
        x = int(positions[i, 0] / DOMAIN_SIZE * SCREEN_SIZE)
        y = int(positions[i, 1] / DOMAIN_SIZE * SCREEN_SIZE)

        # Ensure particles are within screen bounds
        x = min(max(x, 0), SCREEN_SIZE - 1)
        y = min(max(y, 0), SCREEN_SIZE - 1)

        # Assign colors based on species
        if species[i] == 0:
            color = (0, 0, 255)          # Deuterium: Blue
        elif species[i] == 1:
            color = (0, 255, 0)          # Tritium: Green
        elif species[i] == 2:
            color = (255, 255, 255)      # Electrons: White
        elif species[i] == 3:
            color = (0, 255, 255)        # Alpha Particle: Cyan (combined color)
        elif species[i] == 4:
            color = (255, 0, 255)        # Neutron: Magenta (combined color)
        else:
            color = (255, 255, 255)      # Undefined species: White

        # Draw trails
        if len(particle_trails[i]) > 1:
            pygame.draw.lines(screen, color, False, particle_trails[i], 1)
        
        # Draw the particle
        pygame.draw.circle(screen, color, (x, y), PARTICLE_SIZE)

    # Display Fusion Count, Step, Total Charge, and Current Particle Type
    font = pygame.font.SysFont(None, 24)
    fusion_text = font.render(f'Fusion Events: {fusion_count_total}', True, (255, 255, 255))
    step_text = font.render(f'Step: {step}', True, (255, 255, 255))
    charge_color = (0, 255, 0) if abs(total_charge) < 1e-12 else (255, 0, 0)
    charge_text = font.render(f'Total Charge: {total_charge:.2e} C', True, charge_color)
    particle_type_text = font.render(f'Current Particle: {current_particle_type.capitalize()}', True, (255, 255, 255))
    screen.blit(fusion_text, (10, 10))
    screen.blit(step_text, (10, 30))
    screen.blit(charge_text, (10, 50))
    screen.blit(particle_type_text, (10, 70))

    # Update display
    pygame.display.flip()
    clock.tick(FPS)
    step += 1

# Print total fusion events upon completion
print(f"Total Fusion Events: {fusion_count_total}")

# Quit Pygame
pygame.quit()
sys.exit()