# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================
# Adjust these values to control the learning algorithm and environment

# --- Genetic Algorithm - Population & Evolution ---
POPULATION_SIZE = 30        # Number of individuals per generation (larger = more exploration, slower)
ELITE_COUNT = 10             # Number of best individuals to keep each generation
GENERATIONS = 35            # Number of evolution cycles (more = longer search, better convergence)
EPISODES_PER_EVAL = 5       # Number of test runs to average for fitness (more = robust but slower)
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps in each episode before truncation

# --- Genetic Algorithm - Parameter Search Ranges ---
# These define the search space for PID gains. Negative values allow corrective action.
KP_RANGE = (-100, 100)      # Proportional gain range: responds to current error
KI_RANGE = (-100, 100)      # Integral gain range: responds to accumulated error over time
KD_RANGE = (-100, 100)      # Derivative gain range: responds to rate of change of error

PID_RANGES = [KP_RANGE, KI_RANGE, KD_RANGE,
              KP_RANGE, KI_RANGE, KD_RANGE]
# --- Genetic Algorithm - Evolution Parameters ---
MUTATION_RATE = 0.4        # Probability of mutating each gene (0.0-1.0, lower = more stable)
MUTATION_EFFECT = 0.5       # Size of mutations as fraction of current value (lower = finer tuning)
MUTATION_EFFECT_DECAY = 0.997  # Decay factor for mutation effect over generations
CROSSOVER_RATE = .95        # Probability of crossover between parents (0.0-1.0)
ELITISM = True              # Keep best individual in next generation (recommended: True)
VISUALIZE_ALL_INDIVIDUALS_DURING_TRAINING = False
VISUALIZE_BEST = True       # Whether to visualize the best individual of each generation