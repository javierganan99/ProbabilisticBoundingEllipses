TAU = 10 # Time constant for the spatio-temporal density exponential decay
BUFFER_SIZE = 200 # Size of the buffer storing events to track
DENSITY_LIM = 0.5 # Density limit to star tracking
DENSITY_LIM_STOP = 0.1 # Density limit to stop tracking
TIMEOUT = 0.5 # Seconds to reach the desired density, if not reached start again
MIN_TAM = 0 # Min size to consider for the trackable object