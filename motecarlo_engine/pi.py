import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(num_samples: int):
    """
    Estimates the value of Pi using a Monte Carlo simulation.
    """
    # 1. Generate random (x, y) coordinates between -1 and 1
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    # 2. Calculate distance from the origin (0,0) for all points
    distance_squared = x*2 + y*2
    
    # 3. Identify points that fall inside the unit circle (radius <= 1)
    inside_circle = distance_squared <= 1
    num_inside = np.sum(inside_circle)
    
    # 4. Pi estimate = 4 * (points inside circle) / (total points)
    pi_estimate = 4 * num_inside / num_samples
    
    return pi_estimate, x, y, inside_circle

# --- Run the Simulation ---
samples = 10000
pi_val, x_vals, y_vals, inside_mask = monte_carlo_pi(samples)

print(f"Total Samples: {samples}")
print(f"Estimated Pi:  {pi_val}")
print(f"Actual Pi:     {np.pi}")
print(f"Error Margin:  {abs(np.pi - pi_val):.6f}")

# --- Plot the Results ---
plt.figure(figsize=(6, 6))
# Plot points inside the circle in green, outside in red
plt.scatter(x_vals[inside_mask], y_vals[inside_mask], color='green', s=1, label='Inside Circle')
plt.scatter(x_vals[~inside_mask], y_vals[~inside_mask], color='red', s=1, label='Outside Circle')

# Add visual formatting
plt.title(f"Monte Carlo $\\pi$ Estimation\nEstimate: {pi_val} (Samples: {samples})")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend(loc="upper right")
plt.axis('equal')
plt.show()
