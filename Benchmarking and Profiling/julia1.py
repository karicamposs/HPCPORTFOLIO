import time
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt

# Defining global constants for the coordinate space
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn:{fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time

@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs), build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
    
    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    print(f"calculate_z_serial_purepython took {end_time - start_time} seconds")
    assert sum(output) == 33219980
    
    return output

def visualize_julia_set(output, desired_width):
    """Visualize the Julia Set"""
    output_2d = [output[i:i+desired_width] for i in range(0, desired_width*desired_width, desired_width)]
    output_false_gray = np.array(output_2d)
    output_false_gray = np.log(output_false_gray + 1)
    output_pure_gray = np.array(output_2d)
    max_iter = np.max(output_pure_gray)
    output_pure_gray = output_pure_gray / max_iter

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(output_false_gray, cmap="gray", extent=(x1, x2, y1, y2))
    axes[0].set_title("Julia Set - False Gray Scale")
    axes[1].imshow(output_pure_gray, cmap="gray", extent=(x1, x2, y1, y2))
    axes[1].set_title("Julia Set - Pure Gray Scale")
    plt.show()

if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    output = calc_pure_python(desired_width=1000, max_iterations=300)
    visualize_julia_set(output, desired_width=1000)
