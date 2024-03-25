import numpy as np
from random import uniform
import time

class Particle:
    def __init__(self, x, y, ang_vel):
        self.x = x
        self.y = y
        self.ang_vel = ang_vel

class ParticleSimulator:
    def __init__(self, particles):
        self.particles = particles

    def evolve_python(self, dt):
        timestep = 0.00001
        nsteps = int(dt / timestep)
        for i in range(nsteps):
            for p in self.particles:
                norm = (p.x**2 + p.y**2)**0.5
                v_x = (-p.y) / norm
                v_y = p.x / norm
                d_x = timestep * p.ang_vel * v_x
                d_y = timestep * p.ang_vel * v_y
                p.x += d_x
                p.y += d_y

    def evolve_numpy(self, dt):
        timestep = 0.00001
        nsteps = int(dt / timestep)
        r_i = np.array([[p.x, p.y] for p in self.particles])
        ang_vel_i = np.array([p.ang_vel for p in self.particles])
        for i in range(nsteps):
            norm_i = np.sqrt((r_i ** 2).sum(axis=1))
            v_i = r_i[:, [1, 0]]
            v_i[:, 0] *= -1
            v_i /= norm_i[:, np.newaxis]
            d_i = timestep * ang_vel_i[:, np.newaxis] * v_i
            r_i += d_i
        for i, p in enumerate(self.particles):
            p.x, p.y = r_i[i]

def benchmark(npart, method):
    particles = [Particle(uniform(-1.0, 1.0), uniform(-1.0, 1.0), uniform(-1.0, 1.0)) for i in range(npart)]
    simulator = ParticleSimulator(particles)
    start = time.time()
    if method == 'python':
        simulator.evolve_python(0.1)
    elif method == 'numpy':
        simulator.evolve_numpy(0.1)
    end = time.time()
    return end - start
