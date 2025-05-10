import numpy as np
from utils import timer
import pygame


BLACK = (0, 0, 0)

PRESET_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
]


class Body:
    def __init__(self, mass, position, velocity, body_id):

        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.colour = PRESET_COLORS[body_id]

        glow_size = 200
        self.glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)

        glow_x = glow_size / 2
        glow_y = glow_x
        for radius, alpha in zip(range(67, 0, -5), range(1, 30, 5)):
            pygame.draw.circle(
                self.glow_surface, (*self.colour, alpha), (glow_x, glow_y), radius
            )

    def compute_acceleration(self, other_bodies, G=1.0, softening=0.2):
        acceleration = np.zeros(2)
        for other in other_bodies:
            if other is not self:
                r = other.position - self.position
                distance = np.linalg.norm(r)

                factor = G * other.mass / max(distance**3, softening**3)
                acceleration += factor * r
        return acceleration

    def draw(self, frame, system, positions, win):

        scale = 200
        x = positions[frame, system.bodies.index(self), 0] * scale + 500
        y = 1000 - (positions[frame, system.bodies.index(self), 1] * scale + 500)

        win.blit(
            self.glow_surface,
            (
                x - self.glow_surface.get_width() // 2,
                y - self.glow_surface.get_height() // 2,
            ),
        )

        pygame.draw.circle(win, self.colour, (x, y), 10)

        trail_surface = pygame.Surface(
            (win.get_width(), win.get_height()), pygame.SRCALPHA
        )

        trail_length = positions.shape[0] - 1
        for i in range(trail_length, 1, -1):
            point1 = positions[frame - i, system.bodies.index(self)]
            point2 = positions[frame - i + 1, system.bodies.index(self)]
            x1 = point1[0] * scale + 500
            y1 = 1000 - (point1[1] * scale + 500)
            x2 = point2[0] * scale + 500
            y2 = 1000 - (point2[1] * scale + 500)
            fade_factor = int(255 * (1 - i / trail_length))
            colour_with_alpha = (*self.colour[:3], fade_factor)
            pygame.draw.line(trail_surface, colour_with_alpha, (x1, y1), (x2, y2), 3)

        win.blit(trail_surface, (0, 0))

    def get_state(self):
        return np.array(
            [self.position[0], self.position[1], self.velocity[0], self.velocity[1]]
        )


class System:
    def __init__(self, G=1.0, state=None, bodies=None):
        self.G = G
        if state is not None:
            self.bodies = []
            for body in range(int(len(state) / 4)):
                n = int(body * 4)
                self.bodies.append(
                    Body(
                        mass=1.0,
                        position=[state[0 + n], state[1 + n]],
                        velocity=[state[2 + n], state[3 + n]],
                        body_id=body,
                    )
                )
        else:
            self.bodies = bodies

    def compute_accelerations(self):
        accelerations = []
        for body in self.bodies:
            other_bodies = [b for b in self.bodies if b is not body]
            accelerations.append(body.compute_acceleration(other_bodies, self.G))
        return accelerations

    def compute_total_energy(self):

        kinetic_energy = 0.5 * sum(
            body.mass * np.dot(body.velocity, body.velocity) for body in self.bodies
        )

        potential_energy = 0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i + 1 :]:
                distance = np.linalg.norm(body2.position - body1.position)
                potential_energy -= self.G * body1.mass * body2.mass / distance

        return kinetic_energy + potential_energy

    def get_state(self):
        return np.concatenate([body.get_state() for body in self.bodies])

    def integrate(self, dt, num_steps, save_positions=False):

        if save_positions:
            positions = np.zeros((num_steps, len(self.bodies), 2))
            delta_energy = np.zeros(num_steps)
            initial_energy = self.compute_total_energy()

        for step in range(num_steps):

            accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

            new_accelerations = self.compute_accelerations()

            for i, body in enumerate(self.bodies):
                body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt

            if save_positions:
                positions[step] = [body.position for body in self.bodies]
                delta_energy[step] = initial_energy - self.compute_total_energy()

        if save_positions:
            return positions, delta_energy
        else:
            return self.get_state()


def lyapunov(stan_state, total_time=20, delta_t=0.01, divisor=10):
    time = np.arange(0, total_time, delta_t * divisor)
    lyapunov_exponents = []

    stan_system = System(state=stan_state)
    perturb_state = stan_state + np.random.normal(0, 1e-10, stan_state.shape)
    perturb_system = System(state=perturb_state)

    distance_initial = np.linalg.norm(stan_state - perturb_state)

    for step, t in enumerate(time):
        stan_state = stan_system.integrate(delta_t, divisor)
        perturb_state = perturb_system.integrate(delta_t, divisor)

        distance_final = np.linalg.norm(stan_state - perturb_state)
        lyapunov_exponent = np.log(abs(distance_final / distance_initial))
        lyapunov_exponents.append(lyapunov_exponent)

        perturb_state = (
            stan_state
            + distance_initial * (perturb_state - stan_state) / distance_final
        )
        perturb_system = System(state=perturb_state)
        distance_initial = distance_final

    return np.sum(lyapunov_exponents) * (1 / total_time)


def proximity(stan_state, total_time=20, delta_t=0.01):
    time = np.arange(0, total_time, delta_t)
    initial_state = stan_state
    stan_system = System(state=stan_state)

    min_proximity = float("inf")
    di = 0

    for t in time:
        stan_state = stan_system.integrate(delta_t, 1)
        df = np.linalg.norm(stan_state - initial_state)
        if df < di and df < min_proximity:
            min_proximity = df
        di = df

    if min_proximity == float("inf"):
        min_proximity = df

    return min_proximity
