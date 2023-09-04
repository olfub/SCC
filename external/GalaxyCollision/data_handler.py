import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.markers import MarkerStyle

from external.GalaxyCollision.sources.galaxies import RingsMasslessGalaxy
from external.GalaxyCollision.sources.old.__galaxycollision import (
    Galaxy_Collision,
    initial_trajectory,
)


def get_state(merger):
    # given the state merger, return position and velocity of stars and black holes in one variable each
    X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc = [], [], [], [], [], [], [], []
    for j in range(merger.engine.n_massless):
        X.append(str(merger.engine.massless_X[j, 0]))
        Y.append(str(merger.engine.massless_X[j, 1]))
        Xp.append(str(merger.engine.massless_V[j, 0]))
        Yp.append(str(merger.engine.massless_V[j, 1]))
    for j in range(merger.engine.n_centers):
        Xc.append(str(merger.engine.center_X[j, 0]))
        Yc.append(str(merger.engine.center_X[j, 1]))
        Xpc.append(str(merger.engine.center_V[j, 0]))
        Ypc.append(str(merger.engine.center_V[j, 1]))
    return X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc


def state_to_single_array(state):
    # given different variables containing position and velocity information of stars and black holes, return the same
    # data in a two-dimensional numpy array
    X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc = state
    # one particle per row, in each row: particle info (4) + black hole info (4) for each black hole (len(Xp))
    data = np.zeros((len(X), len(Xc) * 4 + 4))
    for i in range(len(X)):
        data[i, 0] = X[i]
        data[i, 1] = Y[i]
        data[i, 2] = Xp[i]
        data[i, 3] = Yp[i]
        for j in range(len(Xc)):
            data[i, 4 + (j * 4)] = Xc[j]
            data[i, 5 + (j * 4)] = Yc[j]
            data[i, 6 + (j * 4)] = Xpc[j]
            data[i, 7 + (j * 4)] = Ypc[j]
    return data


def single_array_to_state(data):
    X = np.zeros(data.shape[0])
    Y = np.zeros(data.shape[0])
    Xp = np.zeros(data.shape[0])
    Yp = np.zeros(data.shape[0])
    Xc = np.zeros(int((data.shape[1] - 4) / 4))
    Yc = np.zeros(int((data.shape[1] - 4) / 4))
    Xpc = np.zeros(int((data.shape[1] - 4) / 4))
    Ypc = np.zeros(int((data.shape[1] - 4) / 4))
    for i in range(len(X)):
        X[i] = data[i, 0]
        Y[i] = data[i, 1]
        Xp[i] = data[i, 2]
        Yp[i] = data[i, 3]
    for j in range(len(Xc)):
        # it should not matter which row to take this information from, it should be identical everywhere (so 0 is used)
        Xc[j] = data[0, 4 + (j * 4)]
        Yc[j] = data[0, 5 + (j * 4)]
        Xpc[j] = data[0, 6 + (j * 4)]
        Ypc[j] = data[0, 7 + (j * 4)]
    return X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc


def set_state(merger, state):
    X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc = state
    for j in range(merger.engine.n_massless):
        merger.engine.massless_X[j, 0] = X[j]
        merger.engine.massless_X[j, 1] = Y[j]
        merger.engine.massless_V[j, 0] = Xp[j]
        merger.engine.massless_V[j, 1] = Yp[j]
    for j in range(merger.engine.n_centers):
        merger.engine.center_X[j, 0] = Xc[j]
        merger.engine.center_X[j, 1] = Yc[j]
        merger.engine.center_V[j, 0] = Xpc[j]
        merger.engine.center_V[j, 1] = Ypc[j]
    return X, Y, Xp, Yp, Xc, Yc, Xpc, Ypc


def sample_simulation(
    time_steps, method="Runge_Kutta", black_holes=2, seed=0, return_nr_particles=False
):
    # Run a simulation starting with a meaningful initialization (calculated using initial_trajectory) and run it for
    # the given amount of time_steps.
    random_state = np.random.RandomState(seed)

    state_len = 4 + 4 * black_holes

    # galaxy 2 initialization

    # periapsis: smallest distance to the center (relative, is multiplied on top of X)
    periapsis = (random_state.rand() * 5) + 3

    # eccentricity: shape of the orbit, between 0 and 1 (round: "0" or oval: "close to 1")
    eccentricity = random_state.rand()

    # true anomaly: position on the circle (0 is on the right of the center, 90 above,...), between 0 and 360
    true_anomlay = random_state.rand() * 360

    # M: mass (of galaxy 1)
    galaxy_1_mass = 10000
    galaxy_2_mass = 500
    galaxy_1_halo_radius = 5
    galaxy_2_halo_radius = 2.5

    X, V = initial_trajectory(periapsis, eccentricity, true_anomlay, galaxy_1_mass)

    # ring linspaces, particles per ring, black hole mass, halo radius, g1 state (position and velocity)

    # GALAXY 1 : Main Galaxy
    rings1 = np.linspace(0.5, 3, 20)
    particles1 = np.full_like(rings1, 40)
    gal1 = RingsMasslessGalaxy(rings1, particles1, galaxy_1_mass, galaxy_1_halo_radius)
    gal1.initialState([0, 0], [0, 0])

    # GALAXY 2 : Dwarf Galaxy Incoming
    # We send the dwarf galaxy on a collision course along an elliptic conic
    rings2 = np.linspace(0.1, 1, 15)
    particles2 = np.full_like(rings2, 20)
    gal2 = RingsMasslessGalaxy(rings2, particles2, galaxy_2_mass, galaxy_2_halo_radius)
    gal2.initialState(X, V)

    # SIMULATION
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merger = Galaxy_Collision([gal1, gal2], current_dir)

    nr_particles = state_to_single_array(get_state(merger)).shape[0]
    data = np.zeros((nr_particles * time_steps, state_len * 2))

    # initial time step
    data_0 = state_to_single_array(get_state(merger))
    # this is the first observation ("left" half of array, one state until "state_len")
    data[0:nr_particles, :state_len] = data_0
    for i in range(time_steps):
        merger.engine.compute(0.1, method=method)
        data_i = state_to_single_array(get_state(merger))
        # here, this is the next time step for the previous input ("right" half of array)
        data[i * nr_particles : (i + 1) * nr_particles, state_len:] = data_i
        if i != time_steps - 1:
            # the "right" side of the array in the lines above are now the new input ("left" side) of the next time step
            # for all but the last time steps at least (the last time step will not have another step after)
            data[(i + 1) * nr_particles : (i + 2) * nr_particles, :state_len] = data_i
    if return_nr_particles:
        return data, nr_particles
    return data


def continue_simulation(
    time_steps,
    method="Runge_Kutta",
    black_holes=2,
    seed=0,
    return_nr_particles=False,
    initial_state=None,
):
    # Run a simulation starting with a meaningful initialization (calculated using initial_trajectory) and run it for
    # the given amount of time_steps.
    random_state = np.random.RandomState(seed)

    state_len = 4 + 4 * black_holes

    # galaxy 2 initialization

    # periapsis: smallest distance to the center (relative, is multiplied on top of X)
    periapsis = (random_state.rand() * 5) + 3

    # eccentricity: shape of the orbit, between 0 and 1 (round: "0" or oval: "close to 1")
    eccentricity = random_state.rand()

    # true anomaly: position on the circle (0 is on the right of the center, 90 above,...), between 0 and 360
    true_anomlay = random_state.rand() * 360

    # M: mass (of galaxy 1)
    galaxy_1_mass = 10000
    galaxy_2_mass = 500
    galaxy_1_halo_radius = 5
    galaxy_2_halo_radius = 2.5

    X, V = initial_trajectory(periapsis, eccentricity, true_anomlay, galaxy_1_mass)

    # ring linspaces, particles per ring, black hole mass, halo radius, g1 state (position and velocity)

    # GALAXY 1 : Main Galaxy
    rings1 = np.linspace(0.5, 3, 20)
    particles1 = np.full_like(rings1, 40)
    gal1 = RingsMasslessGalaxy(rings1, particles1, galaxy_1_mass, galaxy_1_halo_radius)
    gal1.initialState([0, 0], [0, 0])

    # GALAXY 2 : Dwarf Galaxy Incoming
    # We send the dwarf galaxy on a collision course along an elliptic conic
    rings2 = np.linspace(0.1, 1, 15)
    particles2 = np.full_like(rings2, 20)
    gal2 = RingsMasslessGalaxy(rings2, particles2, galaxy_2_mass, galaxy_2_halo_radius)
    gal2.initialState(X, V)

    # SIMULATION
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merger = Galaxy_Collision([gal1, gal2], current_dir)

    nr_particles = state_to_single_array(get_state(merger)).shape[0]
    data = np.zeros((nr_particles * time_steps, state_len * 2))

    if initial_state is not None:
        set_state(merger, single_array_to_state(initial_state))

    # initial time step
    data_0 = state_to_single_array(get_state(merger))
    # this is the first observation ("left" half of array, one state until "state_len")
    data[0:nr_particles, :state_len] = data_0
    for i in range(time_steps):
        merger.engine.compute(0.1, method=method)
        data_i = state_to_single_array(get_state(merger))
        # here, this is the next time step for the previous input ("right" half of array)
        data[i * nr_particles : (i + 1) * nr_particles, state_len:] = data_i
        if i != time_steps - 1:
            # the "right" side of the array in the lines above are now the new input ("left" side) of the next time step
            # for all but the last time steps at least (the last time step will not have another step after)
            data[(i + 1) * nr_particles : (i + 2) * nr_particles, :state_len] = data_i
    if return_nr_particles:
        return data, nr_particles
    return data


def visualize_sample(data, output_dir, highlight_particles=None):
    # Create a gif to visualize a galaxy example
    # data needs to be a list, one element for each time step
    # each list entry (time step): 4 arrays: galaxies x, galaxies y, stars x, stars y
    data = data[:80]
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("xkcd:black")  # Changing figure to black
    ax = fig.add_subplot(111)
    ax.set_facecolor("xkcd:black")  # Changing background to black
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    galaxies_scatter = ax.scatter([], [], c="white", s=30)
    stars_scatter = ax.scatter([], [], c="yellow", s=5)
    stars_highlight = ax.scatter([], [], c="magenta", s=35)

    def update(i):
        # given the data, draw the current frame
        galaxies_x = data[i][0]
        galaxies_y = data[i][1]
        galaxies = np.hstack((galaxies_x[:, np.newaxis], galaxies_y[:, np.newaxis]))
        stars_x = data[i][2]
        stars_y = data[i][3]
        if highlight_particles is not None:
            stars_x_highlight = stars_x[highlight_particles]
            stars_y_highlight = stars_y[highlight_particles]
            highlights = np.hstack(
                (stars_x_highlight[:, np.newaxis], stars_y_highlight[:, np.newaxis])
            )
            stars_highlight.set_offsets(highlights)
            stars_x = np.delete(stars_x, highlight_particles)
            stars_y = np.delete(stars_y, highlight_particles)
            path = None
            if i == 1:
                stars_highlight.set_sizes(np.ones(5) * 15)
            if i == 49:
                marker = MarkerStyle("*")
                stars_highlight.set_paths((marker.get_path(),))
                stars_highlight.set_sizes(np.ones(5) * 30)
        stars = np.hstack((stars_x[:, np.newaxis], stars_y[:, np.newaxis]))
        galaxies_scatter.set_offsets(galaxies)
        stars_scatter.set_offsets(stars)

    simulation = FuncAnimation(fig, update, len(data), interval=100, repeat=False)

    if output_dir is not None:
        animation_writer = animation.writers["pillow"]
        writer = animation_writer(fps=5, bitrate=8000)
        output_dir.mkdir(exist_ok=True, parents=True)
        j = 0
        while os.path.exists(output_dir / f"simulation_gc_{j}.gif"):
            j += 1
        simulation.save(output_dir / f"simulation_gc_{j}.gif", writer=writer)
        return output_dir / f"simulation_gc_{j}.gif"
    else:
        plt.show(block=True)


def sample_data_random(
    rng,
    nr_samples,
    intervention_index=-1,
    intervention_value=0,
    priors=(True, True, True, True),
    method="Runge_Kutta",
    dt=0.1,
    galaxy_params=None,
):
    # a simple data generation approach which covers a wide range of scenarios (at the cost of covering a lot of
    # unrealistic (for example particle velocity going away from a black hole) scenarios, but maybe this still makes the
    # resulting function easier to learn, especially considering interventions)

    # this code is written for a simulation with exactly two black holes

    # if intervention_index == -1, then no intervention takes place, otherwise the value indexed by that is set to
    # the intervention_value

    # priors: whether to apply priors or (if False) generate data fully uniformly in an interval
    # priors for: 1. first black hole position, 2. black hole velocities, 3. star position, 4. star velocities
    # ...which are: 1. gaussian around middle, 2. gaussian around 0, 3. gaussian around black hole, 4. gaussian around 0

    # galaxy_params: several general problem parameters, see the code at the start of this function for more information

    # the default set of galaxy_params is designed to result in meaningful values; one must be careful when setting the
    # values manually as unrealistic configurations might occur (for example, if the mass of the first black hole is
    # very small but its standard deviation of the velocity is large, the generated data would not represent data from
    # actual simulations with this configuration well)
    if galaxy_params is None:
        # this function is written for a square box (boundaries), values apply to both dimensions
        bb = [-10, 10]  # bb : box boundaries

        # galaxy masses and halo radii
        galaxy_1_mass = 10000
        galaxy_2_mass = 500
        galaxy_1_halo_radius = 5
        galaxy_2_halo_radius = 2.5

        # looking at data which generates simulation data without interventions, these maximum velocities seem
        # appropriate given the masses of the black holes
        mv = [2, 30, 20]  # maximum velocity black hole 1, black hole 2, particles

        # standard deviations for the data generation
        black_hole_1_pos_std = 3
        black_hole_1_velocity_std = 1
        black_hole_2_velocity_std = 20
        star_position_std_value = 4
        star_velocity_std = 10
    else:
        galaxy_1_mass = galaxy_params[0]
        galaxy_2_mass = galaxy_params[1]
        galaxy_1_halo_radius = galaxy_params[2]
        galaxy_2_halo_radius = galaxy_params[3]
        bb = galaxy_params[4]
        mv = galaxy_params[5]
        black_hole_1_pos_std = galaxy_params[6]
        black_hole_1_velocity_std = galaxy_params[7]
        black_hole_2_velocity_std = galaxy_params[8]
        star_position_std_value = galaxy_params[9]
        star_velocity_std = galaxy_params[10]

    # black hole 1 position
    if priors[0]:
        position_std = black_hole_1_pos_std
        black_hole_1_xy = rng.normal(0, position_std, size=(nr_samples, 2))
        outside_box = np.absolute(black_hole_1_xy) > bb[1]
        # bound the values to the box
        while True in outside_box:
            indices = np.where(outside_box)
            black_hole_1_xy[indices] = rng.normal(
                0, position_std, size=(nr_samples, 2)
            )[indices]
            outside_box = np.absolute(black_hole_1_xy) > bb[1]
    else:
        black_hole_1_xy = rng.uniform(bb[0], bb[1], size=(nr_samples, 2))

    # black hole 2 position
    # I think having the second one uniform is better than with a prior
    black_hole_2_xy = rng.uniform(bb[0], bb[1], size=(nr_samples, 2))

    # black holes velocity
    if priors[1]:
        bh1_v_std = black_hole_1_velocity_std
        bh2_v_std = black_hole_2_velocity_std
        black_hole_1_vxvy = rng.normal(0, bh1_v_std, size=(nr_samples, 2))
        black_hole_2_vxvy = rng.normal(0, bh2_v_std, size=(nr_samples, 2))
    else:
        black_hole_1_vxvy = rng.uniform(-mv[0], mv[0], size=(nr_samples, 2))
        black_hole_2_vxvy = rng.uniform(-mv[1], mv[1], size=(nr_samples, 2))

    # star position
    if priors[2]:
        # randomly decide to which black holes the stars should "belong" to (matters because of priors)
        which_bh = rng.random(nr_samples) < 0.5
        star_position_mean = np.zeros_like(black_hole_1_xy)
        star_position_mean[which_bh] = black_hole_1_xy[which_bh]
        star_position_mean[which_bh == False] = black_hole_2_xy[which_bh == False]
        # large std so that the entire box can be filled
        star_position_std = np.full_like(star_position_mean, star_position_std_value)

        star_xy = rng.normal(
            star_position_mean, star_position_std, size=(nr_samples, 2)
        )
        # allowing the stars to be 10% outside the box, so that behavior at the box edges is also learned
        outside_box = np.absolute(star_xy) > bb[1] * 1.1
        # bound the values to the box
        while True in outside_box:
            indices = np.where(outside_box)
            star_xy[indices] = rng.normal(
                star_position_mean, star_position_std, size=(nr_samples, 2)
            )[indices]
            outside_box = np.absolute(star_xy) > bb[1] * 1.1
    else:
        star_xy = rng.uniform(bb[0], bb[1], size=(nr_samples, 2))

    # star velocity
    if priors[3]:
        star_v_std = star_velocity_std
        star_vxvy = rng.normal(0, star_v_std, size=(nr_samples, 2))
    else:
        star_vxvy = rng.uniform(-mv[2], mv[2], size=(nr_samples, 2))

    data = np.concatenate(
        (
            star_xy,
            star_vxvy,
            black_hole_1_xy,
            black_hole_1_vxvy,
            black_hole_2_xy,
            black_hole_2_vxvy,
        ),
        axis=1,
    )

    # calculate the next time step

    # GALAXY 1 : Main Galaxy
    gal1 = RingsMasslessGalaxy(
        np.array([1]), np.array([1]), galaxy_1_mass, galaxy_1_halo_radius
    )

    # GALAXY 2 : Dwarf Galaxy
    gal2 = RingsMasslessGalaxy(
        np.array([1]), np.array([1]), galaxy_2_mass, galaxy_2_halo_radius
    )

    # just placeholder, will be changed
    gal1.initialState([0, 0], [0, 0])
    gal2.initialState([0, 0], [0, 0])

    # setup merger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merger = Galaxy_Collision([gal1, gal2], current_dir)

    # create the data to return at the end of this method and fill the "left" half (inputs for the model)
    all_data = np.zeros((nr_samples, 24))
    all_data[:, :12] = data

    if intervention_index != -1:
        # intervene on the data used for the next time step
        # this is not given in the observation data (the intervention value will be a separate value anyway)
        # this also implies that the intervention takes place before the calculation of the next time step and can
        # influence this calculation
        data[:, intervention_index] = intervention_value

    for j in range(nr_samples):
        # set the values in the engine to return the correct next time step with compute
        merger.engine.center_X = np.array([data[j, 4:6], data[j, 8:10]])
        merger.engine.center_V = np.array([data[j, 6:8], data[j, 10:12]])
        merger.engine.massless_X = np.array([data[j, 0:2], data[j, 0:2]])
        merger.engine.massless_V = np.array([data[j, 2:4], data[j, 2:4]])
        merger.engine.compute(dt, method=method)

        # extract the next time step
        data_next_step = state_to_single_array(get_state(merger))
        # have to use one particle per galaxy for the code to work, but both are identical, just use the first one: [0]
        data_next_step = data_next_step[0]
        # the next time step is the expected prediction for the model ("right" half of data)
        all_data[j, 12:] = data_next_step

    return all_data
