import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from external.particles_in_a_box import utils


def sample_data(
    seed,
    n_frames=1,
    intervention_info=None,
    n_particles=None,
    box_size=None,
    radii_limits=None,
    restitution_coef_bc=None,
    restitution_coef_pc=None,
    acceleration_vector=None,
    continue_data=None,
):
    # continue_data: should either be None, then it does nothing, or contain a list of n_particles elements with 4
    # values each, representing x, y, vx, vy; if this is given, these values will overwrite the initialization of the
    # particles which (if the other parameters are correct) results in a continuation of the given sample

    # intervention_info:
    # - int_time: when to apply an intervention (int)
    # - int_part: the particle to be intervened on (int)
    # - int_attr: the attribute to be intervened on (either one of ["x", "y", "vx", "vy"] or int: [0, 1, 2, 3])
    # - int_value: the intervention value (the value, the variable is set to) (float)
    if intervention_info is None:
        int_time = -1
        int_part, int_attr, int_value = None, None, None
    else:
        (int_time, int_part, int_attr, int_value) = intervention_info
    if n_particles is None:
        n_particles = 3
    if box_size is None:
        box_size_x = 6
        box_size_y = 9
    else:
        box_size_x = box_size[0]
        box_size_y = box_size[1]
    if radii_limits is None:
        radii_limits = np.array([0.5, 0.5])
    if restitution_coef_bc is None:
        restitution_coef_bc = 1.0
    if restitution_coef_pc is None:
        restitution_coef_pc = 1.0
    if acceleration_vector is None:
        acceleration_vector = [0.0, -70.0]

    position_limits_x = np.array(
        [-box_size_x / 2 + box_size_x / 10, box_size_x / 2 - box_size_x / 10]
    )
    position_limits_y = np.array(
        [-box_size_y / 2 + box_size_y / 10, box_size_y / 2 - box_size_y / 10]
    )
    velocity_limits_x = position_limits_x / 5
    velocity_limits_y = position_limits_y / 5

    np.random.seed(seed)

    x_init, y_init, v_x_init, v_y_init, radii = utils.get_init_conditions(
        n_particles=n_particles,
        position_limits_x=position_limits_x,
        position_limits_y=position_limits_y,
        velocity_limits_x=velocity_limits_x,
        velocity_limits_y=velocity_limits_y,
        radii_limits=radii_limits,
    )

    particles = utils.Particles(
        x_init, y_init, v_x_init, v_y_init, radii, vector_of_masses=np.pi * radii**2
    )

    if continue_data is not None:
        if len(continue_data) != n_particles or any(
            len(continue_data[i]) != 4 for i in range(n_particles)
        ):
            raise RuntimeError(
                f"continue_data should contain {n_particles} lists containing four values each."
            )
        for i in range(n_particles):
            particles.x[i] = continue_data[i][0]
            particles.y[i] = continue_data[i][1]
            particles.v_x[i] = continue_data[i][2]
            particles.v_y[i] = continue_data[i][3]

    data = np.zeros((n_frames + 1, n_particles * 4))

    def particles_vector(parts, n_parts):
        vecs = []
        for i in range(n_parts):
            vecs.append(np.array([parts.x[i], parts.y[i], parts.v_x[i], parts.v_y[i]]))
        return np.concatenate(vecs)

    for frame in range(n_frames):
        data[frame] = particles_vector(particles, n_particles)
        if int_time == frame:
            if int_attr == "x" or int_attr == 0:
                particles.x[int_part] = int_value
            elif int_attr == "y" or int_attr == 1:
                particles.y[int_part] = int_value
            elif int_attr == "vx" or int_attr == 2:
                particles.v_x[int_part] = int_value
            elif int_attr == "vy" or int_attr == 3:
                particles.v_y[int_part] = int_value
            else:
                raise TypeError(f"Invalid intervention type {int_attr}")
        utils.simulate(
            frame,
            particles,
            acceleration_vector,
            None,
            box_size_x,
            box_size_y,
            restitution_coef_bc,
            restitution_coef_pc,
        )

    data[-1] = particles_vector(particles, n_particles)
    return data


def visualize_sample(
    seed, data, n_particles=None, box_size=None, radii_limits=None, output_dir=None
):
    if n_particles is None:
        n_particles = 3
    if box_size is None:
        box_size_x = 6
        box_size_y = 9
    else:
        box_size_x = box_size[0]
        box_size_y = box_size[1]
    if radii_limits is None:
        radii_limits = np.array([0.5, 0.5])

    position_limits_x = np.array(
        [-box_size_x / 2 + box_size_x / 10, box_size_x / 2 - box_size_x / 10]
    )
    position_limits_y = np.array(
        [-box_size_y / 2 + box_size_y / 10, box_size_y / 2 - box_size_y / 10]
    )
    velocity_limits_x = position_limits_x / 5
    velocity_limits_y = position_limits_y / 5

    np.random.seed(seed)

    x_init, y_init, v_x_init, v_y_init, radii = utils.get_init_conditions(
        n_particles=n_particles,
        position_limits_x=position_limits_x,
        position_limits_y=position_limits_y,
        velocity_limits_x=velocity_limits_x,
        velocity_limits_y=velocity_limits_y,
        radii_limits=radii_limits,
    )

    particles = utils.Particles(
        x_init, y_init, v_x_init, v_y_init, radii, vector_of_masses=np.pi * radii**2
    )

    fig, ax1 = plt.subplots(figsize=(box_size_x / 5, box_size_y / 5))

    def simulate(fid, part_data, parts, ax):
        # use parts from init parts, but other values from data
        current_part_data = part_data[fid]
        n_parts = len(parts.x)
        n_values = int(len(current_part_data) / n_parts)
        for i in range(n_parts):
            parts.x[i] = current_part_data[i * n_values]
            parts.y[i] = current_part_data[i * n_values + 1]
            parts.v_x[i] = current_part_data[i * n_values + 2]
            parts.v_y[i] = current_part_data[i * n_values + 3]

        ax.clear()
        circles = [
            plt.Circle((x_i, y_i), radius=r)
            for x_i, y_i, r in zip(parts.x, parts.y, parts.radii)
        ]
        collection = matplotlib.collections.PatchCollection(
            circles, cmap=matplotlib.cm.jet, alpha=0.8
        )
        collection.set_edgecolors("k")
        collection.set_linewidth(1)
        collection.set_array(np.sqrt(parts.v_x**2 + parts.v_y**2))
        collection.set_clim([0, 50])
        ax.add_collection(collection)
        ax.axis("equal")
        ax.set_xlim([-box_size_x / 2, box_size_x / 2])
        ax.set_ylim([-box_size_y / 2, box_size_y / 2])
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    simulation = FuncAnimation(
        fig,
        simulate,
        fargs=(data, particles, ax1),
        frames=len(data),
        interval=1,
        repeat=False,
    )
    if output_dir is not None:
        animation_writer = animation.writers["pillow"]
        writer = animation_writer(fps=60, bitrate=8000)
        output_dir.mkdir(exist_ok=True, parents=True)
        i = 0
        while os.path.exists(output_dir / f"simulation_acc_comp_{i}.gif"):
            i += 1
        simulation.save(output_dir / f"simulation_acc_comp_{i}.gif", writer=writer)
        return output_dir / f"simulation_acc_comp_{i}.gif"
    else:
        plt.show()
