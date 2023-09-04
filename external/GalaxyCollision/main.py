from sources.engines import *
from sources.galaxies import *
from sources.old.__galaxycollision import *
from sources.simulators import *


def main():
    """Main program"""
    # GALAXY 1 : Main Galaxy
    rings1 = np.linspace(0.5, 3, 20)
    particles1 = np.full_like(rings1, 40)
    gal1 = RingsMasslessGalaxy(rings1, particles1, 10000, 5)
    gal1.initialState([0, 0], [0, 0])

    # GALAXY 2 : Dwarf Galaxy Incoming
    X, V = initial_trajectory(3, 0.6, -135, gal1.centralMass)
    # We send the dwarf galaxy on a collision course along an elliptic conic
    rings2 = np.linspace(0.1, 1, 15)
    particles2 = np.full_like(rings2, 20)
    gal2 = RingsMasslessGalaxy(rings2, particles2, 500, 2.5)
    gal2.initialState(X, V)

    # SIMULATION
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merger = Galaxy_Collision([gal1, gal2], current_dir)
    merger.RUN(0.01, 6, method="Runge_Kutta")  # We run some calculations
    merger.display(
        gif_fps=25, gif_duration=4
    )  # We create a GIF from the showcased Matplotlib animation


if __name__ == "__main__":
    main()
