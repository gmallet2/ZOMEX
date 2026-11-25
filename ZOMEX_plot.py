from matplotlib import pyplot as plt
import numpy as np

class PlotPlanet() : 
    @classmethod
    def plot_planet(_,ax,phi, delta, alpha):
        planet = plt.Circle((0, 0), 1,color="lightgrey",alpha=0.35)
        ax.add_patch(planet)
        #For terminator
        P_star = -np.array([np.cos(phi),np.sin(phi),0.0])
        # Le terminateur vérifie P . P_star = 0.
        u = np.array([-P_star[1],P_star[0],0.0])
        v = np.array([0.0, 0.0, 1.0])
        t = np.linspace(0, 2*np.pi, 2000)
        terminator = (u[:, None]*np.cos(t)+v[:, None]*np.sin(t))
        visible = terminator[0] >= 0
        ax.plot(
            np.where(visible, terminator[1], np.nan),
            np.where(visible, terminator[2], np.nan),
            color="blue",
            lw=1,
            label="Terminator"
        )
        P_HS = np.array([np.cos(np.pi+phi+delta),np.sin(np.pi+phi+delta),0.0])
        # Deux vecteurs orthonormés perpendiculaires
        # au centre du Hot Spot.
        u = np.array([-P_HS[1], P_HS[0],0.0])
        v = np.array([0.0, 0.0, 1.0])
        # Frontière de la calotte :
        # P . P_HS = cos(alpha)
        hotspot = (np.cos(alpha)*P_HS[:, None]+ np.sin(alpha)*(u[:, None]*np.cos(t)+ v[:, None]*np.sin(t)))
        visible = hotspot[0] >= 0
        ax.plot(np.where(visible, hotspot[1], np.nan),
            np.where(visible, hotspot[2], np.nan),
            color="red",lw=2.5,label="Hot Spot boundary")

        if P_HS[0] >= 0: ax.scatter(P_HS[1], P_HS[2],color="red",marker="x",s=70,label="Hot Spot center")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("y (right)")
        ax.set_ylabel("z (up)")
        ax.set_title("Visible hemisphere of the planet")
        ax.legend()
        ax.grid(alpha=0.25)