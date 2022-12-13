"""
    hyperbolic/plot.py
    Nov 8 2022
    Gabriel Moreira
"""

import numpy as np
import matplotlib.pyplot as plt
from poincare import mobius_inverse, mobius_transform

class PoincareSegment:
    def __init__(self, p, q):
        """
            Geodesic segment in Poincare 2D disk model

            Parameters 
            ----------
            p - start point
            q - end point
        """
        self.p = p
        self.q = q
        self.q_transformed = mobius_transform(self.p)(self.q)
        
    def points(self, n):
        inverted_pts = np.array([self.q_transformed * i/n for i in range(0,n+1)])
        pts = mobius_inverse(self.p)(inverted_pts)
        return pts

    def draw(self, ax, n=100, color=[.8,.8,.8], linewidth=.5):
        """
        """
        pts = self.points(n)
        ax.plot(np.real(pts), np.imag(pts), '-', linewidth=linewidth, color=color, zorder=1)


class HyperboloidSegment:
    def __init__(self, p, q):
        """
            Geodesic segment in Hyperboloid in 3-Minkowski space

            Parameters 
            ----------
            p - start point
            q - end point
        """
        self.p = p
        self.q = q

        self.v = (self.q - self.p * np.cosh(1.0)) / np.sinh(1.0)

    def draw(self, ax, color=[.8,.8,.8], linewidth=.5):
        """
        """
        step = 0.01
        t = np.arange(0, 1.0 + step, step)
        pts = self.p * np.cosh(t[:,np.newaxis]) + self.v * np.sinh(t[:,np.newaxis])
        ax.plot3D(pts[:,0], pts[:,1], pts[:,2], '-', linewidth=linewidth, color=color, zorder=1)



def hyperboloid_graph_draw(graph,
                           embeddings,
                           ax,
                           node_color=[1.0, 1.0, 1.0, 1.0],
                           node_size=40,
                           edge_color=[0.0, 0.0, 0.0],
                           edge_width=0.3):
    """
        Draws Networkx graph in hyperboloid model of the hyperbolic plane
    """
    for e in graph.edges():
        segment = HyperboloidSegment(embeddings[e[0]], embeddings[e[1]])
        segment.draw(ax, edge_color, edge_width)
        
    ax.scatter3D([pos[0] for _, pos in embeddings.items()],
                 [pos[1] for _, pos in embeddings.items()],
                 [pos[2] for _, pos in embeddings.items()],
                  color=node_color, s=node_size, edgecolors='black', linewidths=.7, zorder=2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False



def poincare_graph_draw(graph,
                        root_id,
                        embeddings,
                        ax,
                        circle_fill     = False,
                        circle_color    = [0.0, 0.0, 0.0],
                        node_fill_color = [1.0, 1.0, 1.0, 1.0],
                        node_edge_color = [0.0, 0.0, 0.0],
                        node_size       = 100,
                        node_linewidth  = 0.5,
                        beta            = 0.9,
                        edge_color      = [0.0, 0.0, 0.0],
                        edge_linewidth  = 0.3):
    """
        Draws Networkx graph in Poincaré 2D disk model of the hyperbolic plane
    """
    circle = plt.Circle((0, 0), 1.0, color=circle_color, fill=circle_fill, linewidth=0.7)
    ax.add_patch(circle)

    if root_id is not None:
        transform  = mobius_transform(complex(embeddings[root_id][0], embeddings[root_id][1]))
        embeddings = {n : np.array([np.real(transform(complex(embeddings[n][0], embeddings[n][1]))),
                                    np.imag(transform(complex(embeddings[n][0], embeddings[n][1])))]) for n in graph.nodes()}

    if edge_linewidth > 0:
        for e in graph.edges():
            segment = PoincareSegment(complex(embeddings[e[0]][0], embeddings[e[0]][1]),
                                      complex(embeddings[e[1]][0], embeddings[e[1]][1]))
            segment.draw(ax, 100, edge_color, edge_linewidth)
        
    size = [node_size * (1-beta*np.linalg.norm(pos)) for _, pos in embeddings.items()]

    plt.scatter([pos[0] for _, pos in embeddings.items()],
                [pos[1] for _, pos in embeddings.items()],
                 color=node_fill_color, s=size, edgecolors=node_edge_color, linewidths=node_linewidth, zorder=2)

    ax.set_xlim([-1.05,1.05])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.locator_params(axis='x', nbins=4)
    ax.set_aspect('equal')


def poincare_figure(figsize      = (14,5),
                    circle_fill  = False,
                    circle_color = [0.0, 0.0, 0.0, 1.0],
                    circle_radius = 1):

    fig = plt.figure(figsize=(14,5))
    ax = fig.add_subplot(1, 1, 1)
    circle = plt.Circle((0, 0), circle_radius, color=circle_color, fill=circle_fill, linewidth=0.7)
    ax.add_patch(circle)
    ax.set_xlim([-circle_radius+ 0.05, circle_radius+ 0.05])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.locator_params(axis='x', nbins=4)
    ax.set_aspect('equal')

    return fig, ax


def poincare_scatter(pos,
                     ax,
                     circle_fill     = False,
                     circle_color    = [0.0, 0.0, 0.0, 0.0],
                     node_fill_color = [1.0, 1.0, 1.0, 1.0],
                     node_edge_color = [0.0, 0.0, 0.0],
                     node_size       = 40,
                     node_linewidth  = 0.5,
                     beta            = 0.9):
    """
    """
    assert pos.shape[1] == 2

    circle = plt.Circle((0, 0), 1.0, color=circle_color, fill=circle_fill, linewidth=0.7)
    ax.add_patch(circle)

    size = [node_size * (1-beta*np.linalg.norm(pos[i,:])) for i in range(pos.shape[0])]

    plt.scatter(pos[:,0],
                pos[:,1],
                color=node_fill_color, s=size, edgecolors=node_edge_color, linewidths=node_linewidth, zorder=2)

    ax.set_xlim([-1.05,1.05])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.locator_params(axis='x', nbins=4)
    ax.set_aspect('equal')


def poincare_plot(pos,
                  ax,
                  circle_fill   = False,
                  marker        = '-',
                  circle_color  = [0.0, 0.0, 0.0, 0.0],
                  color         = [1.0, 1.0, 1.0, 1.0],
                  linewidth     = 0.5):
    """
    """
    assert pos.shape[1] == 2

    circle = plt.Circle((0, 0), 1.0, color=circle_color, fill=circle_fill, linewidth=0.7)
    ax.add_patch(circle)

    plt.plot(pos[:,0],
             pos[:,1],
             marker, color=color, linewidths=linewidth, zorder=2)

    ax.set_xlim([-1.05,1.05])
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.locator_params(axis='x', nbins=4)
    ax.set_aspect('equal')


def hyperboloid_plot(pos,
                     ax,
                     node_fill_color=[1.0, 1.0, 1.0, 1.0],
                     node_size=40,
                     edge_color=[0.0, 0.0, 0.0],
                     edge_width=0.3):
    """
    """
    assert pos.shape[1] == 3

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],
                 color=node_fill_color, s=node_size, edgecolors=edge_color, linewidths=.7) 
