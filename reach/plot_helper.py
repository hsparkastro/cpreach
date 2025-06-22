import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from . import alphashape
from time import time


def plot_triangles(ax, x, y, theta, size=1.0, color='r', alpha=1.0,
                   label='_', **kwargs):
    """
    Overlay equilateral triangles on a given plot.

    Parameters:
    ax : matplotlib.axes.Axes
        The axis handle on which to plot the triangles.
    x : array-like
        The x-coordinates of the triangle centers.
    y : array-like
        The y-coordinates of the triangle centers.
    theta : array-like
        The orientations (in radians) of the triangles.
    size : float, optional
        The size of the triangles (height). Default is 1.0.
    color : str, optional
        The color of the triangles. Default is 'b' (blue).
    """
    # Compute half-width and height
    half_width = size / 4
    height = size / 2

    # Define the reference triangle (pointing in +x direction at theta=0)
    base_triangle = np.array([
        [height, 0],          # Top vertex
        [-height / 2, -half_width],  # Bottom-left vertex
        [-height / 2, half_width]    # Bottom-right vertex
    ])

    for xi, yi, thetai in zip(x, y, theta):
        # Create rotation matrix
        R = np.array([
            [np.cos(thetai), -np.sin(thetai)],
            [np.sin(thetai), np.cos(thetai)]
        ])

        # Rotate and translate the triangle
        rotated_triangle = (R @ base_triangle.T).T + [xi, yi]

        # Create and add the patch
        triangle_patch = patches.Polygon(
            rotated_triangle, closed=True, color=color, alpha=alpha,
            label=label, **kwargs)
        ax.add_patch(triangle_patch)


def plot_rectangle(ax, pose, dimensions, **kwargs):
    """
    Draw a rotated rectangle at position (x, y) with a given width, height,
    rotation angle (theta),
    and customizable color and alpha value.

    Parameters:
    - x, y: Center position of the rectangle.
    - width, height: Dimensions of the rectangle.
    - theta: Rotation angle in degrees.
    - color: Color of the rectangle.
    - alpha: Alpha transparency of the rectangle.
    """

    x, y, theta = pose
    width, height = dimensions[0], dimensions[1]

    # Convert theta from degrees to radians
    theta_deg = np.rad2deg(theta)

    # Define the initial (non-rotated) bottom-left corner relative to the
    # center (x, y)
    half_width = width / 2
    half_height = height / 2
    # corner_x = x - half_width
    # corner_y = y - half_height

    # Apply rotation to the bottom-left corner
    rotated_x = (- half_width) * np.cos(theta) - \
        (- half_height) * np.sin(theta) + x
    rotated_y = (- half_width) * np.sin(theta) + \
        (- half_height) * np.cos(theta) + y

    # Create a rotated rectangle
    rect = patches.Rectangle(
        (rotated_x, rotated_y),  # bottom-left corner of the rectangle
        width, height,                    # width and height
        angle=theta_deg,                      # rotation angle
        # linewidth=2,                      # rectangle border width
        # edgecolor=color,                # rectangle border color
        **kwargs
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)


def generate_rectangle_points(xytheta_list, num_points_per_edge=11):
    width, length = 5.0, 2.5

    # Define the local (unrotated) rectangle corners
    half_w, half_l = width / 2, length / 2
    corners = np.array([
        [-half_w, -half_l],
        [half_w, -half_l],
        [half_w, half_l],
        [-half_w, half_l],
        [-half_w, -half_l]
    ])

    # Generate points along edges
    edge_points = np.vstack([
        np.linspace(corners[i], corners[i+1], num_points_per_edge)[:-1]
        for i in range(4)
    ])

    xytheta_array = np.array(xytheta_list)
    x_vals, y_vals, theta_vals = xytheta_array.T

    # Compute rotation matrices
    cos_theta, sin_theta = np.cos(theta_vals), np.sin(theta_vals)
    rotation_matrices = np.array([
        [[c, -s], [s, c]] for c, s in zip(cos_theta, sin_theta)
    ])

    # Apply rotation and translation to all points
    rotated_points = np.einsum('ijk,nk->inj', rotation_matrices, edge_points)
    rotated_points += xytheta_array[:, None, :2]

    return rotated_points.reshape(-1, 2)


def sample2poly(points, alpha=None, occupancy=True):

    if occupancy:
        if alpha is None:
            points = generate_rectangle_points(points, num_points_per_edge=2)
        else:
            points = generate_rectangle_points(points, num_points_per_edge=11)
    if alpha is None:
        hull = ConvexHull(points)
        poly = np.array([points[hull.vertices, 0], points[hull.vertices, 1]])
    else:
        t0 = time()
        alpha_shape, convex_pieces = alphashape.alphashape_convex_decomposition(
            points, alpha)
        print(f"alphashape took {time() - t0} seconds")
        poly = np.array(alpha_shape.exterior.xy)

        # alpha_shape = alphashape.alphashape(points, alpha)
        # poly = np.array(alpha_shape.exterior.xy)

    return poly, convex_pieces


def plot_frs(
    ax, vertices, edgecolor='black', facecolor='red',
    label='_', **kwargs


):
    ax.add_patch(patches.Polygon(vertices, edgecolor=edgecolor,
                 facecolor=facecolor, label=label, **kwargs))


# Example usage
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    x = np.array([0, 1, -1])
    y = np.array([0, 1, -1])
    theta = np.array([0, np.pi/4, -np.pi/4])

    plot_triangles(ax, x, y, theta, size=1.0, color='r')

    plt.show()
