"""
Tools for working with alpha shapes.
"""
__all__ = ['alphashape']

import logging
import numpy as np
from typing import Union, Tuple, List
from itertools import combinations
from shapely.geometry import MultiPoint, MultiLineString, Polygon
from shapely.ops import unary_union, polygonize
from shapely.geometry.polygon import orient
from scipy.spatial import Delaunay


def circumcenter(points: Union[List[Tuple[float]], np.ndarray]) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
    """
    points = np.asarray(points)
    num_rows, num_columns = points.shape
    A = np.bmat([[2 * np.dot(points, points.T),
                  np.ones((num_rows, 1))],
                 [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1),
                   np.ones((1))))
    return np.linalg.solve(A, b)[:-1]


def circumradius(points: Union[List[Tuple[float]], np.ndarray]) -> float:
    """
    Calculte the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    points = np.asarray(points)
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))


def alphasimplices(
    points: Union[List[Tuple[float]], np.ndarray]) -> \
        Union[List[Tuple[float]], np.ndarray]:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points)
        except np.linalg.LinAlgError:
            logging.warning('Singular matrix. Likely caused by all points '
                            'lying in an N-1 space.')


def alphashape(points: Union[List[Tuple[float]], np.ndarray],
               alpha: Union[None, float] = None):
    """
    Compute the alpha shape (concave hull) of a set of points.  If the number
    of points in the input is three or less, the convex hull is returned to the
    user.  For two points, the convex hull collapses to a `LineString`; for one
    point, a `Point`.

    Args:

      points (list or ``shapely.geometry.MultiPoint`` or \
          ``geopandas.GeoDataFrame``): an iterable container of points
      alpha (float): alpha value

    Returns:

      ``shapely.geometry.Polygon`` or ``shapely.geometry.LineString`` or
      ``shapely.geometry.Point`` or ``geopandas.GeoDataFrame``: \
          the resulting geometry
    """

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(points) < 4 or (alpha is not None and not callable(
            alpha) and alpha <= 0):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        return points.convex_hull

    # Convert the points to a numpy array
    coords = np.array(points)

    # Create a set to hold unique edges of simplices that pass the radius
    # filtering
    edges = set()

    # Create a set to hold unique edges of perimeter simplices.
    # Whenever a simplex is found that passes the radius filter, its edges
    # will be inspected to see if they already exist in the `edges` set.  If an
    # edge does not already exist there, it will be added to both the `edges`
    # set and the `permimeter_edges` set.  If it does already exist there, it
    # will be removed from the `perimeter_edges` set if found there.  This is
    # taking advantage of the property of perimeter edges that each edge can
    # only exist once.
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(coords):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        # Radius filter
        if circumradius < 1.0 / resolved_alpha:
            for edge in combinations(
                    point_indices, r=coords.shape[-1]):
                if all([e not in edges for e in combinations(
                        edge, r=len(edge))]):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(combinations(
                        edge, r=len(edge)))

    if coords.shape[-1] > 3:
        return perimeter_edges
    elif coords.shape[-1] == 3:
        import trimesh
        result = trimesh.Trimesh(vertices=coords, faces=list(perimeter_edges))
        trimesh.repair.fix_normals(result)
        return result

    # Create the resulting polygon from the edge points
    m = MultiLineString([coords[np.array(edge)] for edge in perimeter_edges])
    triangles = list(polygonize(m))
    result = unary_union(triangles)

    return result


def alphashape_convex_decomposition(
        points: Union[List[Tuple[float]], np.ndarray],
        alpha: Union[None, float] = None,
        area_threshold: Union[None, float] = None,
        max_vertices: int = 20):
    """
    Compute the alpha shape, triangulate only perimeter edges, 
    keep triangles inside, and perform greedy convex merging. If area_threshold
    is None, only performs exact convex merging. If area_threshold is
    given, performs relaxed convex merging.
    Args:
        points (list or ndarray): an iterable container of points
        alpha (float): alpha value
        area_threshold (float): area threshold for relaxed convex merging

    Returns:
        - alpha shape polygon
        - list of convex polygons (greedy merged)
    """

    if len(points) < 4 or \
            (alpha is not None and not callable(alpha) and alpha <= 0):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        result = points.convex_hull
        return result, [result]

    coords = np.array(points)

    # Step 1: Create alpha shape perimeter edges
    edges = set()
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(coords):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        if circumradius < 1.0 / resolved_alpha:
            for edge in combinations(point_indices, r=coords.shape[-1]):
                if all([e not in edges for e in
                        combinations(edge, r=len(edge))]):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(combinations(edge,
                                           r=len(edge)))

    if coords.shape[-1] != 2:
        raise ValueError("Only 2D points are supported.")

    # Step 2: Build alpha shape
    m = MultiLineString([coords[np.array(edge)] for edge in perimeter_edges])
    alpha_shape_polys = list(polygonize(m))
    alpha_shape = unary_union(alpha_shape_polys)

    # Step 3: Extract perimeter points
    perimeter_coords = []
    if alpha_shape.geom_type == 'Polygon':
        perimeter_coords = list(alpha_shape.exterior.coords)[
            :-1]  # remove closing point
    elif alpha_shape.geom_type == 'MultiPolygon':
        for poly in alpha_shape.geoms:
            perimeter_coords.extend(list(poly.exterior.coords)[:-1])
    perimeter_coords = np.array(perimeter_coords)

    # Step 4: Delaunay triangulation on perimeter points
    tri = Delaunay(perimeter_coords)

    # Step 5: Keep only triangles fully inside alpha shape
    valid_triangles = []
    for simplex in tri.simplices:
        triangle = Polygon(perimeter_coords[simplex])
        if triangle.is_valid and triangle.area > 1e-8 and \
                triangle.within(alpha_shape):
            valid_triangles.append(triangle)

    # Step 6: Greedy convex merging
    def is_convex(poly, relaxed=False, base=None, other=None,
                  area_threshold=0.0):
        if not relaxed:
            return poly.is_valid and poly.is_simple and \
                poly.convex_hull.equals(poly)
        else:
            valid_and_simple = poly.is_valid and poly.is_simple
            if not valid_and_simple:
                return False
            before = base.area + other.area
            after = poly.convex_hull.area
            if after - before < area_threshold:
                return True
            return False

    convex_pieces = []
    while valid_triangles:
        base = valid_triangles.pop()
        merged = False
        for i, other in enumerate(valid_triangles):
            candidate = unary_union([base, other])
            if isinstance(candidate, Polygon):
                if area_threshold is None:
                    convexity = is_convex(
                        candidate) and candidate.within(alpha_shape)
                else:
                    convexity = is_convex(
                        candidate, relaxed=True, base=base, other=other,
                        area_threshold=area_threshold)
                num_vertices = len(candidate.exterior.coords) - 1
                if convexity and num_vertices <= max_vertices:
                    base = candidate.convex_hull
                    valid_triangles.pop(i)
                    merged = True
                    break
        if not merged:
            convex_pieces.append(base)
        else:
            valid_triangles.append(base)  # Retry merging further

    # Step 7: Filter out small convex pieces
    convex_pieces = [
        poly for poly in convex_pieces if poly.area > area_threshold]

    return alpha_shape, convex_pieces


def polygon_to_halfspace(poly: Polygon):
    """Given a shapely convex polygon, return (A, b) for half-space 
    representation Ax + b < 0."""
    poly = orient(poly, sign=1)  # ensure counter-clockwise
    coords = np.array(poly.exterior.coords)
    n_edges = len(coords) - 1

    A = []
    b = []

    for i in range(n_edges):
        p1 = coords[i]  # ex: (0, 1)
        p2 = coords[i+1]  # ex: (-1, 0)
        edge = p2 - p1  # ex: (-1, -1)
        normal = np.array([edge[1], -edge[0]])  # outward normal, ex: (-1, 1)
        normal = normal / np.linalg.norm(normal)

        A.append(normal)
        b.append(normal @ p1)  # this gives A@p <= b

    # I need A@x <= b
    A = np.array(A)
    b = np.array(b)
    return A, b
