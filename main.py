from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))


@ti.func
def rgb(r, g, b):
    return vec3(r / 255.0, g / 255.0, b / 255.0)


small_ring_radius = 28
big_ring_radius = int(small_ring_radius * 2 / (3 ** 0.5))
small_ring_center_offset = 3 ** 0.5 / 6 * 2 * small_ring_radius / 2
tri_perpendicular_length = 3 ** 0.5 / 3 * small_ring_radius


def get_circle(p1, p2, p3):
    x, y, z = p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j, p3[0] + p3[1] * 1j
    w = z - x
    w /= y - x
    circle = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    return (-circle.real, -circle.imag), abs(circle + x)


@ti.func
def draw_radius(v, circle_center, circle_radius, precision=0.7):
    return ti.abs(ti.math.distance(v, circle_center) - circle_radius) <= precision


n = 64
center_pos = [0, 10]
big_ring, big_radius = get_circle((center_pos[0] + small_ring_radius, center_pos[1] + tri_perpendicular_length),
                                  (center_pos[0] - small_ring_radius, center_pos[1] + tri_perpendicular_length),
                                  (center_pos[0], center_pos[1] - tri_perpendicular_length * 2))


@ti.func
def draw_arc(v, circle_center, circle_radius, down_vector, max_degree=30, precision=0.7):
    # acos always equal pi/2 because f32 precision, now ti don't support f64 acos function
    # ob = v - circle_center
    # dotValue = ti.math.dot(ob, down_vector)
    # degree = ti.math.degrees(ti.acos(dotValue / (down_vector.norm() * ob.norm())))
    return ti.abs(ti.math.distance(v, circle_center) - circle_radius) <= precision and ti.abs(v[0]) <= max_degree

@ti.func
def draw_line(v, height_edge_bottom, height_edge_top, edge):
    return center_pos[0] - edge < v[0] < center_pos[0] + edge and height_edge_bottom < v[1] < height_edge_top


@ti.kernel
def initialize_voxels():
    # My code here :-)
    yellow, brown, write = rgb(245, 188, 84), rgb(195, 107, 43), rgb(254, 255, 247)

    big_ring_center = ti.Vector([big_ring[0], big_ring[1]])
    ring1_center = ti.Vector([center_pos[0] + small_ring_radius / 2, center_pos[1] + small_ring_center_offset])
    ring2_center = ti.Vector([center_pos[0] - small_ring_radius / 2, center_pos[1] + small_ring_center_offset])
    ring3_center = ti.Vector([center_pos[0], center_pos[1] - tri_perpendicular_length])

    arc1_center = ti.Vector([center_pos[0], center_pos[1] - tri_perpendicular_length - 50 / 31 * small_ring_radius])
    arc2_center = ti.Vector([center_pos[0], center_pos[1] - tri_perpendicular_length - ((5 + 31) / 31 * small_ring_radius)])
    arc_big_center = ti.Vector([center_pos[0], center_pos[1] - tri_perpendicular_length - (149 / 31 * small_ring_radius)])
    arc_center1_radius = 50 / 31 * small_ring_radius
    arc_center2_radius = 67 / 31 * small_ring_radius
    arc_center3_radius = 97 / 31 * small_ring_radius

    line_height_low = big_ring_center[1] + big_radius + tri_perpendicular_length / 3
    line_height_top = arc_big_center[1] + arc_center3_radius

    down_vector = ti.Vector([0, -1])

    for i, j in ti.ndrange((-n, n), (-n, n)):
        v = ti.Vector([i, j])
        if draw_radius(v, big_ring_center, big_radius, 0.7) \
                or draw_radius(v, ring1_center, small_ring_radius, 0.8) \
                or draw_radius(v, ring2_center, small_ring_radius, 0.8) \
                or draw_radius(v, ring3_center, small_ring_radius, 0.8) \
                or draw_arc(v, arc1_center, arc_center1_radius, down_vector, 30, 0.5) \
                or draw_arc(v, arc2_center, arc_center2_radius, down_vector, 45, 0.5) \
                or draw_arc(v, arc_big_center, arc_center3_radius, down_vector, 32) \
                or draw_line(v, line_height_top, line_height_low, 1):
            random = ti.random()
            if random < 0.001:
                scene.set_voxel(vec3(i, 1, j), 2, write)
            elif random < 0.1:
                scene.set_voxel(vec3(i, 1, j), 2, yellow)
            else:
                scene.set_voxel(vec3(i, 1, j), 2, brown)


initialize_voxels()

scene.finish()
