from transforms3d import affines, euler
from . import xyz2zxy


def move_mirror(element, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0.):
    pos = affines.compose([dx, dy, dz + conf['f']],
                          euler.euler2mat(rx, ry, rz, 'sxyz') @ xyz2zxy[:3, :3],
                          [conf['mirr_length'],
                           conf['mirr_rout'],
                           conf['mirr_rout']])
    element.geometry.pos4d[:] = pos


def offsetpoint(element, position_angle, separation):
    element.coords = coords.directional_offset_by(position_angle, separation)
