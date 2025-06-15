def dot(v1, v2):
    return sum([ax1 * ax2 for ax1, ax2 in zip(v1, v2)])


def scalar_mult(scalar, v):
    return [scalar * ax for ax in v]


def orthogonal_projection(v, L):
    L_mag_sq = dot(L, L)
    proj_scalar = dot(v, L) / L_mag_sq
    proj_v = scalar_mult(proj_scalar, L)
    return [round(x, 3) for x in proj_v]
