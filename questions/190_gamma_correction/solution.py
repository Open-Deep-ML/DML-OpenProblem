def apply_gamma_correction(img, gamma):

    if not img or gamma <= 0:
        return -1

    row_len = len(img[0])

    for row in img:
        if len(row) != row_len:
            return -1
        for p in row:
            if p < 0 or p > 255:
                return -1

    result = []

    for row in img:
        new_row = []
        for p in row:
            val = 255 * ((p / 255) ** gamma)
            val = round(val)
            val = min(255, max(0, val))
            new_row.append(val)
        result.append(new_row)

    return result
