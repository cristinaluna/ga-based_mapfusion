'''

Map generator


Authors: C. Luna, E. Ruiz, P. López, D. F. Barrero
Maintainer: @cristinaluna

'''


import numpy as np
import cv2 as cv
import random
from scipy.ndimage import rotate


def draw_room(img, x, y, w, h, filled=True):
    shape_type = random.choice(['rect', 'rotated_rect', 'irregular'])
    if shape_type == 'rect':
        cv.rectangle(img, (x, y), (x + w, y + h), 255, -1 if filled else 2)
    elif shape_type == 'rotated_rect':
        angle = random.uniform(-30, 30)
        rect = ((x + w // 2, y + h // 2), (w, h), angle)
        box = cv.boxPoints(rect).astype(int)
        cv.drawContours(img, [box], 0, 255, -1)
    elif shape_type == 'irregular':
        pts = np.array([
            [x, y],
            [x + w, y + random.randint(-10, 10)],
            [x + w + random.randint(-10, 10), y + h],
            [x + random.randint(-10, 10), y + h + random.randint(-10, 10)]
        ])
        cv.fillPoly(img, [pts], 255)

    # Larger obstacles (more and bigger)
    for _ in range(random.randint(5, 10)):
        ox = x + random.randint(10, max(10, w - 40))
        oy = y + random.randint(10, max(10, h - 40))
        ow = random.randint(15, min(30, w - (ox - x)))
        oh = random.randint(15, min(30, h - (oy - y)))
        cv.rectangle(img, (ox, oy), (ox + ow, oy + oh), 0, -1)


def connect_rooms_with_corridor(img, room1, room2):
    x1, y1, w1, h1 = room1
    x2, y2, w2, h2 = room2

    edges1 = [
        (x1 + w1 // 2, y1),           # top center
        (x1 + w1 // 2, y1 + h1),      # bottom center
        (x1, y1 + h1 // 2),           # left center
        (x1 + w1, y1 + h1 // 2)       # right center
    ]
    edges2 = [
        (x2 + w2 // 2, y2),
        (x2 + w2 // 2, y2 + h2),
        (x2, y2 + h2 // 2),
        (x2 + w2, y2 + h2 // 2)
    ]

    min_dist = float('inf')
    best_pair = (edges1[0], edges2[0])
    for e1 in edges1:
        for e2 in edges2:
            dist = (e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                best_pair = (e1, e2)

    (x_start, y_start), (x_end, y_end) = best_pair

    thickness = random.randint(15, 25)

    if random.random() > 0.5:
        mid_point = (x_end, y_start)
    else:
        mid_point = (x_start, y_end)

    cv.line(img, (x_start, y_start), mid_point, 255, thickness)
    cv.line(img, mid_point, (x_end, y_end), 255, thickness)

    for _ in range(random.randint(6, 10)):
        tx = random.randint(min(x_start, x_end), max(x_start, x_end))
        ty = random.randint(min(y_start, y_end), max(y_start, y_end))
        ow = random.randint(10, 20)
        oh = random.randint(10, 20)
        cv.rectangle(img, (tx, ty), (tx + ow, ty + oh), 0, -1)


def generate_overlapping_submaps(global_size=(1200, 1200), submap_size=(600, 600),
                                 overlap_ratio=0.6, num_rooms=6, seed=None,
                                 apply_rotation=True, max_rotation=5,
                                 apply_noise=True, noise_std=0.02):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    H, W = global_size
    sub_h, sub_w = submap_size

    base = np.zeros((H, W), dtype=np.uint8)
    rooms = []

    for _ in range(num_rooms):
        rw = random.randint(120, 180)  # bigger rooms
        rh = random.randint(120, 180)
        x = random.randint(0, W - rw - 1)
        y = random.randint(0, H - rh - 1)
        draw_room(base, x, y, rw, rh)
        rooms.append((x, y, rw, rh))

    for i in range(len(rooms) - 1):
        connect_rooms_with_corridor(base, rooms[i], rooms[i + 1])

    overlap_px_h = int(sub_h * overlap_ratio)
    overlap_px_w = int(sub_w * overlap_ratio)

    margin = 30
    max_y = H - sub_h - margin
    max_x = W - sub_w - margin

    y1 = random.randint(margin, max_y)
    x1 = random.randint(margin, max_x)

    y2 = y1 + sub_h - overlap_px_h + random.randint(-15, 15)
    x2 = x1 + sub_w - overlap_px_w + random.randint(-15, 15)

    y2 = np.clip(y2, margin, max_y)
    x2 = np.clip(x2, margin, max_x)

    submap1 = base[y1:y1 + sub_h, x1:x1 + sub_w].copy()
    submap2 = base[y2:y2 + sub_h, x2:x2 + sub_w].copy()

    if apply_rotation:
        angle = random.uniform(-max_rotation, max_rotation)  # subtle rotation
        submap2 = rotate(submap2, angle, reshape=False, order=1, mode='constant', cval=0)
        submap2 = (submap2 > 0.5).astype(np.uint8)

    if apply_noise:
        noise = np.random.normal(0, noise_std, submap2.shape)
        submap2 = np.clip(submap2.astype(np.float32) + noise, 0, 1)
        submap2 = (submap2 > 0.5).astype(np.uint8)

    return (submap1 > 0).astype(np.uint8), (submap2 > 0).astype(np.uint8)
