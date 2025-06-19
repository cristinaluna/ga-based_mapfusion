'''

Map fusion using GA algorithm


Authors: C. Luna, E. Ruiz, P. López
Maintainer: @cristinaluna

'''


import numpy as np
import random
import cv2 as cv
from scipy.ndimage import rotate, sobel, binary_dilation
from scipy.signal import correlate2d
from skimage.transform import AffineTransform, warp
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class GeneticMapFusion:
    def __init__(self, ref_map, target_map, max_generations = 100, pop_size=20, generations=40,
                 mutation_std=(4, 4, 4), elite_size=3, selection_pool=10):
        self.ref_map = ref_map
        self.target_map = target_map
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_std = np.array(mutation_std)
        self.elite_size = elite_size
        self.selection_pool = selection_pool
        self.max_generations = max_generations

    def edge_weighted_iou(self, ref, target):
        # Edge dilatation using sobel
        ref_processed = binary_dilation(ref.astype(bool))
        target_processed = binary_dilation(target.astype(bool))

        ref_edges = sobel(ref_processed.astype(float)).astype(bool)
        target_edges = sobel(target_processed.astype(float)).astype(bool)

        intersection = np.logical_and(ref_edges, target_edges).sum()
        union = np.logical_or(ref_edges, target_edges).sum()

        return intersection / union if union > 0 else 0

    def evaluate_fitness(self, individual): # using ncc instead of IoU since it is more robust for non-fully-coincident maps
        tx, ty, theta = individual
        rotated = rotate(self.target_map, theta, reshape=False, order=1, mode='constant', cval=0)

        canvas_h = self.ref_map.shape[0] * 3
        canvas_w = self.ref_map.shape[1] * 3
        ref_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        target_canvas = np.zeros_like(ref_canvas)

        center_y, center_x = canvas_h // 2, canvas_w // 2

        y_ref = center_y - self.ref_map.shape[0] // 2
        x_ref = center_x - self.ref_map.shape[1] // 2
        ref_canvas[y_ref:y_ref + self.ref_map.shape[0], x_ref:x_ref + self.ref_map.shape[1]] = self.ref_map

        y_shift = center_y - rotated.shape[0] // 2 + int(round(ty))
        x_shift = center_x - rotated.shape[1] // 2 + int(round(tx))

        y_start = np.clip(y_shift, 0, canvas_h)
        x_start = np.clip(x_shift, 0, canvas_w)
        y_end = np.clip(y_shift + rotated.shape[0], 0, canvas_h)
        x_end = np.clip(x_shift + rotated.shape[1], 0, canvas_w)

        ry1 = np.clip(-y_shift, 0, rotated.shape[0])
        rx1 = np.clip(-x_shift, 0, rotated.shape[1])
        ry2 = ry1 + (y_end - y_start)
        rx2 = rx1 + (x_end - x_start)

        if any(val <= 0 for val in [y_end - y_start, x_end - x_start, ry2 - ry1, rx2 - rx1]):
            return 0

        try:
            target_canvas[y_start:y_end, x_start:x_end] = rotated[ry1:ry2, rx1:rx2]
        except ValueError:
            return 0

        region_ref = ref_canvas[y_start:y_end, x_start:x_end]
        region_target = target_canvas[y_start:y_end, x_start:x_end]

        if region_ref.shape != region_target.shape:
            return 0

        numerator = np.sum(region_ref * region_target)
        denominator = np.sqrt(np.sum(region_ref ** 2) * np.sum(region_target ** 2))
        score = numerator / denominator if denominator > 0 else 0

        if self.generations == 0:
            print(f"Initial fitness score: {score:.4f} for params {individual}")

        return score

    def generate_initial_population(self, tx_range, ty_range, theta_range):
        return [np.array([
            random.uniform(*tx_range),
            random.uniform(*ty_range),
            random.uniform(*theta_range)
        ]) for _ in range(self.pop_size)]

    def evolve(self, tx_range=(-20, 20), ty_range=(-20, 20), theta_range=(-15, 15), fitness_threshold=0.80, min_generations=30, max_generations=1000):
        if tx_range is None:
            tx_range = (-self.ref_map.shape[1], self.ref_map.shape[1])
        if ty_range is None:
            ty_range = (-self.ref_map.shape[0], self.ref_map.shape[0])

        population = self.generate_initial_population(tx_range, ty_range, theta_range)
        best_fitness = -np.inf
        best_individual = None

        for gen in range(max_generations):
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            gen_best = max(fitness_scores)
            gen_best_ind = population[np.argmax(fitness_scores)]

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_individual = gen_best_ind

            #print(f"Generation {gen}: Best fitness = {gen_best:.4f}")

            # Stop early if above threshold and min generations completed
            if best_fitness >= fitness_threshold and gen >= min_generations:
                print(f"Stopping early at generation {gen} with fitness {best_fitness:.4f}")
                break

            # Elitism: keep top individuals
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda p: -p[0])]
            new_population = sorted_population[:self.elite_size]

            while len(new_population) < self.pop_size:
                parents = random.sample(sorted_population[:self.selection_pool], 2)
                mutation_scale = self.mutation_std * (1 - gen / max_generations)

                child = (parents[0] + parents[1]) / 2
                mutation = np.random.normal(loc=0, scale=mutation_scale)

                if gen % 5 == 0 and random.random() < 0.3:
                    mutation += np.random.normal(loc=0, scale=self.mutation_std * 0.5)
                if random.random() < 0.1:
                    param_idx = np.random.randint(0, 3)
                    mutation[param_idx] += np.random.normal(loc=0, scale=self.mutation_std[param_idx] * 2)

                child += mutation
                new_population.append(child)

            population = new_population

        print(f"Generation {gen}: Best fitness = {gen_best:.4f}")
        return best_individual

    def fuse_maps_aligned(self, best_params):
        # Fuse maps using the best transformation parameters found
        tx, ty, theta = best_params
        rotated = rotate(self.target_map, theta, reshape=False, order=1, mode='constant', cval=0)

        ref_h, ref_w = self.ref_map.shape
        rot_h, rot_w = rotated.shape

        # Create canvas large enough to contain both maps
        canvas_h = max(ref_h, rot_h + abs(int(ty))) + 50
        canvas_w = max(ref_w, rot_w + abs(int(tx))) + 50
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Place reference map
        y_offset = canvas_h // 2 - ref_h // 2
        x_offset = canvas_w // 2 - ref_w // 2
        canvas[y_offset:y_offset + ref_h, x_offset:x_offset + ref_w] = self.ref_map

        # Place transformed target map
        shifted_y = y_offset + int(round(ty))
        shifted_x = x_offset + int(round(tx))
        target_canvas = np.zeros_like(canvas)

        y_start = np.clip(shifted_y, 0, canvas_h)
        x_start = np.clip(shifted_x, 0, canvas_w)
        y_end = np.clip(shifted_y + rot_h, 0, canvas_h)
        x_end = np.clip(shifted_x + rot_w, 0, canvas_w)

        crop_y1 = np.clip(-shifted_y, 0, rot_h)
        crop_x1 = np.clip(-shifted_x, 0, rot_w)
        crop_y2 = crop_y1 + (y_end - y_start)
        crop_x2 = crop_x1 + (x_end - x_start)

        if (crop_y2 > crop_y1) and (crop_x2 > crop_x1):
            try:
                target_canvas[y_start:y_end, x_start:x_end] = rotated[crop_y1:crop_y2, crop_x1:crop_x2]
            except ValueError:
                pass

        fused_map = np.where((canvas == 1) & (target_canvas == 1), 1,
                     np.where((canvas == 1) | (target_canvas == 1), 0.5, 0))

        return fused_map, canvas, target_canvas
      
# ----------------- COMPARISON WITH OTHER METHODS --------

    # aligning using cross correlation
    def align_with_cross_correlation(self):
        correlation = correlate2d(self.ref_map, self.target_map, mode='full')
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        shift_y = max_idx[0] - self.target_map.shape[0] + 1
        shift_x = max_idx[1] - self.target_map.shape[1] + 1

        return np.array([shift_x, shift_y, 0])

    # aligning using ORB descriptors and RANSAC
    def align_with_ransac(self):
        kp_detector = cv.ORB_create(500)
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # ORB needs uint8 images
        ref_uint8 = (self.ref_map * 255).astype(np.uint8)
        target_uint8 = (self.target_map * 255).astype(np.uint8)

        kp1, des1 = kp_detector.detectAndCompute(ref_uint8, None)
        kp2, des2 = kp_detector.detectAndCompute(target_uint8, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            print("Insufficient keypoints for RANSAC.")
            return np.array([0, 0, 0])

        matches = matcher.match(des1, des2)
        if len(matches) < 4:
            print("Not enough matches for RANSAC.")
            return np.array([0, 0, 0])

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)

        M, inliers = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC)
        if M is None:
            return np.array([0, 0, 0])

        tx = M[0, 2]
        ty = M[1, 2]
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi  # radians to degrees

        return np.array([tx, ty, angle])
