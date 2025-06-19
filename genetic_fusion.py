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
import pandas as pd
import matplotlib.pyplot as plt
import imageio

class GeneticMapFusion:
    def __init__(self, ref_map, target_map, map_name = "", max_generations = 100, pop_size=100, generations=30,
                convergence_generations = 5, mutation_std=(4, 4, 4), elite_size=3, selection_pool=10,
                 fitness_weights=(0.7, 0.3)):
        self.ref_map = ref_map
        self.target_map = target_map
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_std = np.array(mutation_std)
        self.elite_size = elite_size
        self.selection_pool = selection_pool
        self.max_generations = max_generations
        self.fitness_log = pd.DataFrame()
        self.map_name = map_name
        self.convergence_generations = convergence_generations
        self.fitness_weights = fitness_weights  # (NCC_weight, EdgeIoU_weight)
        self.frames = []  # For saving GIF

    def edge_weighted_iou(self, ref, target):
        # Edge dilatation using sobel
        ref_processed = binary_dilation(ref.astype(bool))
        target_processed = binary_dilation(target.astype(bool))

        ref_edges = sobel(ref_processed.astype(float)).astype(bool)
        target_edges = sobel(target_processed.astype(float)).astype(bool)

        intersection = np.logical_and(ref_edges, target_edges).sum()
        union = np.logical_or(ref_edges, target_edges).sum()

        return intersection / union if union > 0 else 0

    def evaluate_fitness(self, individual):
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

        # Normalized Cross Correlation (NCC)
        numerator = np.sum(region_ref * region_target)
        denominator = np.sqrt(np.sum(region_ref ** 2) * np.sum(region_target ** 2))
        ncc_score = numerator / denominator if denominator > 0 else 0

        # Edge-weighted IoU
        edge_iou = self.edge_weighted_iou(region_ref, region_target)

        # Weighted combination
        score = self.fitness_weights[0] * ncc_score + self.fitness_weights[1] * edge_iou
        return score

    def generate_initial_population(self, tx_range, ty_range, theta_range):
        # Initialize using Gaussian distribution centered at 0
        tx_std = (tx_range[1] - tx_range[0]) / 6
        ty_std = (ty_range[1] - ty_range[0]) / 6
        theta_std = (theta_range[1] - theta_range[0]) / 6

        return [np.array([
            np.clip(np.random.normal(0, tx_std), *tx_range),
            np.clip(np.random.normal(0, ty_std), *ty_range),
            np.clip(np.random.normal(0, theta_std), *theta_range)
        ]) for _ in range(self.pop_size)]

    def evolve(self, tx_range=(-20, 20), ty_range=(-20, 20), theta_range=(-15, 15),
            fitness_threshold=0.80, generations=30, max_generations=100,
            convergence_generations=5, improvement_threshold=1e-4): # check this params!

        if tx_range is None:
            tx_range = (-self.ref_map.shape[1], self.ref_map.shape[1])
        if ty_range is None:
            ty_range = (-self.ref_map.shape[0], self.ref_map.shape[0])

        population = self.generate_initial_population(tx_range, ty_range, theta_range)
        best_fitness = -np.inf
        best_individual = None

        stagnant_count = 0  # Tracks stagnant generations

        for gen in range(max_generations):
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            if self.fitness_log.empty:
                self.fitness_log = pd.DataFrame(columns=[f"chrom_{i}" for i in range(len(fitness_scores))])
            self.fitness_log.loc[f'g{gen}'] = fitness_scores

            gen_best = max(fitness_scores)
            gen_best_ind = population[np.argmax(fitness_scores)]

            # Track convergence
            if gen_best > best_fitness + improvement_threshold:
                best_fitness = gen_best
                best_individual = gen_best_ind
                stagnant_count = 0  # Reset counter
            else:
                stagnant_count += 1

            # Print fitness if needed
            # print(f"Generation {gen}: Best fitness = {gen_best:.4f}")

            # Early stop: if fitness threshold met *and* minimum generations passed
            if best_fitness >= fitness_threshold and gen >= generations:
                print(f"Stopping early due to threshold at generation {gen} with fitness {best_fitness:.4f}")
                break

            # Early stop: if no significant improvement for N generations
            if stagnant_count >= convergence_generations and gen >= generations:
                print(f"Stopping early due to convergence at generation {gen} with fitness {best_fitness:.4f}")
                break

            # Elitism + mutation
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda p: -p[0])]
            new_population = sorted_population[:self.elite_size]

            # covariance matrix for mutation
            cov_matrix = np.diag((self.mutation_std * (1 - gen / max_generations))**2)

            while len(new_population) < self.pop_size:
                parents = random.sample(sorted_population[:self.selection_pool], 2)

                # Crossover with random weight
                alpha = random.uniform(0.3, 0.7)
                child = alpha * parents[0] + (1 - alpha) * parents[1]
                # Change from random mutation with some normal noise to scale covariance
                # mutation_scale = self.mutation_std * (1 - gen / max_generations)
                # mutation = np.random.normal(loc=0, scale=mutation_scale)

                # Mutate with multivariate normal
                mutation = np.random.multivariate_normal(mean=np.zeros(3), cov=cov_matrix)

                # Occasionally add stronger mutations
                if gen % 5 == 0 and random.random() < 0.3:
                    mutation += np.random.multivariate_normal(mean=np.zeros(3), cov=np.diag((self.mutation_std * 0.5)**2))

                # Occasionally mutate a single parameter more
                if random.random() < 0.1:
                    param_idx = np.random.randint(0, 3)
                    mutation[param_idx] += np.random.normal(loc=0, scale=self.mutation_std[param_idx] * 2)

                child += mutation
                new_population.append(child)

            population = new_population

        # logging
        filename = f"ga_{self.map_name}_pop{self.pop_size}_gen{self.generations}.csv"
        self.fitness_log.to_csv(filename, index=True)

        # save_alignment_fitness
        self.generate_alignment_plot()
        self.save_gif(None, final=True)

        print(f"Final Generation {gen}: Best fitness = {best_fitness:.4f}")
        return best_individual
    
    def save_gif(self, individual, gen=0, fitness=None, final=False):
        if final:
            imageio.mimsave(f"alignment_{self.map_name}.gif", self.frames, duration=0.4)
            return

        fused_map, canvas, target_canvas = self.fuse_maps_aligned(individual)
        display = (fused_map * 255).astype(np.uint8)
        display_color = cv.cvtColor(display, cv.COLOR_GRAY2BGR)

        cv.putText(display_color, f"Gen: {gen}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(display_color, f"Fitness: {fitness:.3f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.frames.append(display_color)

    def generate_alignment_plot(self):
        df = self.fitness_log.transpose()
        best_indices = df.idxmax()
        best_params = [eval(i.replace("chrom_", "")) for i in best_indices.index]
        plt.figure(figsize=(10, 4))
        plt.plot(df.max(axis=0))
        plt.title("Fitness Progression")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.savefig(f"fitness_progress_{self.map_name}.png")
        plt.close()

    def fuse_maps_aligned(self, best_params):
        # Fuse maps using the best transformation parameters found
        tx, ty, theta = best_params
        rotated = rotate(self.target_map, theta, reshape=False, order=1, mode='constant', cval=0)

        ref_h, ref_w = self.ref_map.shape
        rot_h, rot_w = rotated.shape

        # Create canvas large enough to contain both maps
        canvas_h = max(ref_h, rot_h + abs(int(ty))) + ref_h/2
        canvas_w = max(ref_w, rot_w + abs(int(tx))) + ref_w/2
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
