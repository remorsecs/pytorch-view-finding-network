from __future__ import absolute_import
from __future__ import print_function


class ImageCropperEvaluator:
    def __init__(self):
        super(ImageCropperEvaluator, self).__init__()

    def evaluate(self, ground_truth, crops, sizes):
        assert len(ground_truth) == len(crops), "Different number of ground truth and crops."

        image_num = 0
        alpha = 0.75
        alpha_cnt = 0
        accum_boundary_displacement = 0
        accum_overlap_ratio = 0
        # crop_cnt = 0

        for gt, crop, size in zip(ground_truth, crops, sizes):
            # original image size
            width, height = size

            x, y, w, h = gt
            best_x, best_y, best_w, best_h = crop
            boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w)) / float(width) + \
                                    (abs(best_y - y) + abs(best_y + best_h - y - h)) / float(height)
            accum_boundary_displacement += boundary_displacement
            ratio = self._intersection_over_union(gt, crop)
            if ratio >= alpha:
                alpha_cnt += 1
            accum_overlap_ratio += ratio
            image_num += 1
            # crop_cnt += len(crops)

        print('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / image_num))
        print('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (image_num * 4.0)))
        print('Alpha recall: {:.4f}'.format(100 * float(alpha_cnt) / image_num))
        print('Total image evaluated:', image_num)
        # print('Average crops per image:', float(crop_cnt) / image_num)

    @staticmethod
    def _intersection_over_union(gt, crop):
        x1, y1, w1, h1 = gt
        x2, y2, w2, h2 = crop
        intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union = (w1 * h1) + (w2 * h2) - intersection
        return float(intersection) / float(union)
