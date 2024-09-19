import numpy as np

class PointFormatter:
    mode: str = None

    @staticmethod
    def __call__(pts: list, h: int, w: int):
        raise NotImplementedError('Method not implemented')

class SegPointFormatter(PointFormatter):
    mode = 'seg'

    @staticmethod
    def __call__(pts: list, h: int, w: int):
        formatted_pts = []
        for pt in pts:
            formatted_pts.extend([str(pt[0] / w), str(pt[1] / h)])

        return formatted_pts
    
class BoxPointFormatter(PointFormatter):
    mode = 'bbox'

    @staticmethod
    def __call__(pts: list, h: int, w: int):
        pts = np.array(pts)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)

        obj_w = (max_x - min_x)
        obj_h = (max_y - min_y)

        return [
            str((min_x + obj_w / 2.0) / w),
            str((min_y + obj_h / 2.0) / h),
            str(obj_w / w),
            str(obj_h / h)
        ]

class Labelme2Yolov8:
    seg: PointFormatter = SegPointFormatter()
    box: PointFormatter = BoxPointFormatter()   