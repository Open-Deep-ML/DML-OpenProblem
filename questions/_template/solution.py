import numpy as np
def iou_score(bboxA, bboxB):
    xA = np.maximum(bboxA[0], bboxB[0])
    yA = np.maximum(bboxA[1], bboxB[1])
    xB = np.minimum(bboxA[2], bboxB[2])
    yB = np.minimum(bboxA[3], bboxB[3])

    interWidth = np.maximum(0, xB - xA + 1)
    interHeight = np.maximum(0, yB - yA + 1)
    interArea = interWidth * interHeight

    boxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    boxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    #unionArea = boxAArea + boxBArea - interArea
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return round(iou, 3)