# Implement your function below.

def iou_score(bboxA, bboxB):
    #xA = np.maximum(bboxA[0], bboxB[0])

    #interWidth = np.maximum(0, xB - xA + 1)
    #interArea = interWidth * interHeight

    #find bbox areas of A and B

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return round(iou, 3)
