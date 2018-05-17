import cv2
import numpy as np
from util import showImage, drawPerspective, drawBoundaries, drawContour, drawPoint, writeDocumentationImage
from line import Line, partitionLines, filterCloseLines


houghThreshold = 150
hough_threshold_step = 20

def getPerspective(image, points):
    yy, xx, _ = image.shape
    tmp = np.zeros(image.shape[0:2], np.uint8)
    drawContour(tmp, points, (255,), 1)

    grid = None
    for i in range(int(houghThreshold/hough_threshold_step)):
        lines = cv2.HoughLines(tmp, 1, np.pi / 180, houghThreshold-(i * hough_threshold_step))
        if lines is None:
            continue
        lines = [Line(l[0], l[1]) for l in lines[0]]
        (horizontal, vertical) = partitionLines(lines)
        vertical = filterCloseLines(vertical, horizontal=False)
        horizontal = filterCloseLines(horizontal, horizontal=True)

        if len(vertical) == 2 and len(horizontal) == 2:
            grid = (vertical, horizontal)
            break


    if grid is None:
        return None


    if vertical[0].getCenter()[0] > vertical[1].getCenter()[0]:
        v2, v1 = vertical
    else:
        v1, v2 = vertical

    if horizontal[0].getCenter()[1] > horizontal[1].getCenter()[1]:
        h2, h1 = horizontal
    else:
        h1, h2 = horizontal



    perspective = (h1.intersect(v1),
                   h1.intersect(v2),
                   h2.intersect(v2),
                   h2.intersect(v1))

    ## Doc ##
    #tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    #drawContour(tmp, points, (0,0,255), 3)
    #writeDocumentationImage(tmp, "contour_individual_bw")
    #tmp_bw = tmp
    #tmp_orig = image.copy()
    #for tmp in (tmp_bw, tmp_orig):
    #    for l in (v1,v2,h1,h2): l.draw(tmp, (0,255,0), 2)
    #    for p in perspective: drawPoint(tmp, p, (255,0,0), 3)
    #writeDocumentationImage(tmp_bw, "contour_lines_bw")
    #writeDocumentationImage(tmp_orig, "contour_lines_orig")
    ## Doc ##


    return perspective
