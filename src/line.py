import cv2
import numpy as np

class Line:
    def __init__(self, rho, theta):
        self._rho = rho
        self._theta = theta
        self._cos_factor = np.cos(theta)
        self._sin_factor = np.sin(theta)
        self._center = (self._cos_factor * rho, self._sin_factor * rho)

    def getCenter(self):
        return self._center

    def getRho(self):
        return self._rho

    def getTheta(self):
        return self._theta

    def getSegment(self, lenLeft, lenRight):
        a = self._cos_factor
        b = self._sin_factor
        (x0, y0) = self._center

        x1 = int(x0 + lenRight * (-b))
        y1 = int(y0 + lenRight * a)
        x2 = int(x0 - lenLeft * (-b))
        y2 = int(y0 - lenLeft  * a)
        return ((x1, y1), (x2, y2))

    def isHorizontal(self, thresholdAngle=np.pi / 4):
        return abs(np.sin(self._theta)) > np.cos(thresholdAngle)

    def isVertical(self, thresholdAngle=np.pi / 4):
        return abs(np.cos(self._theta)) > np.cos(thresholdAngle)

    def intersect(self, line):
        ct1 = np.cos(self._theta)
        st1 = np.sin(self._theta)
        ct2 = np.cos(line._theta)
        st2 = np.sin(line._theta)
        d = ct1 * st2 - st1 * ct2
        if d == 0.0: raise ValueError('parallel lines: %s, %s)' % (str(self), str(line)))
        x = (st2 * self._rho - st1 * line._rho) / d
        y = (-ct2 * self._rho + ct1 * line._rho) / d
        return (x, y)

    def draw(self, image, color=(0,0,255), thickness=2):
        p1, p2 = self.getSegment(1000,1000)
        cv2.line(image, p1, p2, color, thickness)


    def __repr__(self):
        return "(t: %.2fdeg, r: %.0f)" % (self._theta *360/np.pi, self._rho)

def partitionLines(lines):
    h = filter(lambda x: x.isHorizontal(), lines)
    v = filter(lambda x: x.isVertical(), lines)

    h = [(l._center[1], l) for l in h]
    v = [(l._center[0], l) for l in v]

    h.sort()
    v.sort()

    h = [l[1] for l in h]
    v = [l[1] for l in v]

    return (h, v)

def filterCloseLines(lines, horizontal=True, threshold = 40):
    if horizontal:
        item = 1
    else:
        item = 0

    i = 0
    ret = []

    while i < len(lines):
        itmp = i
        while i < len(lines) and (lines[i]._center[item] - lines[itmp]._center[item] < threshold):
            i += 1
        ret.append(lines[itmp + int((i - itmp) / 2)])
    return ret
