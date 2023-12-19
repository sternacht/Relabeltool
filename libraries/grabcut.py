import cv2 as cv
import numpy as np

def grab_cut(image:np.array, rects, iterCount = 2):
    """
    Parameter:  image: array of the image
                rects: list of tuple (x1, y1, x2, y2)
    """
    total_mask = np.zeros(image.shape[:2], dtype='uint8')
    for rect in rects:
        if len(rect) > 4:
            rect = rect[:4]
        x1, y1, x2, y2 = rect
        rect_xywh = (int(x1-10), int(y1-10), int(x2-x1+20), int(y2-y1+20))
        mask = np.zeros(image.shape[:2], dtype='uint8')
        fgModel = np.zeros((1,65), dtype='float')
        bgModel = np.zeros((1,65), dtype='float')
        (mask, bgModel, fgModel) = cv.grabCut(image, mask, rect_xywh, 
            bgModel, 
            fgModel, 
            iterCount=iterCount, 
            mode=cv.GC_INIT_WITH_RECT)
        outputMask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),0, 1)
        outputMask = (outputMask * 255).astype("uint8")
        total_mask = np.add(total_mask, outputMask)
    # output = cv.bitwise_and(image, image, mask=total_mask)
    # # x2 = rects[0] + rects[2]
    # # y2 = rects[1] + rects[3]
    # # image = cv.rectangle(image, rects[:2], (x2, y2), (0,255,0), 2)
    # cv.imshow("Input", image)
    # cv.imshow("GrabCut Mask", total_mask)
    # cv.imshow("GrabCut Output", output)
    # cv.waitKey(0)
    contours, _ = cv.findContours(total_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    polygons = []

    for object in contours:
        coords = []
        p0 = object[0]
        for point in object:
            coords.append((int(point[0][0]), int(point[0][1])))
        polygons.append(tuple(coords))
    return total_mask, polygons
            

if __name__ == "__main__":
    image = cv.imread(r"libraries\IMG-0068-0100.jpg")
    rect = [(112,253,125, 266)]
    mask, polygons = grab_cut(image, rect)
    print(polygons)