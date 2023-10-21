from enum import Enum

import cv2
import numpy as np
import math
from ..regularization import boundary_regularization

def get_polygon(mask, sample="Dynamic", building=False):
    results = cv2.findContours(
        image=mask, mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE)
        #method=cv2.CHAIN_APPROX_TC89_KCOS)
    cv2_v = cv2.__version__.split(".")[0]
    contours = results[1] if cv2_v == "3" else results[0] 
    hierarchys = results[2] if cv2_v == "3" else results[1]
    if len(contours) != 0:
        polygons = []
        relas = []
        centers = []
        img_shape = mask.shape
        for idx, (contour,hierarchy) in enumerate(zip(contours, hierarchys[0])):
            # print(hierarchy)
            
            epsilon = (0.001 * cv2.arcLength(contour, True)
                       if sample == "Dynamic" else sample)
            if not isinstance(epsilon, float) and not isinstance(epsilon, int):
                epsilon = 0
            # print("epsilon:", epsilon)
            if building is False:
                contour = cv2.approxPolyDP(contour, epsilon / 100, True)
            else:
                contour = boundary_regularization(contour, img_shape, epsilon)
            
            out = approx_poly_DIY(contour)
            center = __cal_center(contour)
            rela = (
                idx,  # own
                hierarchy[-1] if hierarchy[-1] != -1 else None, )  # parent
            polygon = []
            for p in out:
                polygon.append(p[0])
            polygons.append(polygon) 
            relas.append(rela)
            centers.append(center)
        for i in range(len(relas)):
            if relas[i][1] != None:
                for j in range(len(relas)):
                    if relas[j][0] == relas[i][1]:
                        if polygons[i] is not None and polygons[j] is not None:
                            min_i, min_o = __find_min_point(polygons[i],
                                                            polygons[j])
                            
                            polygons[i] = __change_list(polygons[i], min_i)
                            polygons[j] = __change_list(polygons[j], min_o)
                            
                            if min_i != -1 and len(polygons[i]) > 0:
                                polygons[j].extend(polygons[i])
                            polygons[i] = None
        polygons = list(filter(None, polygons))
        return polygons, centers
    else:
        # print("No label range, can't generate bounds")
        return None, None


def __change_list(polygons, idx):
    if idx == -1:
        return polygons
    s_p = polygons[:idx]
    polygons = polygons[idx:]
    polygons.extend(s_p)
    polygons.append(polygons[0])  # closed circle
    return polygons


def __find_min_point(i_list, o_list):
    min_dis = 1e7
    idx_i = -1
    idx_o = -1
    for i in range(len(i_list)):
        for o in range(len(o_list)):
            dis = math.sqrt((i_list[i][0] - o_list[o][0])**2 + (i_list[i][
                1] - o_list[o][1])**2)
            if dis <= min_dis:
                min_dis = dis
                idx_i = i
                idx_o = o
    return idx_i, idx_o


# Calculate the angle based on the coordinates of the three points
def __cal_ang(p1, p2, p3):
    eps = 1e-12
    a = math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] -
                                                                         p3[1]))
    b = math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] -
                                                                         p3[1]))
    c = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] -
                                                                         p2[1]))
    ang = math.degrees(math.acos(
        (b**2 - a**2 - c**2) / (-2 * a * c + eps)))  # p2对应
    return ang


# Calculate the distance between two points
def __cal_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Calculate the center points of the contour
def __cal_center(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)
    

# Boundary point simplification
def approx_poly_DIY(contour, min_dist=10, ang_err=5):
    # print(contour.shape)  # N, 1, 2
    cs = [contour[i][0] for i in range(contour.shape[0])]
    ## 1. First delete the points that are close to the angle between the two adjacent points and the two points before and after
    i = 0
    while i < len(cs):
        try:
            j = (i + 1) if (i != len(cs) - 1) else 0
            if __cal_dist(cs[i], cs[j]) < min_dist:
                last = (i - 1) if (i != 0) else (len(cs) - 1)
                next = (j + 1) if (j != len(cs) - 1) else 0
                ang_i = __cal_ang(cs[last], cs[i], cs[next])
                ang_j = __cal_ang(cs[last], cs[j], cs[next])
                # print(ang_i, ang_j)  # The angle value is -180 to +180
                if abs(ang_i - ang_j) < ang_err:
                    # delete two points less than two points away
                    dist_i = __cal_dist(cs[last], cs[i]) + __cal_dist(cs[i],
                                                                      cs[next])
                    dist_j = __cal_dist(cs[last], cs[j]) + __cal_dist(cs[j],
                                                                      cs[next])
                    if dist_j < dist_i:
                        del cs[j]
                    else:
                        del cs[i]
                else:
                    i += 1
            else:
                i += 1
        except:
            i += 1
    ## 2. Then delete the points with an included angle close to 180 degrees
    i = 0
    while i < len(cs):
        try:
            last = (i - 1) if (i != 0) else (len(cs) - 1)
            next = (i + 1) if (i != len(cs) - 1) else 0
            ang_i = __cal_ang(cs[last], cs[i], cs[next])
            if abs(ang_i) > (180 - ang_err):
                del cs[i]
            else:
                i += 1
        except:
            # i += 1
            del cs[i]
    res = np.array(cs).reshape([-1, 1, 2])
    return res
