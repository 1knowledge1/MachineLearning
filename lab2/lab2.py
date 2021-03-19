import cv2
import os


def detect_shape(contour):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if ar == 1:
            shape = "square"
        else:
            shape = "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    return shape


if __name__ == '__main__':
    if not os.path.exists('./tagged_images'):
        os.mkdir('./tagged_images')
    image_number = 1
    while os.path.exists('./plain_images/image_{0}.jpg'.format(image_number)):
        image = cv2.imread('./plain_images/image_{0}.jpg'.format(image_number))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("image_{0}".format(image_number), thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape_type = detect_shape(c)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
            cv2.putText(image, shape_type, (cX-15, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("image_{0}".format(image_number), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("./tagged_images/image_{0}.jpg".format(image_number), image)
        image_number += 1
