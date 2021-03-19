import cv2
import numpy as np
from enum import Enum
import random
import os


class Figure(Enum):
    TRIANGLE = 1
    SQUARE = 2
    RECTANGLE = 3
    PENTAGON = 4
    CIRCLE = 5


def get_rand_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    if r < 25 and g < 25 and b < 25:
        r = random.randint(75, 255)
    return r, g, b


def place_shapes(img, height, width):
    min_radius = 25
    nx = ny = 4
    area_width = width // nx
    area_height = height // ny
    for i in range(ny):
        for j in range(nx):
            min_x = j * area_width
            max_x = (j + 1) * area_width
            min_y = i * area_height
            max_y = (i + 1) * area_height
            choice = random.choice(list(Figure))
            if choice == Figure.TRIANGLE:
                p1_x = random.randint(min_x, min_x + area_width // 2 - min_radius)
                p1_y = random.randint(min_y, min_y + area_height // 2 - (min_radius // 3))
                p2_x = random.randint(min_x + area_width // 2 + min_radius, max_x)
                p2_y = random.randint(min_y, min_y + area_height // 2 - (min_radius // 3))
                p3_x = random.randint(min_x + area_width // 2 - min_radius, min_x + area_width // 2 + min_radius)
                p3_y = random.randint(min_y + 3 * (area_height // 4), max_y)
                # p3_x = random.randint(min_x, max_x)
                # p3_y = random.randint(min_y + area_height // 2, max_y)
                # p1_x = min_x + 50
                # p1_y = min_y + 112
                # p2_x = min_x + 50
                # p2_y = min_y + 75
                # p3_x = min_x + 100
                # p3_y = min_y
                create_triangle(img, [(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y)], get_rand_color())
            elif choice == Figure.SQUARE:
                length = random.randint(2 * min_radius, area_height // np.sqrt(2) - 2)
                center_x = random.randint(min_x + area_width // 2, max_x - area_width // 2)
                center_y = random.randint(min_y + area_height // 2, max_y - area_height // 2)
                create_square(img, (center_x, center_y), length, random.randint(0, 89), get_rand_color())
            elif choice == Figure.RECTANGLE:
                width = random.randint(2 * min_radius, area_width // np.sqrt(2) - 2)
                height = random.randint(2 * min_radius, area_height // np.sqrt(2) - 2)
                center_x = random.randint(min_x + area_width // 2, max_x - area_width // 2)
                center_y = random.randint(min_y + area_height // 2, max_y - area_height // 2)
                create_rectangle(img, (center_x, center_y), height, width, random.randint(0, 89), get_rand_color())
            elif choice == Figure.PENTAGON:
                radius = random.randint(min_radius, (area_width // 2) - 2)
                center_x = random.randint(min_x + radius, max_x - radius)
                center_y = random.randint(min_y + radius, max_y - radius)
                create_pentagon(img, (center_x, center_y), radius, random.randint(0, 89), get_rand_color())
            elif choice == Figure.CIRCLE:
                radius = random.randint(min_radius, (area_width // 2) - 2)
                center_x = random.randint(min_x + radius, max_x - radius)
                center_y = random.randint(min_y + radius, max_y - radius)
                create_circle(img, (center_x, center_y), radius, get_rand_color())


def create_circle(img, center, radius, color, thickness=-1):
    cv2.circle(img, center, radius, color, thickness)


def create_rectangle(img, center, height, width, angle, color, thickness=-1):
    box = cv2.boxPoints((center, (height, width), angle))
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, color, thickness)


def create_square(img, center, length, angle, color, thickness=-1):
    create_rectangle(img, center, length, length, angle, color, thickness)


def create_triangle(img, points, color, thickness=-1):
    triangle = np.array(points)
    cv2.drawContours(img, [triangle], 0, color, thickness)


def create_pentagon(img, center, radius, angle, color, thickness=-1):
    n = 5
    points = list()
    for i in range(n):
        x = center[0] + radius * np.cos(angle + (2 * i * np.pi) / n)
        y = center[1] + radius * np.sin(angle + (2 * i * np.pi) / n)
        points.append((round(x), round(y)))
    pentagon = np.array(points)
    cv2.drawContours(img, [pentagon], 0, color, thickness)


def generate_images(number, image_height, image_width):
    if not os.path.exists('./plain_images'):
        os.mkdir('./plain_images')
    for i in range(number):
        image = np.zeros((image_height, image_width, 3), np.uint8)
        place_shapes(image, image_height, image_width)
        cv2.imwrite("./plain_images/image_{0}.jpg".format(i + 1), image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    MAX_NUMBER_OF_PICTURE = 20
    IMAGE_HEIGHT = 600
    IMAGE_WIDTH = 600

    correct_input = False
    while not correct_input:
        print('Введите количество изображений (1 - {0}): '.format(MAX_NUMBER_OF_PICTURE), end='')
        number_of_picture = input()
        if number_of_picture.isdigit():
            number_of_picture = int(number_of_picture)
            if number_of_picture and number_of_picture <= MAX_NUMBER_OF_PICTURE:
                generate_images(number_of_picture, IMAGE_HEIGHT, IMAGE_WIDTH)
                correct_input = True
            else:
                print('Число должно находиться в интервале от 1 до {0}.'.format(MAX_NUMBER_OF_PICTURE))
        else:
            print('Некорректный ввод. Введите число от 1 до {0}.'.format(MAX_NUMBER_OF_PICTURE))
