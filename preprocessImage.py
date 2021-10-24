import cv2
import numpy as np
import sys
import time
from matplotlib import pyplot as plt


class PreprocessImage:
    def __init__(self, map_image_path):
        self.MAP_IMAGE_PATH = map_image_path
        self.ESCAPE_KEY_CHARACTER = 27
        self.NO_COLOR = -1
        self.NOT_MARKED = -1
        self.BACKGROUND_MARK = -2
        self.SLEEP_TIME_IN_MILLISECONDS = 100
        self.MINIMUM_BORDER_WIDTH_RATIO = 0.0001  # 0.15
        self.IMPORTANT_COLOR_HIGH_THRESHOLD = 256 - 35
        self.IMPORTANT_COLOR_LOW_THRESHOLD = 35
        self.MINIMUM_REGION_AREA_RATIO = 0.0005
        self.MAXIMUM_NEIGHBOR_PIXEL_COLOR_DIFFERENCE = 50
        self.INF = 10 ** 30
        self.MAXIMUM_NUMBER_OF_REGIONS = 1000
        self.COLORING_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0)]
        self.DX = [-1, +1, 0, 0]
        self.DY = [0, 0, -1, +1]
        self.SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.MAXIMUM_IMAGE_WIDTH = 1000
        self.MAXIMUM_IMAGE_HEIGHT = 1000
        self.image = cv2.imread(self.MAP_IMAGE_PATH, cv2.IMREAD_COLOR)
        self.height = len(self.image)
        self.width = len(self.image[0])
        if self.width > self.MAXIMUM_IMAGE_WIDTH or self.height > self.MAXIMUM_IMAGE_HEIGHT:
            print("Error: please specify an image with smaller dimensions.")
            exit(0)
        self.total_area = self.width * self.height
        self.mark = [[self.NOT_MARKED for i in range(self.width)] for j in range(self.height)]
        self.mark = np.array(self.mark)
        self.nodes = []
        self.regions = [[] for i in range(self.MAXIMUM_NUMBER_OF_REGIONS)]
        self.regions_border = [[] for i in range(self.MAXIMUM_NUMBER_OF_REGIONS)]
        self.nodes_color = [self.NO_COLOR for i in range(self.MAXIMUM_NUMBER_OF_REGIONS)]

    class Node:
        def __init__(self, node_id, node_x, node_y):
            self.id = node_id
            self.x = node_x
            self.y = node_y
            self.adj = []

        def add_edge(self, node):
            self.adj.append(node.id)

    def apply_threshold(self):
        gray_image = self.image[:, :, 0].astype(int) + self.image[:, :, 1].astype(int) + self.image[:, :, 2].astype(int)
        gray_image = np.array(gray_image)
        gray_image = gray_image.astype(int)

        idx_low = np.where(gray_image < self.IMPORTANT_COLOR_LOW_THRESHOLD*3)
        idx_low = np.array(idx_low)
        self.image[idx_low[0, :], idx_low[1, :]] = (255, 255, 255)
        self.mark[idx_low[0, :], idx_low[1, :]] = self.BACKGROUND_MARK # -2

        idx_hight = np.where(gray_image > self.IMPORTANT_COLOR_HIGH_THRESHOLD*3)
        idx_hight = np.array(idx_hight)
        self.image[idx_hight[0, :], idx_hight[1, :]] = (255, 255, 255)
        self.mark[idx_hight[0, :], idx_hight[1, :]] = self.BACKGROUND_MARK # -2

    def get_all_regions_pixels(self):
        for y in range(self.height):
            for x in range(self.width):
                region_mark = self.mark[y][x]
                self.regions[region_mark].append((x, y))
                if self.is_on_border(x, y):
                    self.regions_border[region_mark].append((x, y))

    # map every pixel on image to region
    def whiten_background(self):
        idx_not_marked = np.where(self.mark == self.NOT_MARKED)
        idx_not_marked = np.array(idx_not_marked)
        self.image[idx_not_marked[0, :], idx_not_marked[1, :]] = (255, 255, 255)

        idx_background = np.where(self.mark == self.NOT_MARKED)
        idx_background = np.array(idx_background)
        self.image[idx_background[0, :], idx_background[1, :]] = (255, 255, 255)

    # find node on region
    def find_graph_nodes(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.mark[y][x] == self.NOT_MARKED:
                    color_area = self.get_region_area(x, y, self.NOT_MARKED, len(self.nodes))
                    if color_area > self.MINIMUM_REGION_AREA_RATIO * self.total_area:
                        self.nodes.append(self.Node(len(self.nodes), x, y))
                    else:
                        self.get_region_area(x, y, len(self.nodes), self.NOT_MARKED)
        self.get_all_regions_pixels()

    def is_inside(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return True

    def is_on_border(self, x, y):
        if self.mark[y][x] == self.BACKGROUND_MARK:
            return False
        for k in range(4):
            x2 = x + self.DX[k]
            y2 = y + self.DY[k]
            if self.is_inside(x2, y2) and self.mark[y2][x2] == self.BACKGROUND_MARK:
                return True
        return False

    def same_pixel_colors(self, x1, y1, x2, y2):
        if not self.is_inside(x1, y1) or not self.is_inside(x2, y2):
            return False
        b1, g1, r1 = self.image[y1][x1]
        b2, g2, r2 = self.image[y2][x2]
        r1, g1, b1 = int(r1), int(g1), int(b1)
        r2, g2, b2 = int(r2), int(g2), int(b2)
        diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
        return diff <= 3 * self.MAXIMUM_NEIGHBOR_PIXEL_COLOR_DIFFERENCE

    def get_region_area(self, start_x, start_y, src_mark, dst_mark):
        if not self.is_inside(start_x, start_y) or self.mark[start_y][start_x] != src_mark:
            return 0
        color_area = 0
        queue = [(start_x, start_y)]
        self.mark[start_y][start_x] = dst_mark
        while queue:
            x, y = queue.pop(0)
            self.mark[y][x] = dst_mark
            color_area += 1
            for k in range(4):
                x2 = x + self.DX[k]
                y2 = y + self.DY[k]
                if self.is_inside(x2, y2) and self.mark[y2][x2] == src_mark and self.same_pixel_colors(x, y, x2, y2):
                    self.mark[y2][x2] = dst_mark
                    queue.append((x2, y2))
        return color_area

    def are_adjacent(self, node1: Node, node2: Node):
        start_x, start_y = node1.x, node1.y
        end_x, end_y = node2.x, node2.y
        min_distance_sqr = self.INF
        u = self.regions_border[self.mark[start_y][start_x]]
        u = np.array(u)

        u_0 = u[:, 0]
        u_0 = u_0.reshape(u_0.shape[0], 1)
        u_1 = u[:, 1]
        u_1 = u_1.reshape(u_1.shape[0], 1)

        v = self.regions_border[self.mark[end_y][end_x]]
        v = np.array(v)

        v_0 = v[:, 0]
        v_0 = v_0.reshape(v_0.shape[0], 1)
        v_1 = v[:, 1]
        v_1 = v_1.reshape(v_1.shape[0], 1)

        v_0_matrix = np.dot(np.ones([u_0.shape[0], 1]), v_0.T)
        v_1_matrix = np.dot(np.ones([u_1.shape[0], 1]), v_1.T)
        diff_matrix = (u_0-v_0_matrix)*(u_0-v_0_matrix) + (u_1-v_1_matrix)*(u_1-v_1_matrix)
        min_distance_sqr = diff_matrix.min()
        idx = np.where(diff_matrix == min_distance_sqr)
        idx = np.array(idx)
        start_x, start_y = int(u[idx[0][0], 0]), int(u[idx[0][0], 1])
        end_x, end_y = int(v[idx[1][0], 0]), int(v[idx[1][0], 1])

        dx, dy = end_x - start_x, end_y - start_y
        if abs(dx) + abs(dy) <= 1:
            return True
        dx, dy = float(dx), float(dy)
        border_width_threshold = self.MINIMUM_BORDER_WIDTH_RATIO * (self.width * self.width + self.height * self.height)
        if min_distance_sqr >= border_width_threshold:
            return False
        total_steps = int(2 * ((self.width * self.width + self.height * self.height) ** 0.5))
        for i in range(total_steps):
            x = int(start_x + i * dx / total_steps + 0.5)
            y = int(start_y + i * dy / total_steps + 0.5)
            if self.mark[y][x] >= 0 and (x != start_x or y != start_y) and (x != end_x or y != end_y):
                return False
        return True

    def add_graph_edges(self):
        i = 0
        while i < len(self.nodes):
            j = i + 1
            while j < len(self.nodes):
                if self.are_adjacent(self.nodes[i], self.nodes[j]):
                    self.nodes[i].add_edge(self.nodes[j])
                    self.nodes[j].add_edge(self.nodes[i])
                j += 1
            i += 1

    def change_region_color(self, node: Node, pixel_color):
        region_idx = self.mark[node.y][node.x]
        mean_x = 0
        mean_y = 0
        i = 0
        while i < len(self.regions[region_idx]):
            x = self.regions[region_idx][i][0]
            y = self.regions[region_idx][i][1]
            mean_x += x
            mean_y += y
            self.image[y][x] = pixel_color
            i += 1
        mean_x /= len(self.regions[region_idx])
        mean_y /= len(self.regions[region_idx])
        cv2.putText(self.image, str(node.id), (int(mean_x), int(mean_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)

    def colorize_map(self, path, best_solution):
        for i in range(len(self.nodes)):
            self.change_region_color(self.nodes[i], self.COLORING_COLORS[best_solution[i]])
            pass
        cv2.imwrite(path, self.image)

    def img_2_matrix(self):
        self.apply_threshold()
        self.image = cv2.medianBlur(self.image, 3)
        self.apply_threshold()
        self.image = cv2.filter2D(self.image, -1, self.SHARPEN_KERNEL)
        self.apply_threshold()
        self.find_graph_nodes()
        self.add_graph_edges()
        self.whiten_background()
        # self.colorize_map(0)

    def get_adjacency_matrix(self):
        self.adj_matrix = np.zeros([len(self.nodes), len(self.nodes)])
        ids = list(range(len(self.nodes)))
        for i in ids:
            self.adj_matrix[i][self.nodes[i].adj] = 1
        np.savetxt('text1.txt', self.adj_matrix, fmt='%.2f')
        return self.adj_matrix


if __name__ == "__main__":
    print('Please wait for preprocessing...')
    start = time.time()
    preImg = PreprocessImage("Images/us2.jpg")
    preImg.img_2_matrix()
    end = time.time()
    print('Preprocessing finished.')
    print("number node: ", len(preImg.nodes))
    for node in preImg.nodes:
        print("node: ", node.id)
        for adj in node.adj:
            print(adj)
        print("--" * 10)
    print("time execute: ", (end - start))
    print("matrix", preImg.get_adjacency_matrix())
