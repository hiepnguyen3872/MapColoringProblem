from preprocessImage import PreprocessImage  # import class PreprocessImage
from GeneticAlgorithm import GA_coloringmap
from mrv import run_mrv
from csp_ac3 import Run
import time

if __name__ == "__main__":
    # Khởi tạo biến preImg với đường dẫn của ảnh cần tô màu
    preImg = PreprocessImage("img\\usa.jpg")

    preImg.img_2_matrix()  # Chuyển đổi hình ảnh thành ma trận kề
    matrix = preImg.get_adjacency_matrix()  # Lấy ma trận kề sau khi chuyển đổi

    colors = ['red', 'green', 'blue', 'yellow']  # list of colors

    t1 = time.time()
    output1 = GA_coloringmap(matrix, colors)  # Genetic algorithm
    t2 = time.time()
    output2 = run_mrv(matrix, colors)  # Forward checking
    t3 = time.time()
    output3 = Run(matrix, colors)  # CSP_AC3
    t4 = time.time()

    # tô màu lại bản đồ theo kết quả của thuật toán
    preImg.colorize_map("result\\1.png", output1)
    preImg.colorize_map("result\\2.png", output2)
    preImg.colorize_map("result\\3.png", output3)

    print('Thuat toan 1: {}'.format(t2-t1))
    print('Thuat toan 2: {}'.format(t3-t2))
    print('Thuat toan 3: {}'.format(t4-t3))
