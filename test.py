from preprocessImage import PreprocessImage # import class PreprocessImage
from GeneticAlgorithm import GeneticAlgorithm
import time

if __name__ == "__main__":
    preImg = PreprocessImage("usa.png") # Khởi tạo biến preImg với đường dẫn của ảnh cần tô màu
    preImg.img_2_matrix() # Chuyển đổi hình ảnh thành ma trận kề
    matrix = preImg.get_adjacency_matrix() # Lấy ma trận kề sau khi chuyển đổi
    genetic = GeneticAlgorithm(matrix) # dùng thuật toán để tô màu trên ma trận
    output = genetic.genetic_algorithm() # lấy về kết quả của thuật toán
    preImg.colorize_map("result.png", output) # tô màu lại bản đồ theo kết quả của thuật toán