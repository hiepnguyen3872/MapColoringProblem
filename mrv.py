import numpy as np


class Region:
    def __init__(self, degree, color):
        self.degree = degree
        self.color = -1
        self.neighbourRegion = []
        self.numberOfPossibleColors = len(color)
        self.possibleColors = [True]*self.numberOfPossibleColors
        self.changed = []

    def setColor(self, color):
        if self.possibleColors[color] == False:
            return False
        # for i in self.neighbourRegion:
        #     if i not in visitedRegion:
        #         if (listOfRegion[i].numberOfPossibleColors == 1) and (listOfRegion[i].possibleColors[color] == True):
        #             return False
        self.color = color
        for i in self.neighbourRegion:
            if i not in visitedRegion:
                if listOfRegion[i].possibleColors[color] == True:
                    listOfRegion[i].possibleColors[color] = False
                    listOfRegion[i].numberOfPossibleColors -= 1
                    self.changed.append(i)
        return True

    def restore(self, color):
        for i in self.changed:
            listOfRegion[i].possibleColors[color] = True
            listOfRegion[i].numberOfPossibleColors += 1
        self.changed.clear()
        self.color = -1

    def addNeibourRegion(self, l):
        self.neighbourRegion.append(l)

    def removeColor(self, c):
        self.possibleColors.remove(c)

    def __lt__(self, other):
        if self.numberOfPossibleColors > other.numberOfPossibleColors:
            return True
        elif self.numberOfPossibleColors == other.numberOfPossibleColors:
            if self.degree < other.degree:
                return True
        return False


def setRegion():
    length = len(matrix)
    for i in range(length):
        listOfRegion.append(Region(np.sum(matrix[i]), colors))
        for j in range(length):
            if matrix[i][j] == 1:
                listOfRegion[i].addNeibourRegion(j)


def checkContrain(x, color):
    for i in listOfRegion[x].neighbourRegion:
        if color == listOfRegion[i].color:
            return False
    return True


def findNextRegion():
    flag = True
    nextRegion = -1
    for x in range(len(listOfRegion)):
        if x not in visitedRegion:
            if flag:
                nextRegion = x
                flag = False
            else:
                if listOfRegion[nextRegion] < listOfRegion[x]:
                    nextRegion = x
    return nextRegion


def mrv(x):
    flag = True
    if x == -1:
        print("-------------------------------")
        for i in range(len(listOfRegion)):
            print("Vertex {} has color {}".format(
                i, colors[listOfRegion[i].color]))
            ans.append(listOfRegion[i].color)
        return True
    for color in range(len(colors)):
        if checkContrain(x, color):
            if listOfRegion[x].setColor(color):
                print('Vertex {} -> {}'.format(x, colors[color]))
                flag = False
                visitedRegion.add(x)
                y = findNextRegion()
                if mrv(y):
                    return True
                visitedRegion.remove(x)
                listOfRegion[x].restore(color)
    if flag:
        print('Vertex {} -> No suitable color'.format(x))
    return False


def main():
    setRegion()
    x = findNextRegion()
    if mrv(x) == False:
        print("-------------------------------")
        print('No solution')
    print("-------------------------------")


def run_mrv(_matrix, _colors):
    global matrix
    global colors
    global listOfRegion
    global visitedRegion
    global ans
    matrix = _matrix
    colors = _colors
    listOfRegion = []
    visitedRegion = set()
    ans = []
    main()
    return ans


# m = [[0, 1, 1, 0, 0, 0, 1],
#      [1, 0, 0, 1, 0, 0, 0],
#      [1, 0, 0, 1, 1, 1, 1],
#      [0, 1, 1, 0, 1, 0, 0],
#      [0, 0, 1, 1, 0, 1, 1],
#      [0, 0, 1, 0, 1, 0, 1],
#      [1, 0, 1, 0, 1, 1, 0]]

# c = ['red', 'green', 'blue', 'yellow']
# listOfRegion = []
# visitedRegion = set()
# print(run(m, c))
