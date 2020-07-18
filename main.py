import numpy as np
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

class RadicalMaker:

    def __init__(self):
        self.structure_str = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷⿳⿲'
        self.structure_str_two = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷'
        self.structure_str_three = '⿲⿳'
        self.radicals = self.get_radicals()

    def get_radicals(self):
        f = open('radicals.txt','r',encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        f.close()
        return lines

    def create_new_canvas(self,size):

        canvas = np.zeros(size,dtype='uint8')
        # print("canvas.shape",canvas.shape)
        return canvas

    def get_word_image(self,word):

        def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=100):
            if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype(
                "font/simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


        def get_mininum_block(img):

            slack = 10
            # print(img)
            pixel_total_i = []
            pixel_total_j = []
            w, h, c = img.shape
            for i in range(w):
                for j in range(h):
                    if (img[i][j] != [0,0,0]).any():
                        pixel_total_i.append(i)
                        pixel_total_j.append(j)

            minh, maxh = (min(pixel_total_i),max(pixel_total_i))
            minw, maxw = (min(pixel_total_j),max(pixel_total_j))

            return img[minh-slack:maxh+slack,minw-slack:maxw+slack]

        canvas = self.create_new_canvas((500,500,3))
        # print(canvas.shape)
        word_img = cv2ImgAddText(canvas, word, 100, 100, (0, 255, 0), 100)
        word_img_small = get_mininum_block(word_img)
        return word_img_small

    def image_add(self,img_base,img_new,left,top,right,bottom):
        # 取整操作
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        w = right - left
        h = bottom - top
        img_new = cv2.resize(img_new,(w,h))

        for i in range(left,right):
            for j in range(top,bottom):
                relative_i, relative_j = i - left, j - top

                # print(relative_i,relative_j)
                # 空白像素
                if (img_new[relative_j][relative_i] == [0,0,0]).all():
                    continue
                else:
                    img_base[j][i] = img_new[relative_j][relative_i]

        return img_base

    def sequence_is_valid(self,radical_sequence):
        '''
        传入一个部首序列，观察该序列是否是正常
        '''
        for i in radical_sequence:
            if i not in self.radicals:
                print("{0} is not radicals!".format(i))
                return False

        stack = []
        for index,i in enumerate(radical_sequence):
            if i in self.structure_str_two:
                if stack != []:
                    stack[-1] = stack[-1] - 1
                    if stack[-1] == 0:
                        stack.pop()
                stack.append(2)
            elif i in self.structure_str_three:
                if stack != []:
                    stack[-1] = stack[-1] - 1
                    if stack[-1] == 0:
                        stack.pop()
                stack.append(3)
            elif i == '<eos>':
                if stack != []:
                    print("Stack is not empty in index {0}'s <eos>".format(index))
                    return False
            else:
                if stack == []:
                    print("Stack is empty in index {0}".format(index))
                    return False
                stack[-1] = stack[-1] - 1
                if stack[-1] == 0:
                    stack.pop()

        if stack != []:
            print("Stack is not empty!")
            return False

        return True

    def make_composed_word(self,radical_sequence):
        # radical sequence 最好给一个list
        assert self.sequence_is_valid(radical_sequence), "Sequence is invalid!"

        # 用栈实现
        # 作为起始画布，不断地划分成小区域
        stack = []

        left = 0
        top = 0
        right = 500
        bottom = 500
        canvas = self.create_new_canvas((500, 500, 3))

        for index, i in enumerate(radical_sequence):
            if i in self.structure_str:
                energy = 0
                if i in self.structure_str_two:
                    energy = 2
                elif i in self.structure_str_three:
                    energy = 3

                # 此处也需要分割空间
                if stack == []:
                    stack.append([i,energy,left,top,right,bottom])
                elif stack[-1][0] == '⿱':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, 0.5*now_top+0.5*now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left, 0.5*now_top+0.5*now_bottom, now_right, now_bottom])
                elif stack[-1][0] == '⿰':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,0.5*now_left+0.5*now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.5+now_right*0.5,now_top,now_right,now_bottom])
                elif stack[-1][0] == '⿴':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_left*0.2+now_right*0.8,now_top*0.2+now_bottom*0.8])
                elif stack[-1][0] == '⿻':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                elif stack[-1][0] == '⿵':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_left*0.2+now_right*0.8,now_bottom])
                elif stack[-1][0] == '⿶':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top,now_left*0.2+now_right*0.8,now_top*0.2+now_bottom*0.8])
                elif stack[-1][0] == '⿷':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_right,now_top*0.2+now_bottom*0.8])
                elif stack[-1][0] == '⿸':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_right,now_bottom])
                elif stack[-1][0] == '⿺':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.8+now_right*0.2,now_top,now_right,now_top*0.2+now_bottom*0.8])
                elif stack[-1][0] == '⿹':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top,now_right,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left,now_top*0.8+now_bottom*0.2,now_right*0.8,now_bottom])
                elif stack[-1][0] == '⿲':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 3:
                        stack[-1][1] = 2
                        stack.append([i, energy, now_left,now_top,now_left*0.66+now_right*0.33,now_bottom])
                    elif stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left*0.66+now_right*0.33,now_top,now_left*0.33+now_right*0.66,now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left*0.33+now_right*0.66,now_top,now_right,now_bottom])
                elif stack[-1][0] == '⿳':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 3:
                        stack[-1][1] = 2
                        stack.append([i, energy, now_left,now_top,now_right,now_top*0.66+now_bottom*0.33])
                    elif stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left,now_top*0.66+now_bottom*0.33,now_right,now_top*0.33+now_bottom*0.66])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left,now_top*0.33+now_bottom*0.66,now_right,now_bottom])







            else:
                last_info = stack[-1]
                now_left = last_info[2]
                now_top = last_info[3]
                now_right = last_info[4]
                now_bottom = last_info[5]
                # now_left,now_top,now_right,now_bottom

                # self.structure_str = '⿻⿸⿰⿲⿴⿺⿱⿵⿹⿶⿷⿳⿲⿳'
                # self.structure_str_two = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷'
                # self.structure_str_three = '⿲⿳'

                # 在这个地方进行枚举
                if last_info[0] == '⿱':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,int(0.5*now_top+0.5*now_bottom))
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,int(0.5*now_top+0.5*now_bottom),now_right,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿰':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,0.5*now_left+0.5*now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.5+now_right*0.5,now_top,now_right,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿴':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_left*0.2+now_right*0.8,now_top*0.2+now_bottom*0.8)
                        stack.pop()

                elif last_info[0] == '⿻':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿵':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_left*0.2+now_right*0.8,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿶':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top,now_left*0.2+now_right*0.8,now_top*0.2+now_bottom*0.8)
                        stack.pop()

                elif last_info[0] == '⿷':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_right,now_top*0.2+now_bottom*0.8)
                        stack.pop()

                elif last_info[0] == '⿸':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top*0.8+now_bottom*0.2,now_right,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿺':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.8+now_right*0.2,now_top,now_right,now_top*0.2+now_bottom*0.8)
                        stack.pop()

                elif last_info[0] == '⿹':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top*0.8+now_bottom*0.2,now_left*0.2+now_right*0.8,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿲':
                    if last_info[1] == 3:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_left*0.66+now_right*0.33,now_bottom)
                        stack[-1][1] = 2
                    elif last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.66+now_right*0.33,now_top,now_left*0.33+now_right*0.66,now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left*0.33+now_right*0.66,now_top,now_right,now_bottom)
                        stack.pop()

                elif last_info[0] == '⿳':
                    if last_info[1] == 3:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top,now_right,now_top*0.66+now_bottom*0.33)
                        stack[-1][1] = 2
                    elif last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top*0.66+now_bottom*0.33,now_right,now_top*0.33+now_bottom*0.66)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,now_top*0.33+now_bottom*0.66,now_right,now_bottom)
                        stack.pop()


        return canvas

if __name__ == '__main__':
    rm = RadicalMaker()
    # canvas = rm.create_new_canvas((500,500))
    # print(canvas.shape)
    # rm.sequence_is_valid(['⿱','我','口'])
    str = '⿱ ⿰ 我 ⿱ 日 一 牛'
    lis = str.split()
    img = rm.make_composed_word(lis)
    cv2.imshow('img',img)
    cv2.waitKey(0)