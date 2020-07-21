import numpy as np
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
import copy
import random
import shutil
import os
import urllib.request as request

class Generator:

    def __init__(self):
        self.download_resource()
        self.sg = SequenceGenerator()
        self.rm = RadicalMaker()


    def download_resource(self):

        if os.path.exists('./resource'):
            if os.path.exists('./resource/font/mingliub.ttc') and \
                    os.path.exists('./resource/font/simsun.ttc') and \
                    os.path.exists('./resource/ids.txt') and \
                    os.path.exists('./resource/radicals.txt'):
                print("Resource Exist!")
                return
            else:
                shutil.rmtree('./resource')
        else:
            os.makedirs('./resource/font')



        print("Download resource...")
        request.urlretrieve('https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/resource/ids.txt', './resource/ids.txt')
        request.urlretrieve('https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/resource/radicals.txt', './resource/radicals.txt')
        request.urlretrieve('https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/resource/font/mingliub.ttc', './resource/font/mingliub.ttc')
        request.urlretrieve('https://fudan-ocr.oss-cn-shanghai.aliyuncs.com/resource/font/simsun.ttc', './resource/font/simsun.ttc')

    def help(self):
        print("Usage of generator: \n"
              "- root: Where to save the generated images and label (str) default:'./image'\n"
              "- num: The number of images you want to generate (int) default: -1\n"
              "- color: Color of character (tuple) default: (255,0,0)\n"
              "- max_length: Max length of the characters in an image (int) default: 4\n"
              "- compress: Whether to compress each character (bool) default: True\n")

    def generate(self,root='./image',num=-1,color=(255,0,0),max_length=4,compress=True):

        if not os.path.exists(root):
            os.mkdir(root)
            f_record = open(os.path.join(root,'label.txt'),'w+',encoding='utf-8')
        else:
            shutil.rmtree(root)
            os.mkdir(root)
            f_record = open(os.path.join(root,'label.txt'),'w+',encoding='utf-8')


        cnt = 0
        while True:
            # 判断是否结束
            if cnt == num:
                print("Finish! {0} images were saved in {1}".format(num, root))
                return

            # 随机选定长度
            length = random.randint(1,max_length)
            # print("Length {0}".format(length))
            img_total = None
            imgs = []
            label = []

            for i in range(length):
                self.rm.clean_waiting_list()
                sequence, mode, label_sequence = self.sg.generate()
                # print(sequence)
                label.extend(label_sequence)
                label.extend(['<eos>'])

                self.rm.set_color(color)
                img = self.rm.make_composed_word(sequence)
                if compress:
                    img = self.rm.compress(img)

                imgs.append(img)
                if img_total is None:
                    img_total = img
                else:
                    # 加上随机间距
                    gap = np.zeros((img_total.shape[0],random.randint(20,80),3))
                    img_total = np.concatenate((img_total, gap), 1)

                    img_total = np.concatenate((img_total, img), 1)
            label.extend(['<stop>'])
            label_str = ' '.join(label)
            f_record.write('{0}.png {1}\n'.format(cnt,label_str))


            h,w, _ = img_total.shape
            png_img = np.zeros((h,w,4))
            for i in range(h):
                for j in range(w):
                    if (img_total[i][j] == [0,0,0]).all():
                        # print('hi')
                        png_img[i][j] = [0, 0, 0,0]
                    else:
                        a,b,c = img_total[i][j]
                        png_img[i][j] = [a,b,c,255]

            cv2.imwrite('image/{0}.png'.format(cnt), png_img)
            print("{0}.png saved! Sequence: {1} ".format(cnt,label_str))


            cnt += 1

class SequenceGenerator:

    def __init__(self):
        self.structure = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷⿳⿲'
        self.structure_two_easy = '⿰⿱'
        self.structure_two_hard = '⿻⿸⿴⿺⿵⿹⿶⿷'
        self.structure_two = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷'
        self.structure_three = '⿳⿲'

    def get_template(self):
        f = open('./resource/ids.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]

        template = []
        for line in lines:
            temp = []
            line = line[2:].split()
            for i in line:
                if i in self.structure:
                    if i in '⿲⿳':
                        temp.append(3)
                    else:
                        temp.append(2)
                else:
                    temp.append(1)

            # set
            # if temp not in template:
            # 按权重
            if len(temp) < 8 and len(temp) >= 2:
                template.append(temp)
        return template

    def get_rely(self):
        f = open('./resource/ids.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]

        rely_dict = {}
        single = []

        for line in lines:
            stack = []
            sequence = line[2:].split()

            for i in sequence:

                # 独体字
                if len(sequence) == 1:
                    single.append(sequence)
                    continue

                if i in self.structure:
                    if stack != []:
                        stack[-1][1] = stack[-1][1] - 1
                        if stack[-1][1] == 0:
                            stack.pop()

                        if i in self.structure_two:
                            stack.append([i, 2])
                        else:
                            stack.append([i, 3])
                    else:
                        if i in self.structure_two:
                            stack.append([i, 2])
                        else:
                            stack.append([i, 3])
                else:
                    # rely = (stack[-1][0],stack[-1][1])
                    # print("stack",stack,sequence)
                    stack[-1][1] = stack[-1][1] - 1

                    if (stack[-1][0], stack[-1][1]) not in rely_dict:
                        rely_dict[(stack[-1][0], stack[-1][1])] = []

                    if i in rely_dict[(stack[-1][0], stack[-1][1])]:
                        pass
                    else:
                        if len(i) == 1:
                            rely_dict[(stack[-1][0], stack[-1][1])].append(i)

                    if stack[-1][1] == 0:
                        stack.pop()

        return rely_dict

    def get_radical_frequency(self):
        f = open('./resource/ids.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]

        dict = {}

        for line in lines:
            sequence = line[2:].split()
            for i in sequence:
                if i not in dict:
                    dict[i] = 1
                else:
                    dict[i] += 1

        dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        return dict

    def get_hard_word(self,rely_dict):

        f = open('./resource/ids.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]

        hard_word = []

        for line in lines:
            word = line[0]
            sequence = line[2:].split()

            for i in self.structure_two_hard:
                if i in sequence:
                    hard_word.append([word,sequence])
                    break

        # ⿰⿱
        hard_word,hard_sequence = hard_word[random.randint(0,len(hard_word)-1)]

        choice = random.randint(0,3)
        if choice == 0:
            rely_radical = rely_dict[('⿱',0)]
            length = len(rely_radical)
            rely_choice = random.randint(0, length - 1)
            paint_sequence = ['⿱',hard_word,rely_radical[rely_choice]]
            label_sequence = ['⿱',*hard_sequence,rely_radical[rely_choice]]
        if choice == 1:
            rely_radical = rely_dict[('⿱', 1)]
            length = len(rely_radical)
            rely_choice = random.randint(0, length - 1)
            paint_sequence = ['⿱', rely_radical[rely_choice],hard_word]
            label_sequence = ['⿱', rely_radical[rely_choice],*hard_sequence]
        if choice == 2:
            rely_radical = rely_dict[('⿰', 0)]
            length = len(rely_radical)
            rely_choice = random.randint(0, length - 1)
            paint_sequence = ['⿰', hard_word, rely_radical[rely_choice]]
            label_sequence = ['⿰', *hard_sequence, rely_radical[rely_choice]]
        if choice == 3:
            rely_radical = rely_dict[('⿰', 1)]
            length = len(rely_radical)
            rely_choice = random.randint(0, length - 1)
            paint_sequence = ['⿰', rely_radical[rely_choice],hard_word]
            label_sequence = ['⿰', rely_radical[rely_choice],*hard_sequence]


        return (paint_sequence,'hard',label_sequence)



    def generate(self):

        # # 统计模板长度的频数
        # dict = {}
        # for i in template:
        #     length = len(i)
        #     if length not in dict:
        #         dict[length] = 1
        #     else:
        #         dict[length] += 1
        #
        # dict = sorted(dict.items(), key=lambda x: x[1],reverse=True)
        # print(dict)

        template = self.get_template()
        rely_dict = self.get_rely()

        # 选择简单模式或困难模式
        mode = random.randint(0,2)
        if mode == 2:
            # 使用困难模式的符号
            sequence = self.get_hard_word(rely_dict)
            return sequence



        # 随机搜索一个模板
        template_length = len(template)
        choice = random.randint(0, template_length)
        template_chosen = template[choice]
        # print(template_chosen)

        sequence = []

        for i in template_chosen:
            if i == 2:
                choice = random.randint(0, 1)
                sequence.append(self.structure_two_easy[choice])
            elif i == 3:
                choice = random.randint(0, 1)
                sequence.append(self.structure_three[choice])
            else:
                '''
                radical的选择是有讲究的，有些radical必须依赖于特定的偏旁部首
                '''
                sequence.append(1)

        stack = []
        for index, i in enumerate(sequence):
            if str(i) in self.structure:
                if stack != []:
                    stack[-1][1] = stack[-1][1] - 1
                    if stack[-1][1] == 0:
                        stack.pop()

                    if i in self.structure_two:
                        stack.append([i, 2])
                    else:
                        stack.append([i, 3])
                else:
                    if i in self.structure_two:
                        stack.append([i, 2])
                    else:
                        stack.append([i, 3])
            else:
                # rely = (stack[-1][0],stack[-1][1])
                # print("stack",stack,sequence)
                '''
                工程
                这里有个bug
                '''
                try:
                    stack[-1][1] = stack[-1][1] - 1
                except:
                    print("error!")
                    print(sequence,stack)
                    exit(0)

                rely_radical = rely_dict[(stack[-1][0], stack[-1][1])]
                length = len(rely_radical)
                choice = random.randint(0, length - 1)
                sequence[index] = rely_radical[choice]

                if stack[-1][1] == 0:
                    stack.pop()

        return (sequence,'easy',sequence)


class RadicalMaker:

    def __init__(self,color=(0,255,0)):
        self.structure_str = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷⿳⿲'
        self.structure_str_two = '⿻⿸⿰⿴⿺⿱⿵⿹⿶⿷'
        self.structure_str_three = '⿲⿳'
        self.radicals = self.get_radicals()
        self.squash = 0.7
        self.wait_for_compress = []
        self.color = color

    def clean_waiting_list(self):
        self.wait_for_compress = []

    def set_color(self,color):
        self.color = color

    def get_radicals(self):
        f = open('./resource/radicals.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        f.close()
        return lines

    def create_new_canvas(self, size):

        canvas = np.zeros(size, dtype='uint8')
        # print("canvas.shape",canvas.shape)
        return canvas

    def get_word_image(self, word):

        def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=100, font="./resource/font/simsun.ttc"):
            if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            # mingliub.ttc simsun.ttc
            fontStyle = ImageFont.truetype(
                font, textSize, encoding="utf-8")
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
                    if (img[i][j] != [0, 0, 0]).any():
                        pixel_total_i.append(i)
                        pixel_total_j.append(j)

            minh, maxh = (min(pixel_total_i), max(pixel_total_i))
            minw, maxw = (min(pixel_total_j), max(pixel_total_j))

            return img[minh - slack:maxh + slack, minw - slack:maxw + slack]

        canvas = self.create_new_canvas((180, 180, 3))
        # print(canvas.shape)
        word_img = cv2ImgAddText(canvas, word, 100, 100, self.color, 50)
        if np.sum(word_img) == 0:
            word_img = cv2ImgAddText(canvas, word, 100, 100, self.color, 50, "./resource/font/mingliub.ttc")

        word_img_small = get_mininum_block(word_img)
        return word_img_small

    def image_add(self, img_base, img_new, left, top, right, bottom):
        # 取整操作
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        '''
        工程
        添加数据增强功能
        '''

        def rotate(image, angle, center=None, scale=1.0):
            # 获取图像尺寸
            (h, w) = image.shape[:2]

            # 若未指定旋转中心，则将图像中心设为旋转中心
            if center is None:
                center = (w / 2, h / 2)

            # 执行旋转
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))

            # 返回旋转后的图像
            return rotated

        w = right - left
        h = bottom - top
        img_new = cv2.resize(img_new, (w, h))
        # 随机旋转
        rotate_angle = random.randint(-1,1)
        img_new = rotate(img_new,rotate_angle)

        for i in range(left, right):
            for j in range(top, bottom):
                relative_i, relative_j = i - left, j - top

                # print(relative_i,relative_j)
                # 空白像素
                if (img_new[relative_j][relative_i] == [0, 0, 0]).all():
                    continue
                else:
                    img_base[j][i] = img_new[relative_j][relative_i]

        return img_base

    def sequence_is_valid(self, radical_sequence):
        '''
        传入一个部首序列，观察该序列是否是正常
        '''
        for i in radical_sequence:
            if i not in self.radicals:
                print("{0} is not radicals!".format(i))
                return False

        stack = []
        for index, i in enumerate(radical_sequence):
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

    def compress(self, canvas):

        for compress_operation in self.wait_for_compress[::-1]:
            # print(compress_operation)

            if len(compress_operation) == 3:
                symbol, coord1, coord2 = compress_operation

                coord1 = [int(i) for i in coord1]
                coord2 = [int(i) for i in coord2]
                l1, t1, r1, b1 = coord1
                l2, t2, r2, b2 = coord2

                # print(l1, t1, r1, b1)


                if symbol == '⿱':
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[b1 - expand:b1 + expand, l1:r1]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print()
                            break



                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[t1 + expand:b1, l1:r1] = canvas[t1:b1 - expand, l1:r1]
                    canvas[t1:t1 + expand, l1:r1] = [0, 0, 0]

                    canvas[b1:b2 - expand, l1:r1] = canvas[b1 + expand:b2, l1:r1]
                    canvas[b2 - expand:b2, l1:r1] = [0, 0, 0]

                elif symbol == '⿰':
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[ t1:b1,r1 - expand:r1 + expand]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print("fail")
                            break



                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[t1:b1,l1 + expand:r1] = canvas[t1:b1,l1:r1 - expand]
                    canvas[t1:b1,l1:l1 + expand] = [0, 0, 0]

                    canvas[t1:b1,r1:r2 - expand] = canvas[t1:b1,r1 + expand:r2]
                    canvas[t1:b1,r2 - expand:r2] = [0, 0, 0]

            elif len(compress_operation) == 4:
                # ⿳

                symbol, coord1, coord2, coord3 = compress_operation

                coord1 = [int(i) for i in coord1]
                coord2 = [int(i) for i in coord2]
                coord3 = [int(i) for i in coord3]
                l1, t1, r1, b1 = coord1
                l2, t2, r2, b2 = coord2
                l3, t3, r3, b3 = coord3

                # print(l1, t1, r1, b1)

                if symbol == '⿳':
                    # 上半部分
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[b1 - expand:b1 + expand, l1:r1]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print()
                            break

                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[t1 + 2*expand:b1+expand, l1:r1] = canvas[t1:b1 - expand, l1:r1]
                    canvas[t1:t1 + 2*expand, l1:r1] = [0, 0, 0]

                    # 下半部分
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[b2 - expand:b2 + expand, l1:r1]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print()
                            break

                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[b2 - expand:b3 - 2*expand, l1:r1] = canvas[b2+expand:b3, l1:r1]
                    canvas[b3-2*expand:b3, l1:r1] = [0, 0, 0]

                    # canvas[b1:b2 - expand, l1:r1] = canvas[b1 + expand:b2, l1:r1]
                    # canvas[b2 - expand:b2, l1:r1] = [0, 0, 0]

                elif symbol == '⿲':
                    # 上半部分
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[t1:b1,r1-expand:r1+expand]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print()
                            break

                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[t1:b1,l1+2*expand:r1+expand] = canvas[t1:b1,l1:r1-expand]
                    canvas[t1:b1,l1:l1+2*expand] = [0, 0, 0]

                    # 下半部分
                    expand = 0
                    while True:
                        expand += 1
                        if np.sum(canvas[t1:b1,r2-expand:r2+expand]) == 0:
                            if expand > 20:
                                break
                            continue
                        else:
                            # print()
                            break

                    expand = expand - 1
                    # print("可供压缩的空间为", expand)
                    canvas[t1:b1,r2 - expand:r3 - 2*expand] = canvas[t1:b1,r2+expand:r3]
                    canvas[t1:b1,r3-2*expand:r3] = [0, 0, 0]

                    # canvas[b1:b2 - expand, l1:r1] = canvas[b1 + expand:b2, l1:r1]
                    # canvas[b2 - expand:b2, l1:r1] = [0, 0, 0]

            else:
                print('error')
                print(compress_operation)
                exit(0)

        return canvas

    def make_composed_word(self, radical_sequence, width=500, height=500):
        # radical sequence 最好给一个list
        # assert self.sequence_is_valid(radical_sequence), "Sequence is invalid!"

        # 用栈实现
        # 作为起始画布，不断地划分成小区域
        stack = []

        left = 0
        top = 0
        right = 200
        bottom = 200
        canvas = self.create_new_canvas((200,200, 3))

        for index, i in enumerate(radical_sequence):
            if i in self.structure_str:
                energy = 0
                if i in self.structure_str_two:
                    energy = 2
                elif i in self.structure_str_three:
                    energy = 3

                # 此处也需要分割空间
                if stack == []:
                    stack.append([i, energy, left, top, right, bottom])
                elif stack[-1][0] == '⿱':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, 0.5 * now_top + 0.5 * now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left, 0.5 * now_top + 0.5 * now_bottom, now_right, now_bottom])

                        coord1 = (now_left, now_top, now_right, 0.5 * now_top + 0.5 * now_bottom)
                        coord2 = (now_left, 0.5 * now_top + 0.5 * now_bottom, now_right, now_bottom)
                        self.wait_for_compress.append(('⿱', coord1, coord2))

                elif stack[-1][0] == '⿰':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, 0.5 * now_left + 0.5 * now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * 0.5 + now_right * 0.5, now_top, now_right, now_bottom])

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, 0.5 * now_left + 0.5 * now_right, now_bottom)
                        coord2 = (now_left * 0.5 + now_right * 0.5, now_top, now_right, now_bottom)
                        self.wait_for_compress.append(('⿰', coord1, coord2))


                elif stack[-1][0] == '⿴':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * self.squash + now_right * (1 - self.squash),
                                      now_top * self.squash + now_bottom * (1 - self.squash),
                                      now_left * (1 - self.squash) + now_right * self.squash,
                                      now_top * (1 - self.squash) + now_bottom * self.squash])
                elif stack[-1][0] == '⿻':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                elif stack[-1][0] == '⿵':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * self.squash + now_right * (1 - self.squash),
                                      now_top * self.squash + now_bottom * (1 - self.squash),
                                      now_left * (1 - self.squash) + now_right * self.squash, now_bottom])
                elif stack[-1][0] == '⿶':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * self.squash + now_right * (1 - self.squash), now_top,
                                      now_left * (1 - self.squash) + now_right * self.squash,
                                      now_top * (1 - self.squash) + now_bottom * self.squash])
                elif stack[-1][0] == '⿷':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * self.squash + now_right * (1 - self.squash),
                                      now_top * self.squash + now_bottom * (1 - self.squash), now_right,
                                      now_top * (1 - self.squash) + now_bottom * self.squash])
                elif stack[-1][0] == '⿸':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * self.squash + now_right * (1 - self.squash),
                                      now_top * self.squash + now_bottom * (1 - self.squash), now_right, now_bottom])
                elif stack[-1][0] == '⿺':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append(
                            [i, energy, now_left * self.squash + now_right * (1 - self.squash), now_top, now_right,
                             now_top * (1 - self.squash) + now_bottom * self.squash])
                elif stack[-1][0] == '⿹':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top, now_right, now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left, now_top * self.squash + now_bottom * (1 - self.squash),
                                      now_right * self.squash, now_bottom])
                elif stack[-1][0] == '⿲':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 3:
                        stack[-1][1] = 2
                        stack.append([i, energy, now_left, now_top, now_left * 0.66 + now_right * 0.33, now_bottom])
                    elif stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append(
                            [i, energy, now_left * 0.66 + now_right * 0.33, now_top, now_left * 0.33 + now_right * 0.66,
                             now_bottom])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left * 0.33 + now_right * 0.66, now_top, now_right, now_bottom])

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, now_left * 0.66 + now_right * 0.33,
                                  now_bottom)
                        coord2 = (now_left * 0.66 + now_right * 0.33, now_top,
                                  now_left * 0.33 + now_right * 0.66, now_bottom)
                        coord3 = (now_left * 0.33 + now_right * 0.66, now_top,
                                  now_right, now_bottom)
                        self.wait_for_compress.append(('⿲', coord1, coord2, coord3))

                elif stack[-1][0] == '⿳':
                    now_left = stack[-1][2]
                    now_top = stack[-1][3]
                    now_right = stack[-1][4]
                    now_bottom = stack[-1][5]
                    if stack[-1][1] == 3:
                        stack[-1][1] = 2
                        stack.append([i, energy, now_left, now_top, now_right, now_top * 0.66 + now_bottom * 0.33])
                    elif stack[-1][1] == 2:
                        stack[-1][1] = 1
                        stack.append([i, energy, now_left, now_top * 0.66 + now_bottom * 0.33, now_right,
                                      now_top * 0.33 + now_bottom * 0.66])
                    elif stack[-1][1] == 1:
                        stack.pop()
                        stack.append([i, energy, now_left, now_top * 0.33 + now_bottom * 0.66, now_right, now_bottom])

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, now_right,
                                  now_top * 0.66 + now_bottom * 0.33)
                        coord2 = (now_left, now_top * 0.66 + now_bottom * 0.33,
                                  now_right, now_top * 0.33 + now_bottom * 0.66)
                        coord3 = (now_left, now_top * 0.33 + now_bottom * 0.66,
                                  now_right, now_bottom)
                        self.wait_for_compress.append(('⿳', coord1, coord2, coord3))





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
                '''
                工程
                加上一些compress的任务
                '''
                if last_info[0] == '⿱':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right,
                                                int(0.5 * now_top + 0.5 * now_bottom))
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, int(0.5 * now_top + 0.5 * now_bottom),
                                                now_right, now_bottom)
                        stack.pop()

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, now_right, int(0.5 * now_top + 0.5 * now_bottom))
                        coord2 = (now_left, int(0.5 * now_top + 0.5 * now_bottom), now_right, now_bottom)
                        self.wait_for_compress.append(('⿱', coord1, coord2))

                elif last_info[0] == '⿰':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, 0.5 * now_left + 0.5 * now_right,
                                                now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left * 0.5 + now_right * 0.5, now_top, now_right,
                                                now_bottom)
                        stack.pop()

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, 0.5 * now_left + 0.5 * now_right,now_bottom)
                        coord2 = (now_left * 0.5 + now_right * 0.5, now_top, now_right,now_bottom)
                        self.wait_for_compress.append(('⿰', coord1, coord2))

                elif last_info[0] == '⿴':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash),
                                                now_top * self.squash + now_bottom * (1 - self.squash),
                                                now_left * (1 - self.squash) + now_right * self.squash,
                                                now_top * (1 - self.squash) + now_bottom * self.squash)
                        stack.pop()

                elif last_info[0] == '⿻':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack.pop()

                elif last_info[0] == '⿵':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash),
                                                now_top * self.squash + now_bottom * (1 - self.squash),
                                                now_left * (1 - self.squash) + now_right * self.squash, now_bottom)
                        stack.pop()

                elif last_info[0] == '⿶':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash), now_top,
                                                now_left * (1 - self.squash) + now_right * self.squash,
                                                now_top * (1 - self.squash) + now_bottom * self.squash)
                        stack.pop()

                elif last_info[0] == '⿷':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash),
                                                now_top * self.squash + now_bottom * (1 - self.squash), now_right,
                                                now_top * (1 - self.squash) + now_bottom * self.squash)
                        stack.pop()

                elif last_info[0] == '⿸':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash),
                                                now_top * self.squash + now_bottom * (1 - self.squash), now_right,
                                                now_bottom)
                        stack.pop()

                elif last_info[0] == '⿺':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img,
                                                now_left * self.squash + now_right * (1 - self.squash), now_top,
                                                now_right, now_top * (1 - self.squash) + now_bottom * self.squash)
                        stack.pop()

                elif last_info[0] == '⿹':
                    if last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left,
                                                now_top * self.squash + now_bottom * (1 - self.squash),
                                                now_left * (1 - self.squash) + now_right * self.squash, now_bottom)
                        stack.pop()

                elif last_info[0] == '⿲':
                    if last_info[1] == 3:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_left * 0.66 + now_right * 0.33,
                                                now_bottom)
                        stack[-1][1] = 2
                    elif last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left * 0.66 + now_right * 0.33, now_top,
                                                now_left * 0.33 + now_right * 0.66, now_bottom)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left * 0.33 + now_right * 0.66, now_top,
                                                now_right, now_bottom)
                        stack.pop()

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, now_left * 0.66 + now_right * 0.33,
                                                now_bottom)
                        coord2 = (now_left * 0.66 + now_right * 0.33, now_top,
                                                now_left * 0.33 + now_right * 0.66, now_bottom)
                        coord3 = (now_left * 0.33 + now_right * 0.66, now_top,
                                                now_right, now_bottom)
                        self.wait_for_compress.append(('⿲', coord1, coord2, coord3))

                elif last_info[0] == '⿳':
                    if last_info[1] == 3:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top, now_right,
                                                now_top * 0.66 + now_bottom * 0.33)
                        stack[-1][1] = 2
                    elif last_info[1] == 2:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top * 0.66 + now_bottom * 0.33,
                                                now_right, now_top * 0.33 + now_bottom * 0.66)
                        stack[-1][1] = 1
                    elif last_info[1] == 1:
                        word_img = self.get_word_image(i)
                        canvas = self.image_add(canvas, word_img, now_left, now_top * 0.33 + now_bottom * 0.66,
                                                now_right, now_bottom)
                        stack.pop()

                        # 加上一些压缩的操作
                        coord1 = (now_left, now_top, now_right,
                                                now_top * 0.66 + now_bottom * 0.33)
                        coord2 = (now_left, now_top * 0.66 + now_bottom * 0.33,
                                                now_right, now_top * 0.33 + now_bottom * 0.66)
                        coord3 = (now_left, now_top * 0.33 + now_bottom * 0.66,
                                                now_right, now_bottom)
                        self.wait_for_compress.append(('⿳', coord1, coord2,coord3))

        return canvas

def png_experiment():
    img = cv2.imread('image/none.png')
    print(img)



if __name__ == '__main__':
    # png_experiment()
    # exit(0)
    # rg = RadicalGenerator()
    # generate_sequence = rg.generate()
    # print(generate_sequence)
    #
    # rm = RadicalMaker()
    # # canvas = rm.create_new_canvas((500,500))
    # # print(canvas.shape)
    # # rm.sequence_is_valid(['⿱','我','口'])
    #
    # # img = rm.get_word_image('𠂎')
    # str = ' '.join(generate_sequence)
    #
    # # str = '⿲ 口 口 口'
    # lis = str = ['⿰', '⿱', '口', '⿱', '又', '木', '肃']
    # img = rm.make_composed_word(lis)
    # img_origin = copy.deepcopy(img)
    # img_compress = rm.compress(img)
    #
    # combine_img = np.concatenate((img_origin,img_compress),1)
    #
    # cv2.imshow('img', combine_img)
    # cv2.waitKey(0)

    tg = Generator()
    tg.help()
    tg.generate(num=-1,color=(255,0,0))

    # sg = SequenceGenerator()
    # result = sg.get_hard_word()
    # print(result)