# -*- conding:utf-8 -*-
import pygame
import sys

pygame.init()
# 绘制窗口
screen = pygame.display.set_mode((600, 400), 0, 32)
# 绘制背景
background = pygame.Surface(screen.get_size())
# 填充颜色
background.fill(color=(255, 255, 23))
# 创建字体对象
font = pygame.font.Font('simsun.ttc', 56)
# 文本与颜色
text = font.render("I love Python 哈哈扌", 1, (255, 10, 10))
# 获取中心的坐标
center = (background.get_width() / 2, background.get_height() / 2)
# 获取设置后新的坐标区域
textpos = text.get_rect(center=center)
print(text.get_rect(center=center))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    # 将字体填充到背景
    background.blit(text, textpos)
    # 将背景填充到窗口
    screen.blit(background, (0, 0))
    pygame.display.update()