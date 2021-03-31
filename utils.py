import math


def get_venn_circle_height(num):
    baseHeight = 20
    unit_a = math.pow(baseHeight / 2, 2) * math.pi / 42
    output_a = unit_a * (42 + math.pow(num - 42, 1.1))
    output_r = math.sqrt(output_a / math.pi)
    return output_r * 2
