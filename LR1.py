import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt


def get_sum_x(matrix, n, m):
    result = []
    for i in range(n):
        sum = 0
        for j in range(m):
            sum += matrix[i][j]
        result.append(sum)
    return result


def get_sum_y(matrix, n, m):
    result = []
    for j in range(m):
        sum = 0
        for i in range(n):
            sum += matrix[i][j]
        result.append(sum)
    return result


def generate_value(matrix, n, m):
    x = np.random.uniform()
    p_line = make_probabilities_line(matrix)
    pos = 0
    if x <= p_line[0]:
        result = pos
    pos += 1
    while pos < len(p_line):
        if p_line[pos - 1] < x <= p_line[pos]:
            result = pos
            break
        pos += 1
    n = result // m
    m = result - n * m
    n += 1
    m += 1
    return n, m


def make_probabilities_line(matrix):
    p_line = []
    for i in range(len(matrix)):
        p_line.extend(matrix[i])
    sum = 0
    result = []
    for i in range(len(p_line)):
        sum += p_line[i]
        result.append(sum)
    return result


def get_appearances(values, n, m):
    x_appearances = []
    for j in range(n):
        x_appearances.append(0)
    y_appearances = []
    for i in range(m):
        y_appearances.append(0)
    for value in values:
        x_appearances[value[0] - 1] += 1
        y_appearances[value[1] - 1] += 1
    return x_appearances, y_appearances


def histogram(appearances, n, expected_probabilities):
    indexes = []
    for i in range(n):
        indexes.append(i + 1)
    plt.bar(indexes, appearances)
    plt.title(expected_probabilities)
    plt.show()


def get_data(matrix, n, m):
    x = get_sum_x(matrix, n, m)
    y = get_sum_y(matrix, n, m)
    mx = 0
    mx2 = 0
    for i in range(n):
        mx += (i + 1) * x[i]
        mx2 += (i + 1) ** 2 * x[i]
    my = 0
    my2 = 0
    for i in range(m):
        my += (i + 1) * y[i]
        my2 += (i + 1) ** 2 * y[i]
    dx = mx2 - mx ** 2
    dy = my2 - my ** 2
    mxy = 0
    for i in range(n):
        for j in range(m):
            mxy += (i + 1) * (j + 1) * matrix[i][j]
    cov = mxy - mx * my
    rxy = cov / math.sqrt(dx * dy)
    return mx, my, dx, dy, rxy


def normalize_matrix(values, values_amount, n, m):
    normalization_value = 1 / values_amount
    matrix = []
    for i in range(n):
        line = []
        for j in range(m):
            line.append(0)
        matrix.append(line)
    for value in values:
        matrix[value[0] - 1][value[1] - 1] += 1
    for i in range(n):
        for j in range(m):
            matrix[i][j] *= normalization_value
    return matrix


def get_m_intervals(values):
    a = 0.05
    values_amount = len(values)
    x, y, sx, sy = get_interval_data(values)
    x_delta = sx * stats.t.ppf((2 - a) / 2, 10)
    x_delta /= math.sqrt(values_amount - 1)
    min_x = x - x_delta
    max_x = x + x_delta
    y_delta = sy * stats.t.ppf((2 - a) / 2, 10)
    y_delta /= math.sqrt(values_amount - 1)
    min_y = y - y_delta
    max_y = y + y_delta
    return min_x, max_x, min_y, max_y


def get_d_intervals(values):
    a = 0.05
    values_amount = len(values)
    x, y, sx, sy = get_interval_data(values)

    min_x = values_amount * sx
    min_x /= stats.chi2.isf(df=values_amount, q=a / 2)
    max_x = values_amount * sx
    max_x /= stats.chi2.isf(df=values_amount, q=1 - a / 2)

    min_y = values_amount * sy
    min_y /= stats.chi2.isf(df=values_amount, q=a / 2)
    max_y = values_amount * sy
    max_y /= stats.chi2.isf(df=values_amount, q=1 - a / 2)
    return min_x, max_x, min_y, max_y


def get_interval_data(values):
    values_amount = len(values)
    y_values = []
    x_values = []
    for value in values:
        y_values.append(value[1])
        x_values.append(value[0])
    x, y, sx, sy = 0, 0, 0, 0
    for i in range(values_amount):
        x += x_values[i]
        y += y_values[i]
    x /= values_amount
    y /= values_amount
    for i in range(values_amount):
        sx += (x_values[i] - x) ** 2
        sy += (y_values[i] - y) ** 2
    sx /= values_amount - 1
    sy /= values_amount - 1
    return x, y, sx, sy


def check_mises(matrix, generated_matrix, n, m):
    a = 0.05
    critical_value = 0.461
    result = 0
    m_line = make_line(matrix)
    gen_line = make_line(generated_matrix)
    m_sum = 0
    gen_sum = 0
    for i in range(n * m):
        m_sum += m_line[i]
        gen_sum += gen_line[i]
        delta = m_sum
        delta -= gen_sum
        delta **= 2
        result += delta
    result = 1 / (12 * n * m) + result
    return a, critical_value, result


def make_line(matrix):
    line = []
    for i in range(len(matrix)):
        line.extend(matrix[i])
    return line

if __name__ == '__main__':

    n = 4
    m = 5
    matrix = [[0.04, 0.06, 0.03, 0.07, 0.025],
              [0.01, 0.1, 0.055, 0.045, 0.04],
              [0.035, 0.015, 0.025, 0.075, 0.05],
              [0.05, 0.08, 0.065, 0.11, 0.02]]

    sum_x = get_sum_x(matrix, n, m)
    sum_y = get_sum_y(matrix, n, m)

    values = []
    values_amount = 10000
    for i in range(values_amount):
        result = generate_value(matrix, n, m)
        values.append(result)

    success = True

    x_appearances, y_appearances = get_appearances(values, n, m)
    print("X")
    print("numbers: " + str(x_appearances))
    print("p: " + str(sum_x))
    print('\n')

    print("Y")
    print("numbers: " + str(y_appearances))
    print("p: " + str(sum_y))
    print('\n')

    histogram(x_appearances, n, sum_x)
    histogram(y_appearances, m, sum_y)

    mx, my, dx, dy, rxy = get_data(matrix, n, m)
    print("matrix: ")
    print(np.array(matrix))
    print("M[X]: " + str(mx))
    print("M[Y]: " + str(my))
    print("D[X]: " + str(dx))
    print("D[Y]: " + str(dy))
    print("rxy: " + str(rxy))
    print('\n')

    generated_matrix = normalize_matrix(values, values_amount, n, m)

    mx, my, dx, dy, rxy = get_data(generated_matrix, n, m)
    min_m_x, max_m_x, min_m_y, max_m_y = get_m_intervals(values)
    min_d_x, max_d_x, min_d_y, max_d_y = get_d_intervals(values)
    print("generated matrix: ")
    print(np.array(matrix))
    print("M[X]: " + str(mx))

    print(str(min_m_x) + " - " + str(max_m_x))
    if min_m_x < mx < max_m_x:
        print("OK")
    else:
        print("Invalid M interval for x")
        success = False

    print("M[Y]: " + str(my))
    print(str(min_m_y) + " - " + str(max_m_y))
    if min_m_y < my < max_m_y:
        print("OK")
    else:
        print("Invalid M interval for y")
        success = False

    print("D[X]: " + str(dx))
    print(str(min_d_x) + " - " + str(max_d_x))
    if min_d_x < dx < max_d_x:
        print("OK")
    else:
        print("Invalid D interval for x")
        success = False

    print("D[Y]: " + str(dy))
    print(str(min_d_y) + " - " + str(max_d_y))
    if min_d_y < dy < max_d_y:
        print("OK")
    else:
        print("Invalid D interval for y")
        success = False

    print("rxy: " + str(rxy))
    print('\n')

    print("Mises: ")
    a, critical_value, result = check_mises(matrix, generated_matrix, n, m)
    print("critical value: " + str(critical_value))
    print("result: " + str(result))
    if result < critical_value:
        print("OK")
    else:
        print("Invalid Mises criteria")
        success = False

    if success:
        print("Valid result")
    else:
        print("Invalid result")
