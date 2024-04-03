from numpy import cos, linspace, sin
from math import factorial
import pandas as pd
import matplotlib.pyplot as plt


# функция
def func(x: float):
    return 0.5 * x ** 2 - cos(2 * x)


# выч n-ой производной
def derv_func(x: float, n: int = 2) -> float:
    """
    Вторая-n производная функции y = 0.5x^2 - cos(2x)
    От второй производной далее, по тригонометрическим свойствам, можно считать n-ю производную.

    :param x: Точка x
    :param n: Степень производной. Считается от 2 для удобства.
              Потому что именно на 2-й производной убирается константа
    :return: Значение n-й производной функции в точке x
    :rtype: float
    """
    if n % 2 == 0:
        return (-2) * (n / 2) * cos(2 * x)
    else:
        return (-2) * ((n - 1) / 2) * sin(2 * x)


rng_pos = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
range_graph = (0.6, 1.1)


# отображение функции
def generate_points(rng: tuple[float, float], count_points: int, function) -> list[tuple[float, float]]:
    res = []
    for pos in linspace(*rng, count_points):
        res.append((pos, function(pos)))
    return res


# Генерация точек
points = generate_points(range_graph, 100, func)

# Разделение точек на отдельные списки для значений x и y
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

# Построение графика
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='func(x)', color='blue', linestyle='-')
plt.xlabel('x')
plt.ylabel('func(x)')
plt.title('График функции func(x)')
plt.grid(True)
plt.legend()
plt.show()


# реализует многочлен Лагранжа
def lagrange(bp: float, points: list[tuple[float, float]]) -> float:
    count_points = len(points)
    result = 0
    for k, point in enumerate(points):
        multiply = point[1]
        for j in range(0, k - 1 + 1):
            x = points[j][0]
            multiply *= ((bp - x) / (point[0] - x))
        for i in range(k + 1, count_points):
            x = points[i][0]
            multiply *= ((bp - x) / (point[0] - x))
        result += multiply
    return result


# возвращает норму заданной функции
def get_norm(function, rng: tuple[float, float], *args) -> float:
    return max(abs(function(linspace(*rng, num=1000), *args)))


# вычисление относительной ошибки
def rel_error(abs_er: float, norm_f: float) -> float:
    return (abs_er / norm_f) * 100


# расчет теоретической ошибки
def teor_error(count_points: int, rng: tuple[float, float]) -> float:
    return (get_norm(derv_func, rng, count_points + 1) / factorial(count_points + 1)) * (
            (rng[1] - rng[0]) ** (count_points + 1))


abs_e_mass = []
rel_e_mass = []
ter_e_mass = []
# создание узлов
for count_pts in rng_pos:
    full_points = generate_points(range_graph, count_pts, func)
    # вычисление норм
    norm = get_norm(func, range_graph)
    lag_norm = get_norm(lagrange, range_graph, full_points)
    abs_e = max(abs(lagrange(linspace(*range_graph, num=1000), full_points) - func(linspace(*range_graph, num=1000))))
    der_e = get_norm(derv_func, range_graph)
    rel_e = rel_error(abs_e, lag_norm)
    ter_e = teor_error(count_pts, range_graph)
    # списки ошибок
    abs_e_mass.append(abs_e)
    rel_e_mass.append(rel_e)
    ter_e_mass.append(ter_e)

abs_df = pd.DataFrame(abs_e_mass, rng_pos, columns=["Value"])
rel_df = pd.DataFrame(rel_e_mass, rng_pos, columns=["Value"])
ter_df = pd.DataFrame(ter_e_mass, rng_pos, columns=["Value"])
# dataframe сохраняется в .csv
total_df = pd.DataFrame({"Абсолютная ошибка ": abs_e_mass,
                         "Относительная ошибка ": rel_e_mass,
                         "Теоретическая ошибка ": ter_e_mass},
                        index=rng_pos)
total_df.to_csv("errors.csv")

abs_df.plot(title="Абсолютная ошибка ", xticks=rng_pos, xlabel="n", ylabel="Разность")
rel_df.plot(title="Относительная ошибка ", xticks=rng_pos, yticks=range(0, 100 + 1, 10), xlabel="n", ylabel="%")
ter_df.plot(title="Теоретическая ошибка ", xlabel="n", ylabel="Разность")
# для теоретической ошибки
plt.xticks(rng_pos)
plt.show()

plt.plot(func(linspace(*range_graph, num=10000)))
plt.xticks([i for i in range(0, 10 ** 3 + 1, 200)], [0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
plt.show()
