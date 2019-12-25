import inline as inline
import matplotlib
import numpy as np

#%matplotlib inline

from matplotlib import pyplot as plt


# набор данных ( необходимо для работы sgd)
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1, -1, 1, 1, 1])


# стохастический градиентынй спуск
def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))     # инициализация вектора весов
    eta = 1                     # скорость обучения
    epochs = 20                 # количсество эпох

    for t in range(epochs):                    # итерация n раз
        for i, x in enumerate(X):              # итерация по каждой выборке
            if (np.dot(X[i], w) * Y[i]) <= 0:  # условие неправильности классификации
                w = w + eta * X[i] * Y[i]      # обновление правила для весов

    return w


w = perceptron_sgd(X, y)
print(w)


for d, sample in enumerate(X):

    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)

    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)


plt.plot([-2, 6], [6, 0.5])     # Построить график
plt.show()                      # Отобразить график



# Учится (2 график)
def perceptron_sgd_plot(X, Y):
    '''
    train perceptron and plot the total loss in each epoch.

    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w) * Y[i]) <= 0:
                total_error += (np.dot(X[i], w) * Y[i])
                w = w + eta * X[i] * Y[i]
        errors.append(total_error * -1)  # подсчет ошибок в каждой эпохе

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.show()

    return w


print(perceptron_sgd_plot(X, y))


# Evaluation
for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# добавить тестовые значения
plt.scatter(2, 2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4, 3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by perceptron_sgd()
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]

x2x3 = np.array([x2, x3])
X, Y, U, V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X, Y, U, V, scale=1, color='blue')

plt.show()