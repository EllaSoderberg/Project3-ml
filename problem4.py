import numpy as np


def poly_features(data, degree, p):
    values = []

    if degree > 1:
        values = poly_features(data, degree - 1, p)
        k = len(values)

        if degree > 2:
            for i in range(1, len(data)):
                p[len(data) - 1 - i] = p[len(data) - 1 - i] + p[len(data) - i]

        for i in range(0, len(data)):
            add = k - p[i]
            for j in range(add, k):
                values.append(data[i] * values[j])

    else:
        values.append(1)

        for i in range(0, len(data)):
            values.append(data[i])
    return values

def poly_transormation(data, degree):
    newdata = []

    for j in range(0, len(data)):
        p = []
        for i in range(len(data[j]), 0, -1):
            p.append(i)

        values = poly_features(data[j], degree, p)
        newdata.append(values)

    return np.array(newdata)






