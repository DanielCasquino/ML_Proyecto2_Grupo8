import numpy as np

class Nodo:
    # Define what your data members will be
    def __init__(self, index=None, umbral=None):
        # Initialize data members
        self.label = None  # label solo para la hoja

        self.index = None  # feature de division
        self.umbral = None  # valor de division

        self.left = None
        self.right = None

    def IsTerminal(self, Y):
        # return true if this node has the same labels in Y
        return (len(Y) < 5) or (len(np.unique(Y)) == 1 and np.unique(Y)[
            0] == self.label)  # los datos de Y son todos de la misma featura

    def BestSplit(self, x, y):
        # write your code here

        if (self.IsTerminal(y)):
            if (len(np.unique(y)) == 1):
                self.label = y[0]
            else:
                labels_y, ocurrencias_y = np.unique(y, return_counts=True)
                self.label = labels_y[np.argmax(ocurrencias_y)]
            return [], [], [], []
        else:
            index_feature = 0
            best_umbral = 0

            x_features = np.array(x).T

            max_info_gain = -1
            # recorrer todas las features disponibles
            for i in range(x_features.shape[0]):
                feature = x_features[i]

                umbrales = np.unique(feature)
                for umbral in umbrales:
                    info_gain = self.InfoGain(feature, umbral, y)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        index_feature = i
                        best_umbral = umbral
            self.index = index_feature
            self.umbral = best_umbral

            left_data_x = []
            right_data_x = []
            left_data_y = []
            right_data_y = []
            for i in range(len(x)):
                sample = x[i]
                if sample[index_feature] <= best_umbral:
                    left_data_x.append(sample)
                    left_data_y.append(y[i])
                else:
                    right_data_x.append(sample)
                    right_data_y.append(y[i])

            return left_data_x, right_data_x, left_data_y, right_data_y

    def Entropy(self, Y):
        # write your code here
        labels, ocurrencias = np.unique(Y, return_counts=True)
        probabilities = ocurrencias / len(Y)
        return -1 * np.sum(probabilities * np.log2(probabilities))

    def Gini(self, Y):
        # write your code here
        labels, ocurrencias = np.unique(Y, return_counts=True)
        probabilities = ocurrencias / len(Y)
        return 1 - np.sum(probabilities ** 2)

    def InfoGain(self, feature, umbral, label):
        samples = len(label)
        # separar los datos en dos grupos

        left = np.array([label[i] for i in range(samples) if feature[i] <= umbral])
        right = np.array([label[i] for i in range(samples) if feature[i] > umbral])

        # calcular la entropia de cada lado
        left_entropy = self.Entropy(left)
        right_entropy = self.Entropy(right)

        # calcular la entropia total
        total_entropy = (len(left) / samples) * left_entropy + (len(right) / samples) * right_entropy
        # calcular la entropia de label
        label_entropy = self.Entropy(label)

        return label_entropy - total_entropy


class DT:
    # Defina cuales será sus mimbros datos

    def __init__(self, X, Y, index):
        # Inicializar los mimbros datos
        self.m_Root = self.create_DT(X, Y)

    def create_DT(self, X, Y):
        print(Y)
        # write your code here
        samples = len(Y)
        features = len(X)
        if (len(np.unique(Y)) == 1):
            new_Node = Nodo()
            new_Node.label = Y[0]
            return new_Node
        elif samples < 5:
            new_Node = Nodo()
            labels_y, ocurrencias_y = np.unique(Y, return_counts=True)
            new_Node.label = labels_y[np.argmax(ocurrencias_y)]
            return new_Node
        else:
            new_Node, left_data_x, right_data_x, left_data_y, right_data_y = self.Find_Best_Split(X, Y)
            # print(new_Node.index, len(left_data_x), len(right_data_x), len(left_data_y), len(right_data_y))
            if (new_Node.IsTerminal(Y)):
                return new_Node
            else:
                new_Node.left = self.create_DT(left_data_x, left_data_y)
                new_Node.right = self.create_DT(right_data_x, right_data_y)
                return new_Node

    # return the best feature node with their 2 children
    def Find_Best_Split(self, X, Y):
        new_Node = Nodo()
        left_data_x, right_data_x, left_data_y, right_data_y = new_Node.BestSplit(X, Y)

        return new_Node, left_data_x, right_data_x, left_data_y, right_data_y

    def predict(self, X):
        # write your code here
        y_pred = []
        for sample in X:
            y_pred.append(self.predictAux(sample, self.m_Root))
        return y_pred

    def predictAux(self, X, node):
        feature_x = X[node.index]
        if node.left is None and node.right is None:
            return node.label
        if feature_x <= node.umbral:
            if node.left is None:
                return node.label
            else:
                return self.predictAux(X, node.left)
        else:
            if node.right is None:
                return node.label
            else:
                return self.predictAux(X, node.right)


