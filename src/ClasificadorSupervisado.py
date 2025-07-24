import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class ClasificadorSupervisado:
    def __init__(self):
        self.X = None
        self.y = None
        self.folds = {}
        self.modelos_entrenados = defaultdict(dict)
        self.metricas_por_modelo = defaultdict(dict)
        self.metricas_promedio = {}
        self.modelos_finales = {}
        self.algoritmos_usados = []
        self.scalers = defaultdict(dict)
        self.escalar_datos = False
        self.columnas_a_escalar = []

    def cargar_datos(self, X, y):
        self.X = X
        self.y = y

    def crear_folds(self, metodo='holdout', test_size=0.2, k=5, random_state=42):
        self.folds = {}
        if metodo == 'holdout':
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            self.folds['fold_1'] = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
        elif metodo == 'stratified':
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for train_idx, val_idx in splitter.split(self.X, self.y):
                self.folds['fold_1'] = {
                    'X_train': self.X[train_idx], 'y_train': self.y[train_idx],
                    'X_val': self.X[val_idx], 'y_val': self.y[val_idx]
                }
        elif metodo == 'kfold':
            splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
            for i, (train_idx, val_idx) in enumerate(splitter.split(self.X, self.y), 1):
                self.folds[f'fold_{i}'] = {
                    'X_train': self.X[train_idx], 'y_train': self.y[train_idx],
                    'X_val': self.X[val_idx], 'y_val': self.y[val_idx]
                }
        elif metodo == 'stratified_kfold':
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            for i, (train_idx, val_idx) in enumerate(splitter.split(self.X, self.y), 1):
                self.folds[f'fold_{i}'] = {
                    'X_train': self.X[train_idx], 'y_train': self.y[train_idx],
                    'X_val': self.X[val_idx], 'y_val': self.y[val_idx]
                }
        return self.folds

    def _escalar_fold(self, X_train, X_val, columnas):
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_train_scaled[columnas] = scaler.fit_transform(X_train[columnas])
        X_val_scaled[columnas] = scaler.transform(X_val[columnas])
        return X_train_scaled, X_val_scaled, scaler

    def _evaluar_modelo(self, y_true, y_pred, y_proba):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        best_idx = np.argmin(distances)
        best_threshold = thresholds[best_idx]
        best_distance = distances[best_idx]
        return {
            'confusion_matrix': cm,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1_score': f1,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': auc_score,
            'best_threshold': best_threshold,
            'best_distance': best_distance
        }

    def calcular_metricas_promedio(self, metricas_por_modelo):
        metricas_promedio = {}
        for nombre_alg, resultados in metricas_por_modelo.items():
            metricas_promedio[nombre_alg] = {}
            for key in resultados[next(iter(resultados))].keys():
                if key in ['confusion_matrix', 'fpr', 'tpr']:
                    continue
                valores = []
                for fold_result in resultados.values():
                    valor = fold_result[key]
                    if not isinstance(valor, np.ndarray):
                        valores.append(valor)
                metricas_promedio[nombre_alg][key] = np.mean(valores)
        return metricas_promedio

    def entrenar_modelos(self, hiperparametros=None, algoritmos_usar=None, escalar=False, columnas_a_escalar=[]):
        if hiperparametros is None:
            hiperparametros = {}

        self.escalar_datos = escalar
        self.columnas_a_escalar = columnas_a_escalar

        todos_los_algoritmos = {
            'knn': KNeighborsClassifier(**hiperparametros.get('knn', {})),
            'log_reg': LogisticRegression(**hiperparametros.get('log_reg', {})),
            'tree': DecisionTreeClassifier(**hiperparametros.get('tree', {})),
            'rf': RandomForestClassifier(**hiperparametros.get('rf', {})),
            'nb': GaussianNB(**hiperparametros.get('nb', {})),
        }

        if algoritmos_usar is None:
            algoritmos_usar = list(todos_los_algoritmos.keys())

        self.algoritmos_usados = algoritmos_usar

        for nombre_alg in algoritmos_usar:
            modelo = todos_los_algoritmos[nombre_alg]
            self.metricas_por_modelo[nombre_alg] = {}
            self.modelos_entrenados[nombre_alg] = {}
            self.scalers[nombre_alg] = {}

            for fold, datos in self.folds.items():
                X_train, X_val = datos['X_train'].copy(), datos['X_val'].copy()
                if self.escalar_datos:
                    X_train, X_val, scaler = self._escalar_fold(X_train, X_val, self.columnas_a_escalar)
                    self.scalers[nombre_alg][fold] = scaler

                modelo.fit(X_train, datos['y_train'])
                y_pred = modelo.predict(X_val)
                y_proba = modelo.predict_proba(X_val)[:, 1] if hasattr(modelo, 'predict_proba') else y_pred
                metricas = self._evaluar_modelo(datos['y_val'], y_pred, y_proba)
                self.modelos_entrenados[nombre_alg][fold] = modelo
                self.metricas_por_modelo[nombre_alg][fold] = metricas

        self.metricas_promedio = self.calcular_metricas_promedio(self.metricas_por_modelo)
        return self.modelos_entrenados, self.metricas_por_modelo, self.metricas_promedio

    def predecir_por_fold(self, X):
        predicciones = {}
        for nombre_alg in self.algoritmos_usados:
            predicciones[nombre_alg] = {}
            for fold, modelo in self.modelos_entrenados[nombre_alg].items():
                X_copy = X.copy()
                if self.escalar_datos and nombre_alg in self.scalers and fold in self.scalers[nombre_alg]:
                    scaler = self.scalers[nombre_alg][fold]
                    X_copy[self.columnas_a_escalar] = scaler.transform(X[self.columnas_a_escalar])
                pred = modelo.predict(X_copy)
                proba = modelo.predict_proba(X_copy) if hasattr(modelo, 'predict_proba') else None
                predicciones[nombre_alg][fold] = {
                    'clases': pred.tolist(),
                    'probabilidades': proba.tolist() if proba is not None else None
                }
        return predicciones

    def graficar_metricas_seaborn(self):
        metricas_a_graficar = ['recall', 'specificity', 'accuracy', 'f1_score', 'auc']
        for alg in self.algoritmos_usados:
            resultados = self.metricas_por_modelo[alg]
            datos = []
            for fold, metricas in resultados.items():
                for metrica in metricas_a_graficar:
                    datos.append({
                        'Fold': fold,
                        'Métrica': metrica,
                        'Valor': metricas[metrica],
                        'Algoritmo': alg
                    })
            df_metricas = pd.DataFrame(datos)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_metricas, x='Fold', y='Valor', hue='Métrica')
            plt.title(f'Métricas por Fold - {alg}')
            plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()