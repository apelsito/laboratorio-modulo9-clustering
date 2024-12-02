# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy.stats import chi2_contingency
# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt


# Para tratar el problema de desbalance
# -----------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def exploracion_dataframe(self, columna_control, estadisticos = False):
        """
        Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
        valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
        para columnas categóricas y numéricas, agrupadas por la columna de control.

        Params:
        - dataframe (DataFrame): El DataFrame que se va a explorar.
        - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

        Returns: 
        No devuelve nada directamente, pero imprime en la consola la información exploratoria.
        """
        print(f"El número de datos es {self.dataframe.shape[0]} y el de columnas es {self.dataframe.shape[1]}")
        print("\n ..................... \n")

        print(f"Los duplicados que tenemos en el conjunto de datos son: {self.dataframe.duplicated().sum()}")
        print("\n ..................... \n")
        
        
        # generamos un DataFrame para los valores nulos
        print("Los nulos que tenemos en el conjunto de datos son:")
        df_nulos = pd.DataFrame(self.dataframe.isnull().sum() / self.dataframe.shape[0] * 100, columns = ["%_nulos"])
        display(df_nulos[df_nulos["%_nulos"] > 0])
        
        # Tipos de columnas
        print("\n ..................... \n")
        print(f"Los tipos de las columnas son:")
        display(pd.DataFrame(self.dataframe.dtypes, columns = ["tipo_dato"]))
        
        # Enseñar solo las columnas categoricas (o tipo objeto)
        print("\n ..................... \n")
        print("Los valores que tenemos para las columnas categóricas son: ")
        dataframe_categoricas = self.dataframe.select_dtypes(include = "O")
        
        for col in dataframe_categoricas.columns:
            print(f"La columna {col.upper()} tiene los siguientes valores únicos:")
            print(f"Mostrando {pd.DataFrame(self.dataframe[col].value_counts()).head().shape[0]} categorías con más valores del total de {len(pd.DataFrame(self.dataframe[col].value_counts()))} categorías ({pd.DataFrame(self.dataframe[col].value_counts()).head().shape[0]}/{len(pd.DataFrame(self.dataframe[col].value_counts()))})")
            display(pd.DataFrame(self.dataframe[col].value_counts()).head())    
        
        # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
        if estadisticos == True:
            for categoria in self.dataframe[columna_control].unique():
                dataframe_filtrado = self.dataframe[self.dataframe[columna_control] == categoria]
                #Describe de objetos
                print("\n ..................... \n")

                print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
                display(dataframe_filtrado.describe(include = "O").T)

                #Hacer un describe
                print("\n ..................... \n")
                print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
                display(dataframe_filtrado.describe().T)
        else: 
            pass
        print("\n----------\n")
        print("Las principales estadísticas de las variables númericas son:")
        display(self.dataframe.describe().T)

        print("\n----------\n")
        print("Las principales estadísticas de las variables categóricas son:")
        display(self.dataframe.describe(include = ["O","category"]).T)

        print("\n----------\n")
        print("Las características principales del dataframe son:")
        display(self.dataframe.info())

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include=["O", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(20, 10)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
            axes[indice].set_title(f"Distribución de {columna}")
        plt.suptitle("Distribución de variables numéricas")

        if len(lista_num) % 2 !=0:
            fig.delaxes(axes[-1])
        plt.tight_layout()


    def plot_categoricas(self, color="grey", tamano_grafica=(20, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_cat) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(lista_cat):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.suptitle("Distribución de variables categóricas")
        
        if len(lista_cat) % 2 !=0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()



    def plot_relacion(self, vr, tamano_grafica=(20, 10)):
        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(self.dataframe.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = True)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma",
                              legend=True
                              )
                              

            axes[indice].set_title(f"Relación {columna} vs {vr}",size=25)   
        if (len(self.dataframe.columns) % 2) != 0:
            fig.delaxes(axes[-1])
        else:
            pass
        plt.tight_layout()
    
        
    def deteccion_outliers(self, color = "grey",tamano_grafica=(20, 10)):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize= tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            axes[indice].set_title(f"Outliers {columna}")  

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada 

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="magma",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
        
    

class Desbalanceo:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente

    def visualizar_clase(self, color="orange", edgecolor="black"):
        plt.figure(figsize=(8, 5))  # para cambiar el tamaño de la figura
        fig = sns.countplot(data=self.dataframe, 
                            x=self.variable_dependiente,  
                            color=color,  
                            edgecolor=edgecolor)
        fig.set(xticklabels=["No", "Yes"])
        plt.show()

    def balancear_clases_pandas(self, metodo):
        # Contar las muestras por clase
        contar_clases = self.dataframe[self.variable_dependiente].value_counts()
        clase_mayoritaria = contar_clases.idxmax()
        clase_minoritaria = contar_clases.idxmin()

        # Separar las clases
        df_mayoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_mayoritaria]
        df_minoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_minoritaria]

        if metodo == "downsampling":
            # Submuestrear la clase mayoritaria
            df_majority_downsampled = df_mayoritaria.sample(contar_clases[clase_minoritaria], random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_majority_downsampled, df_minoritaria])

        elif metodo == "upsampling":
            # Sobremuestrear la clase minoritaria
            df_minority_upsampled = df_minoritaria.sample(contar_clases[clase_mayoritaria], replace=True, random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_mayoritaria, df_minority_upsampled])

        else:
            raise ValueError("Método no reconocido. Use 'downsampling' o 'upsampling'.")

        return df_balanced

    def balancear_clases_imblearn(self, metodo):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        if metodo == "RandomOverSampler":
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

        elif metodo == "RandomUnderSampler":
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

        else:
            raise ValueError("Método no reconocido. Use 'RandomOverSampler' o 'RandomUnderSampler'.")

        df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    
    def balancear_clases_smote(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

    def balancear_clases_smote_tomek(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    
    def balancear_clases_smotenc(self,categorical_columns,neighbors_k = 5):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smotenc = SMOTENC(categorical_features=categorical_columns, random_state=42, k_neighbors=neighbors_k) 
        X_resampled, y_resampled = smotenc.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

    def balancear_clases_tomek(self):
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]
        
        # Configurar Tomek Links
        tomek = TomekLinks(sampling_strategy="auto")
        
        # Aplicar Tomek Links
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        
        # Crear un DataFrame resultante
        df_resampled = pd.concat(
            [pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)],
            axis=1
        )
        
        return df_resampled
    