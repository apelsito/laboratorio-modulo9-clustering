import pickle
import pandas as pd
import numpy as np

# Cargar los modelos y transformadores entrenados
def load_models():
    """
    Carga modelos y transformadores previamente entrenados desde archivos pickle.

    Retorna:
    --------
    tuple
        Una tupla que contiene los siguientes objetos:
            - target_encoder: Codificador para transformar la variable objetivo.
            - one_hot_encoder: Codificador One-Hot para variables categóricas.
            - robust_scaler: Escalador robusto para normalizar los datos.
            - modelo: Modelo de aprendizaje automático (XGBoost en este caso).

    Notas:
    ------
    - Los archivos deben estar ubicados en la carpeta `../datos/modelos-encoders/`.
    - Se espera que los archivos tengan los nombres:
        - `target_encoder.pkl`
        - `one_hot_encoder.pkl`
        - `robust_scaler.pkl`
        - `modelo_xgb.pkl`

    Ejemplo:
    --------
    target_encoder, one_hot_encoder, robust_scaler, modelo = load_models()
    """

    with open('../datos/encoders_y_modelos/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('../datos/encoders_y_modelos/onehot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)
    with open('../datos/encoders_y_modelos/robust_scaler.pkl', 'rb') as f:
        robust_scaler = pickle.load(f)    
    with open('../datos/encoders_y_modelos/best_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return target_encoder, one_hot_encoder, robust_scaler, modelo

def load_options():

    # Cargar cada archivo con su correspondiente variable
    
    with open("../datos/lista_opciones/0_lista_opciones_BusinessTravel.plk", 'rb') as archivo:
        lista_business_travel = pickle.load(archivo)

    with open('../datos/lista_opciones/1_lista_opciones_Department.plk', 'rb') as archivo:
        lista_department = pickle.load(archivo)

    with open('../datos/lista_opciones/2_lista_opciones_DistanceFromHome.plk', 'rb') as archivo:
        lista_distance_from_home = pickle.load(archivo)

    with open('../datos/lista_opciones/3_lista_opciones_Education.plk', 'rb') as archivo:
        lista_education = pickle.load(archivo)

    with open('../datos/lista_opciones/4_lista_opciones_EducationField.plk', 'rb') as archivo:
        lista_education_field = pickle.load(archivo)

    with open('../datos/lista_opciones/5_lista_opciones_Gender.plk', 'rb') as archivo:
        lista_gender = pickle.load(archivo)

    with open('../datos/lista_opciones/6_lista_opciones_JobLevel.plk', 'rb') as archivo:
        lista_job_level = pickle.load(archivo)

    with open('../datos/lista_opciones/7_lista_opciones_JobRole.plk', 'rb') as archivo:
        lista_job_role = pickle.load(archivo)

    with open('../datos/lista_opciones/8_lista_opciones_MaritalStatus.plk', 'rb') as archivo:
        lista_marital_status = pickle.load(archivo)

    with open('../datos/lista_opciones/9_lista_opciones_NumCompaniesWorked.plk', 'rb') as archivo:
        lista_num_companies_worked = pickle.load(archivo)

    with open('../datos/lista_opciones/10_lista_opciones_PercentSalaryHike.plk', 'rb') as archivo:
        lista_percent_salary_hike = pickle.load(archivo)

    with open('../datos/lista_opciones/11_lista_opciones_StockOptionLevel.plk', 'rb') as archivo:
        lista_stock_option_level = pickle.load(archivo)

    with open('../datos/lista_opciones/12_lista_opciones_TrainingTimesLastYear.plk', 'rb') as archivo:
        lista_training_times_last_year = pickle.load(archivo)

    with open('../datos/lista_opciones/13_lista_opciones_EnvironmentSatisfaction.plk', 'rb') as archivo:
        lista_environment_satisfaction = pickle.load(archivo)

    with open('../datos/lista_opciones/14_lista_opciones_JobSatisfaction.plk', 'rb') as archivo:
        lista_job_satisfaction = pickle.load(archivo)

    with open('../datos/lista_opciones/15_lista_opciones_WorkLifeBalance.plk', 'rb') as archivo:
        lista_work_life_balance = pickle.load(archivo)

    with open('../datos/lista_opciones/16_lista_opciones_JobInvolvement.plk', 'rb') as archivo:
        lista_job_involvement = pickle.load(archivo)

    return lista_business_travel,lista_department,lista_distance_from_home,lista_education,lista_education_field,lista_gender,lista_job_level,lista_job_role,lista_marital_status,lista_num_companies_worked,lista_percent_salary_hike,lista_stock_option_level,lista_training_times_last_year,lista_environment_satisfaction,lista_job_satisfaction,lista_work_life_balance,lista_job_involvement


def realizar_prediccion(edad, viajes, departamento, distancia_a_casa,
                        educacion, area_educacion, genero, nivel_laboral,
                        rol_laboral, estado_marital, sueldo_mensual, companies_previas,
                        aumento_salario_porcentual, opciones_bolsa, anios_trabajados,
                        cursos_acometidos_anio_pasado, anios_en_empresa, anios_desde_ascenso,
                        anios_con_el_manager, satisfaccion_ambiente_trabajo, satisfaccion_laboral,
                        balance_vida_trabajo, participacion_laboral,
                        encoder_ordinales, encoder_nominales, scaler, modelo):

    cols_nominales = ["DistanceFromHome","Education", "Gender", "JobLevel", "JobRole", "PercentSalaryHike", "StockOptionLevel","TrainingTimesLastYear","JobInvolvement"]

    cols_escalar = ["Age","BusinessTravel", "Department", "EducationField",
                    "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked","TotalWorkingYears",
                    "YearsAtCompany","YearsSinceLastPromotion","YearsWithCurrManager",
                    "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]
    
    df = pd.read_pickle("../datos/02_datos_gestionados.plk")


    # Datos de una nueva casa para predicción
    employee = pd.DataFrame({
        'Age': [edad],
        'BusinessTravel': [viajes],
        'Department': [departamento],
        'DistanceFromHome': [distancia_a_casa],
        'Education': [educacion],
        'EducationField': [area_educacion],
        'Gender': [genero],
        'JobLevel': [nivel_laboral],
        'JobRole': [rol_laboral],
        'MaritalStatus': [estado_marital],
        'MonthlyIncome': [sueldo_mensual],
        'NumCompaniesWorked': [companies_previas],
        'PercentSalaryHike': [aumento_salario_porcentual],
        'StockOptionLevel': [opciones_bolsa],
        'TotalWorkingYears': [anios_trabajados],
        'TrainingTimesLastYear': [cursos_acometidos_anio_pasado],
        'YearsAtCompany': [anios_en_empresa],
        'YearsSinceLastPromotion': [anios_desde_ascenso],
        'YearsWithCurrManager': [anios_con_el_manager],
        'EnvironmentSatisfaction': [satisfaccion_ambiente_trabajo],
        'JobSatisfaction': [satisfaccion_laboral],
        'WorkLifeBalance': [balance_vida_trabajo],
        'JobInvolvement' : [participacion_laboral]
    })

    df_pred = pd.DataFrame(employee)
    df_pred

    # Hacemos el OneHot Encoder
    onehot = encoder_nominales.transform(df_pred[cols_nominales])
    # Obtenemos los nombres de las columnas del codificador
    column_names = encoder_nominales.get_feature_names_out(cols_nominales)
    # Convertimos a un DataFrame
    onehot_df = pd.DataFrame(onehot.toarray(), columns=column_names)

    # Realizamos el target encoder
    df_pred["Attrition"] = np.nan #La creo porque la espera, luego se borra
    df_pred = encoder_ordinales.transform(df_pred)

    # Quitamos las columnas que ya han sido onehoteadas 
    df_pred.drop(columns= cols_nominales,inplace=True)
    df_pred = pd.concat([df_pred, onehot_df], axis=1)

    # Escalamos los valores
    df_pred[cols_escalar] = scaler.transform(df_pred[cols_escalar])

    # Realizamos la predicción
    df_pred.drop(columns="Attrition",inplace=True)
    prediccion = modelo.predict(df_pred)
    
    return prediccion