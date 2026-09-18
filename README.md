# Titanic - Análisis y Preprocesamiento de Datos

Proyecto individual de la actividad de manejo masivo de datos. Consiste en el análisis
exploratorio y preprocesamiento del dataset **Titanic - Machine Learning from Disaster**
de Kaggle.

## Dataset

- **Fuente:** https://www.kaggle.com/c/titanic/data
- **Archivo usado:** `train.csv`
- **Descripción:** Registros de pasajeros del Titanic (edad, sexo, clase, tarifa, puerto
  de embarque, etc.) junto con la variable objetivo `Survived` (1 = sobrevivió, 0 = no
  sobrevivió).

> El dataset **no se incluye** en este repositorio (ver `.gitignore`). Debes descargarlo
> manualmente desde Kaggle y colocarlo en `data/raw/train.csv`.

## Objetivo

1. Explorar la estructura y calidad de los datos (valores nulos, tipos, distribución).
2. Realizar limpieza y preprocesamiento: imputación de nulos, codificación de variables
   categóricas, creación de nuevas variables (feature engineering) y escalado.
3. Dejar un dataset procesado, listo para un futuro modelo de clasificación.
4. Generar visualizaciones que resuman los hallazgos principales.

## Estructura del proyecto

```
titanic-project/
├── data/
│   ├── raw/            # dataset original (train.csv) - no versionado
│   └── processed/      # dataset limpio, generado por el script
├── outputs/             # gráficas generadas (.png)
├── src/
│   └── preprocessing.py # script principal de EDA + limpieza
├── requirements.txt
├── .gitignore
└── README.md
```

## Cómo reproducirlo

```bash
# 1. Clonar el repositorio
git clone <URL-DE-ESTE-REPO>
cd titanic-project

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset de Kaggle y colocarlo en:
#    data/raw/train.csv

# 5. Ejecutar el script
python src/preprocessing.py
```

Al finalizar, el script genera:
- `data/processed/train_clean.csv` → dataset preprocesado.
- `outputs/*.png` → gráficas de la exploración (distribución de edad, supervivencia
  por clase/sexo, mapa de correlación, etc.)
- Un resumen impreso en consola con estadísticas clave.

## Preprocesamiento aplicado

- **Valores nulos:**
  - `Age`: imputado con la mediana por grupo de `Pclass` y `Sex`.
  - `Embarked`: imputado con la moda.
  - `Cabin`: se descarta por alto porcentaje de valores faltantes; se conserva solo
    un indicador binario `HasCabin`.
- **Variables categóricas:** codificación con `pd.get_dummies` para `Sex` y `Embarked`.
- **Feature engineering:**
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone` (1 si viaja solo, 0 si no)
  - `Title` extraído del nombre (Mr, Mrs, Miss, Master, Otro)
- **Escalado:** `Fare` y `Age` escalados con `StandardScaler`.

## Hallazgos principales (a completar tras ejecutar)

- Tasa de supervivencia general: `__%`
- Supervivencia por sexo: `__`
- Supervivencia por clase: `__`
- Variable con más valores nulos: `__`

## Autor

Nombre: _(completar)_
Actividad: Manejo masivo de datos - Proyecto individual (pareja: nombre del compañero)
