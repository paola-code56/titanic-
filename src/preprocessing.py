"""
Titanic - Análisis exploratorio y preprocesamiento de datos
============================================================

Este script:
1. Carga el dataset crudo (data/raw/train.csv)
2. Realiza un análisis exploratorio (EDA) básico
3. Limpia y transforma las variables
4. Genera un dataset procesado (data/processed/train_clean.csv)
5. Guarda gráficas de apoyo en outputs/

Uso:
    python src/preprocessing.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # para poder correr sin entorno gráfico
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el dataset en '{path}'.\n"
            "Descárgalo desde https://www.kaggle.com/c/titanic/data "
            "y colócalo como data/raw/train.csv"
        )
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("EXPLORACIÓN INICIAL")
    print("=" * 60)
    print(f"Filas x columnas: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum().sort_values(ascending=False))
    print("\nTasa de supervivencia general: "
          f"{df['Survived'].mean() * 100:.2f}%")


def plot_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # Supervivencia por sexo
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Sex", y="Survived")
    plt.title("Tasa de supervivencia por sexo")
    plt.savefig(os.path.join(OUTPUTS_DIR, "survival_by_sex.png"), dpi=150)
    plt.close()

    # Supervivencia por clase
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Pclass", y="Survived")
    plt.title("Tasa de supervivencia por clase")
    plt.savefig(os.path.join(OUTPUTS_DIR, "survival_by_class.png"), dpi=150)
    plt.close()

    # Distribución de edad
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Age"].dropna(), bins=30, kde=True)
    plt.title("Distribución de edad de los pasajeros")
    plt.savefig(os.path.join(OUTPUTS_DIR, "age_distribution.png"), dpi=150)
    plt.close()

    # Mapa de correlación (solo variables numéricas)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de correlación")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nGráficas guardadas en: {OUTPUTS_DIR}")


def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title if title in common else "Otro"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Imputación de nulos ---
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin: alto % de nulos -> indicador binario en lugar de la variable original
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df = df.drop(columns=["Cabin"])

    # --- Feature engineering ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].apply(extract_title)

    # Columnas que no aportan al análisis/modelo
    df = df.drop(columns=["Name", "Ticket", "PassengerId"])

    # --- Codificación de categóricas ---
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], drop_first=True)

    # --- Escalado ---
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

    return df


def main() -> None:
    df = load_data(RAW_PATH)
    explore(df)
    plot_eda(df)

    df_clean = preprocess(df)

    output_path = os.path.join(PROCESSED_DIR, "train_clean.csv")
    df_clean.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Dataset procesado guardado en: {output_path}")
    print(f"Forma final: {df_clean.shape}")
    print("\nPrimeras filas del dataset procesado:")
    print(df_clean.head())


if __name__ == "__main__":
    main()
