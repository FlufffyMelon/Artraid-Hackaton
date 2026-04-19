# Предсказание выкупа заказов (Buyout Prediction)

Двухмодельная система предсказания выкупа заказов на основе данных AmoCRM.

## Как это работает

Система автоматически определяет тип клиента и применяет соответствующую модель:

- **Старые клиенты** (`contact_Число сделок` >= 1): LogReg на одном признаке (число прошлых сделок)
- **Новые клиенты** (`contact_Число сделок` < 1): LogReg на ~20 признаках

Дополнительно обучена CatBoost-модель на тех же признаках для сравнения и проведён SHAP-анализ важности признаков.

## Использование

```python
from model import BuyoutPredictor

predictor = BuyoutPredictor("features.yaml", "data")

predictions = predictor.predict(df)          # 0/1
probabilities = predictor.predict_proba(df)  # float [0, 1]
```

На вход подаётся `pd.DataFrame` – запись из AmoCRM (один или несколько клиентов). Подготовка данных происходит согласно `features.yaml`.

## Структура проекта

| Файл / каталог | Описание |
|----------------|----------|
| `model.py` | Класс `BuyoutPredictor` для инференса |
| `features.yaml` | Конфиг используемых признаков |
| `01_data_analysis.ipynb` | Чтение, очистка, подготовка и визуализация данных |
| `02_model_training.ipynb` | Обучение LogReg на старых и новых пользователях |
| `03_advanced_models.ipynb` | Обучение Catboost модели на тех же данных |
| `utils/` | Вспомогательные методы: `features`, `data`, `encoding`, `metrics`, `time_split`, `plotting` |
| `data/` | Генерируемые данные: `clean.csv`, `contexts.joblib`, `logreg_*.joblib`, `catboost_new.cbm`, `model_meta.yaml` |
| `russia-cities.json` | Справочник городов России |

## Запуск

1. Установить зависимости: `pip install -r requirements.txt`
2. Поместить CSV-датасет (`dataset_2025-03-01_2026-03-29_external.csv`) в корень репозитория
3. Запустить все ячейки `01_data_analysis.ipynb` — анализ данных, подготовка признаков, сохранение `data/clean.csv` и `data/contexts.joblib`
4. Запустить все ячейки `02_model_training.ipynb` — обучение LogReg, сохранение `data/logreg_*.joblib` и `data/model_meta.yaml`
5. (Опционально) Запустить `03_advanced_models.ipynb` — CatBoost, SHAP-анализ, сохранение `data/catboost_new.cbm`
6. Использовать `BuyoutPredictor` из `model.py`

## Зависимости

```
pandas, numpy, pyyaml, scikit-learn, catboost, shap, matplotlib, joblib, tabulate, jupyter, nbformat
```
