# Предсказание выкупа заказов (Buyout Prediction)

Двухмодельная система предсказания выкупа заказов на основе данных AmoCRM.

## Как это работает

Система автоматически определяет тип клиента и применяет соответствующую модель:

- **Повторные клиенты** (`contact_Число сделок` >= 1): buyout 97%+, LogReg на одном признаке
- **Новые клиенты** (первый заказ): LogReg на ~25 признаках, ROC-AUC 0.87

Дополнительно обучены и сравнены продвинутые модели (Random Forest, CatBoost, XGBoost) — лучший результат у CatBoost (ROC-AUC 0.88).

## Использование

```python
from model import BuyoutPredictor

predictor = BuyoutPredictor("model_weights.pkl")

predictions = predictor.predict(df)          # 0/1
probabilities = predictor.predict_proba(df)  # float [0, 1]
```

На вход подается `pd.DataFrame` с исходными колонками из датасета AmoCRM (одна или несколько строк). Предобработка, feature engineering и маршрутизация между моделями выполняются автоматически.

## Структура проекта

| Файл | Описание |
|------|----------|
| `model.py` | Класс `BuyoutPredictor` для инференса |
| `01_data_analysis.ipynb` | EDA, feature engineering, подготовка данных |
| `02_model_training.ipynb` | Обучение LogReg, валидация, сохранение весов |
| `03_advanced_models.ipynb` | Random Forest, CatBoost, XGBoost + SHAP-анализ |
| `model_weights.pkl` | Обученные веса (генерируется `02_model_training.ipynb`) |
| `russia-cities.json` | Справочник городов для гео-матчинга |

## Запуск

1. Установить зависимости: `pip install -r requirements.txt`
2. Поместить CSV-датасет (`dataset_2025-03-01_2026-03-29_external.csv`) в текущую или родительскую директорию
3. Запустить все ячейки `01_data_analysis.ipynb` — анализ данных и подготовка признаков
4. Запустить все ячейки `02_model_training.ipynb` — обучение LogReg и сохранение `model_weights.pkl`
5. (Опционально) Запустить `03_advanced_models.ipynb` — сравнение с RF, CatBoost, XGBoost и SHAP-анализ
6. Использовать `BuyoutPredictor` из `model.py`

## Зависимости

```
pandas, numpy, scikit-learn, catboost, xgboost, shap, matplotlib, joblib
```
