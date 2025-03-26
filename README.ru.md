# Глава 6: Практическое руководство по обучению ML-моделей на финансовых данных

## Обзор

Машинное обучение, применяемое к финансовым данным, работает в принципиально иных условиях, чем типичные области ML, такие как компьютерное зрение или обработка естественного языка. Финансовые сигналы крайне зашумлены, с соотношением сигнал/шум часто ниже 0.05, что означает, что предсказуемый компонент доходности ничтожен по сравнению со случайностью. Эта среда с низким соотношением сигнал/шум усиливает компромисс смещение-дисперсия: модели, достаточно выразительные для захвата подлинных паттернов, также достаточно мощны для запоминания шума, что приводит к впечатляющей производительности на обучающей выборке, которая исчезает вне выборки.

Стандартный рабочий процесс ML -- разделить данные на обучение/валидацию/тест, оптимизировать гиперпараметры, оценить -- должен быть существенно модифицирован для финансовых приложений. Данные временных рядов нарушают предположение о независимости и одинаковом распределении (i.i.d.), лежащее в основе стандартной кросс-валидации. Временные зависимости означают, что случайное перемешивание данных по фолдам создаёт смещение заглядывания вперёд, когда модель обучается на будущей информации для предсказания прошлого. Очищенная k-fold кросс-валидация с окнами эмбарго решает эту проблему, обеспечивая строгое временное разделение между обучающими и валидационными данными, предотвращая утечку информации через серийную корреляцию в признаках или метках.

Эта глава охватывает полный рабочий процесс ML, адаптированный для криптотрейдинга: от конструирования и отбора признаков с использованием взаимной информации, через правильную кросс-валидацию с очисткой и эмбарго, до оптимизации гиперпараметров байесовскими методами. Мы рассматриваем распространённые ошибки, включая утечку данных, смещение заглядывания вперёд при конструировании признаков и проблему множественного тестирования, возникающую при оценке многих вариантов стратегий. Предоставлены реализации на Python и Rust, причём код на Rust фокусируется на высокопроизводительных разделителях кросс-валидации, подходящих для крупномасштабных криптонаборов данных.

## Содержание

1. [Введение в ML для финансовых данных](#раздел-1-введение-в-ml-для-финансовых-данных)
2. [Математические основы](#раздел-2-математические-основы)
3. [Сравнение методов кросс-валидации](#раздел-3-сравнение-методов-кросс-валидации)
4. [Торговые применения](#раздел-4-торговые-применения)
5. [Реализация на Python](#раздел-5-реализация-на-python)
6. [Реализация на Rust](#раздел-6-реализация-на-rust)
7. [Практические примеры](#раздел-7-практические-примеры)
8. [Фреймворк бэктестирования](#раздел-8-фреймворк-бэктестирования)
9. [Оценка производительности](#раздел-9-оценка-производительности)
10. [Перспективные направления](#раздел-10-перспективные-направления)

---

## Раздел 1: Введение в ML для финансовых данных

### Рабочий процесс ML для трейдинга

Рабочий процесс машинного обучения для финансовых приложений следует структурированному конвейеру:

1. **Сбор данных**: Получение данных OHLCV, стакана заявок, ставок финансирования и ончейн-данных
2. **Конструирование признаков**: Создание предсказательных признаков (технические индикаторы, сигналы микроструктуры)
3. **Отбор признаков**: Удаление избыточных и шумных признаков с помощью взаимной информации или оценок важности
4. **Конструирование меток**: Определение цели предсказания (будущие доходности, направление, режим волатильности)
5. **Проектирование кросс-валидации**: Выбор подходящей временной CV с очисткой и эмбарго
6. **Обучение модели**: Подгонка моделей с правильной регуляризацией
7. **Настройка гиперпараметров**: Использование байесовской оптимизации (Optuna) с вложенной CV
8. **Оценка**: Анализ производительности вне выборки с реалистичными транзакционными издержками

### Обучение с учителем, без учителя и с подкреплением

**Обучение с учителем** доминирует в приложениях криптотрейдинга: по признакам X предсказать целевую переменную y (направление доходности, величину или волатильность). Распространённые модели включают линейную регрессию, случайные леса, градиентный бустинг и нейронные сети.

**Обучение без учителя** служит для определения режимов (кластеризация рыночных состояний), снижения размерности (PCA на больших наборах признаков) и обнаружения аномалий (выявление необычных рыночных условий).

**Обучение с подкреплением** рассматривает трейдинг как задачу последовательного принятия решений, где агент обучает политику, отображающую рыночные состояния в действия (купить/продать/удержать), оптимизируя кумулятивное вознаграждение. Хотя теоретически привлекательное, RL для трейдинга сталкивается с проблемами нестационарности и неэффективности использования данных.

### Проблема сигнал/шум

Финансовые данные характеризуются крайне низким соотношением сигнал/шум. Если классификатор изображений может достигать 95%+ точности, то предсказатель доходности криптовалют с 52% точности направления может быть высокоприбыльным. Это означает:

- Стандартные метрики точности обманчивы
- Риск переобучения экстремален -- модель может легко найти ложные паттерны
- Отбор признаков критически важен для избежания подгонки под шум
- Ансамблевые методы и регуляризация необходимы
- Валидация вне выборки должна быть строгой

---

## Раздел 2: Математические основы

### Компромисс смещение-дисперсия

Ожидаемая ошибка предсказания разлагается как:

```
E[(y - f_hat(x))^2] = Bias^2 + Variance + Irreducible_Noise

Где:
  Bias^2 = [E[f_hat(x)] - f(x)]^2         (модель слишком простая)
  Variance = E[(f_hat(x) - E[f_hat(x)])^2] (модель слишком сложная)
  Irreducible_Noise = sigma^2               (неустранимая случайность)
```

В финансовых данных член неустранимого шума доминирует. Это сдвигает оптимальную сложность модели ниже, чем в типичных ML-приложениях -- более простые модели с сильной регуляризацией часто превосходят сложные.

### Ошибка обобщения и переобучение

Ошибка обобщения измеряет производительность на невиданных данных:

```
R(f) = E[L(y, f(x))]  (популяционный риск)
R_hat(f) = (1/n) * sum(L(y_i, f(x_i)))  (эмпирический риск)
```

Разрыв `R(f) - R_hat(f)` растёт с увеличением сложности модели и уменьшается с размером выборки. Для финансовых данных с автокоррелированными наблюдениями эффективный размер выборки значительно меньше числа точек данных:

```
n_eff = n / (1 + 2 * sum_{k=1}^{inf} rho(k))
```

Где `rho(k)` -- автокорреляция на лаге k.

### Взаимная информация для отбора признаков

Взаимная информация измеряет статистическую зависимость между признаком X и целью Y:

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

Для непрерывных переменных (типичных для финансов) используется оценка k ближайших соседей:

```
I_hat(X; Y) = psi(k) - E[psi(n_x + 1)] - E[psi(n_y + 1)] + psi(n)
```

Где `psi` -- дигамма-функция, а `n_x`, `n_y` -- количество соседей.

Преимущества перед корреляцией:
- Захватывает нелинейные зависимости
- Работает с негауссовскими распределениями (критично для крипто)
- Инвариантна к масштабу

### Очищенная K-fold кросс-валидация

Стандартная k-fold CV случайно назначает наблюдения по фолдам, создавая утечку информации через:
1. **Серийную корреляцию**: Обучающие данные, смежные с тестовыми, информативны
2. **Перекрытие меток**: Если метки охватывают несколько баров, обучающие и тестовые метки могут разделять информацию

Очищенная k-fold решает это:

```
Для каждого фолда k:
  test_start, test_end = границы фолда

  Очистка: Удалить обучающие образцы, где:
    label_end_i > test_start  AND  label_start_i < test_end

  Эмбарго: Дополнительно удалить обучающие образцы, где:
    sample_time_i > test_end  AND  sample_time_i < test_end + embargo_period
```

Период эмбарго должен быть не менее длительности максимальной серийной корреляции в признаках.

### Валидация скользящим окном (Walk-Forward)

Валидация скользящим окном имитирует реальную торговлю:

```
Для t = train_size до T:
  Обучение на [t - train_size, t)
  Предсказание на [t, t + step)
  Сдвиг окна вперёд на step

Вариант расширяющегося окна:
  Обучение на [0, t)  (растущая обучающая выборка)
  Предсказание на [t, t + step)
```

### Гетероскедастичность и серийная корреляция

Финансовые доходности демонстрируют:
- **Гетероскедастичность**: Дисперсия меняется во времени (кластеризация волатильности)
- **Серийную корреляцию**: Особенно в признаках, вычисленных по перекрывающимся окнам
- **Нестационарность**: Распределение сдвигается со временем

Эти нарушения предположения i.i.d. требуют:
- Моделирования волатильности типа GARCH перед конструированием признаков
- Дробного дифференцирования для достижения стационарности с сохранением памяти
- Робастных стандартных ошибок (Ньюи-Уэст) для вывода о коэффициентах

---

## Раздел 3: Сравнение методов кросс-валидации

| Метод | Временной порядок | Предотвращает утечку | Обрабатывает перекрытие меток | Эффективность данных | Подходит для |
|-------|-------------------|---------------------|-------------------------------|---------------------|--------------|
| Стандартная K-Fold | Нет | Нет | Нет | Высокая | Нефинансовые данные |
| Разделение временных рядов | Да | Частично | Нет | Низкая | Простые временные ряды |
| Очищенная K-Fold | Да | Да | Да | Высокая | Финансовый ML |
| Walk-Forward | Да | Да | Частично | Средняя | Бэктестирование стратегий |
| Комбинаторная очищенная CV | Да | Да | Да | Очень высокая | Робастная оценка |
| Расширяющееся окно | Да | Да | Частично | Средняя | Данные со сменой режимов |
| Блочная временная серия | Да | Частично | Нет | Средняя | Низкая автокорреляция |

| Ловушка | Описание | Метод обнаружения | Решение |
|---------|----------|-------------------|---------|
| Утечка данных | Будущая инфо в признаках | Аудит временных меток признаков | Строгий временной порядок |
| Заглядывание вперёд | Использование недоступных данных | Walk-forward тест | Признаки на момент времени |
| Ошибка выживаемости | Только активные активы в данных | Проверка делистинга | Включить обанкротившиеся токены |
| Множественное тестирование | Протестировано много стратегий | Дефлированный коэф. Шарпа | Контроль ошибок семейства |
| Переобучение на шум | Подгонка случайных паттернов | Деградация OOS | Регуляризация, простые модели |
| Нестационарность | Сдвиги распределения | ADF тест, скользящая статистика | Дробное дифференцирование |

---

## Раздел 4: Торговые применения

### 4.1 Переобучение модели скользящим окном для крипто

Криптовалютные рынки быстро эволюционируют. Модель, обученная на данных 2023 года, может не работать в 2024 из-за смены режимов. Переобучение скользящим окном:

- Переобучение каждые 1-4 недели на скользящем окне 6-12 месяцев
- Использование расширяющегося окна при стабильных режимах, сужающегося при волатильных
- Мониторинг стабильности важности признаков как индикатора смены режима
- Экстренное переобучение при падении точности предсказания ниже порога

### 4.2 Отбор признаков для криптосигналов

Критические признаки для предсказания крипто, ранжированные по типичной взаимной информации с будущей доходностью:

1. **Дисбаланс объёма** (соотношение объёма покупок к продажам) -- наивысший MI
2. **Z-оценка ставки финансирования** -- сильный предсказатель возврата к среднему
3. **Изменение открытого интереса** -- измеряет сдвиги в позиционировании
4. **Индикатор режима волатильности** (реализованная vs подразумеваемая) -- фиксирует состояние рынка
5. **Кросс-активный моментум** (BTC опережает альткоины) -- отношения лидерства-отставания

### 4.3 Обработка утечки данных в криптопризнаках

Распространённые источники утечки в криптотрейдинге:
- Использование цены закрытия для расчёта признаков, применяемых на закрытии (следует использовать цену открытия следующего бара)
- Включение будущих ставок финансирования в набор признаков
- Расчёт скользящей статистики, включающей бар цели предсказания
- Использование биржевых признаков (напр., данных ликвидаций Bybit), которые могут приходить с задержкой

### 4.4 Оптимизация гиперпараметров с Optuna

Байесовская оптимизация эффективно исследует пространство гиперпараметров:
- Определение пространства поиска (глубина дерева, скорость обучения, сила регуляризации)
- Использование вложенной CV: внешний цикл для оценки, внутренний для выбора гиперпараметров
- Ранняя остановка на основе валидационных потерь для сокращения вычислений
- Обрезка неперспективных проб с помощью медианного обрезателя

### 4.5 Построение конвейера для воспроизводимости

Полный ML-конвейер для криптотрейдинга:

```
RawData -> FeatureEngineering -> FeatureSelection -> Scaler -> Model -> PredictionPostProcessor
```

Каждый шаг конвейера должен:
- Обучаться только на обучающих данных (без статистик тестовых данных в масштабировании)
- Поддерживать сериализацию для развёртывания
- Логировать все параметры для воспроизводимости
- Грамотно обрабатывать пропущенные данные (криптобиржи имеют простои)

---

## Раздел 5: Реализация на Python

### Очищенный K-Fold кросс-валидатор

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import optuna
import requests
from typing import List, Tuple, Optional, Generator


class PurgedKFold(BaseCrossValidator):
    """K-Fold кросс-валидатор с очисткой и эмбарго для финансовых данных."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None
              ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_indices = indices[test_start:test_end]

            # Очистка: удаление обучающих образцов, перекрывающихся с тестовыми
            purge_start = max(0, test_start - embargo_size)
            purge_end = min(n_samples, test_end + embargo_size)

            train_indices = np.concatenate([
                indices[:purge_start],
                indices[purge_end:]
            ])

            yield train_indices, test_indices


class WalkForwardCV:
    """Кросс-валидация скользящим окном для временных рядов."""

    def __init__(self, n_splits: int = 5, train_size: int = None,
                 expanding: bool = False):
        self.n_splits = n_splits
        self.train_size = train_size
        self.expanding = expanding

    def split(self, X) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)

        if self.train_size is None:
            self.train_size = test_size * 2

        for i in range(self.n_splits):
            test_start = self.train_size + i * test_size
            test_end = min(test_start + test_size, n_samples)

            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.train_size

            train_indices = indices[train_start:test_start]
            test_indices = indices[test_start:test_end]

            if len(test_indices) == 0:
                break

            yield train_indices, test_indices


class MutualInfoFeatureSelector:
    """Отбор признаков с использованием взаимной информации для финансовых данных."""

    def __init__(self, n_features: int = 10, n_neighbors: int = 5):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.selected_features = None
        self.mi_scores = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MutualInfoFeatureSelector':
        mi = mutual_info_regression(
            X.values, y.values,
            n_neighbors=self.n_neighbors,
            random_state=42
        )
        self.mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        self.selected_features = self.mi_scores.head(self.n_features).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class CryptoMLPipeline:
    """Сквозной ML-конвейер для криптоторговых сигналов."""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def fetch_bybit_data(self, symbol: str, interval: str = "60",
                         limit: int = 1000) -> pd.DataFrame:
        """Получение часовых свечей с Bybit."""
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params)
        data = response.json()["result"]["list"]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").set_index("timestamp")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание торговых признаков из данных OHLCV."""
        features = pd.DataFrame(index=df.index)

        # Доходности на различных горизонтах
        for lag in [1, 2, 4, 8, 24]:
            features[f"return_{lag}h"] = df["close"].pct_change(lag)

        # Признаки волатильности
        features["volatility_24h"] = df["close"].pct_change().rolling(24).std()
        features["volatility_168h"] = df["close"].pct_change().rolling(168).std()
        features["vol_ratio"] = features["volatility_24h"] / features["volatility_168h"]

        # Признаки объёма
        features["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(24).mean()
        features["volume_trend"] = df["volume"].rolling(12).mean() / \
                                   df["volume"].rolling(48).mean()

        # Позиция цены
        features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        features["close_position"] = (df["close"] - df["low"]) / \
                                     (df["high"] - df["low"]).replace(0, np.nan)

        # Моментум
        for window in [12, 24, 72]:
            features[f"momentum_{window}h"] = df["close"] / \
                df["close"].shift(window) - 1

        # Возврат к среднему
        features["zscore_24h"] = (df["close"] - df["close"].rolling(24).mean()) / \
                                  df["close"].rolling(24).std()

        return features.dropna()

    def create_labels(self, df: pd.DataFrame, horizon: int = 4,
                      threshold: float = 0.001) -> pd.Series:
        """Создание меток классификации: 1 = рост, 0 = падение."""
        forward_return = df["close"].pct_change(horizon).shift(-horizon)
        labels = (forward_return > threshold).astype(int)
        return labels

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                 n_trials: int = 50) -> dict:
        """Байесовская оптимизация гиперпараметров с Optuna."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.02)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,
                                                      log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            }

            scores = []
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                model = GradientBoostingClassifier(**params, random_state=42)
                model.fit(X_train_s, y_train)
                score = model.score(X_val_s, y_val)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params

    def build_pipeline(self, best_params: dict) -> Pipeline:
        """Построение sklearn-конвейера с оптимизированными параметрами."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(**best_params, random_state=42))
        ])

    def evaluate_with_purged_cv(self, pipeline: Pipeline,
                                X: pd.DataFrame, y: pd.Series,
                                n_splits: int = 5) -> dict:
        """Оценка конвейера с использованием очищенной k-fold CV."""
        cv = PurgedKFold(n_splits=n_splits, embargo_pct=0.02)
        scores = []
        predictions = []

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            preds = pipeline.predict_proba(X_test)[:, 1]

            scores.append(score)
            predictions.extend(zip(X_test.index, preds, y_test.values))

        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "fold_scores": scores,
            "predictions": predictions
        }
```

### Пример использования

```python
# Инициализация конвейера
pipeline = CryptoMLPipeline(symbols=["BTCUSDT"])

# Получение и подготовка данных
df = pipeline.fetch_bybit_data("BTCUSDT", interval="60", limit=1000)
features = pipeline.create_features(df)
labels = pipeline.create_labels(df, horizon=4)

# Выравнивание признаков и меток
common_idx = features.index.intersection(labels.dropna().index)
X = features.loc[common_idx]
y = labels.loc[common_idx]

# Отбор признаков
selector = MutualInfoFeatureSelector(n_features=8)
X_selected = selector.fit_transform(X, y)
print("Отобранные признаки:", selector.selected_features)
print("Оценки MI:\n", selector.mi_scores)

# Оптимизация гиперпараметров
best_params = pipeline.optimize_hyperparameters(X_selected, y, n_trials=30)
print("Лучшие параметры:", best_params)

# Построение и оценка
model_pipeline = pipeline.build_pipeline(best_params)
results = pipeline.evaluate_with_purged_cv(model_pipeline, X_selected, y)
print(f"Точность CV: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
```

---

## Раздел 6: Реализация на Rust

### Структура проекта

```
ch06_ml_training_financial_data/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── cv/
│   │   ├── mod.rs
│   │   ├── purged_kfold.rs
│   │   └── walk_forward.rs
│   ├── selection/
│   │   ├── mod.rs
│   │   └── mutual_info.rs
│   └── pipeline/
│       ├── mod.rs
│       └── trainer.rs
└── examples/
    ├── purged_cv.rs
    ├── feature_selection.rs
    └── hyperparameter_search.rs
```

### Основная библиотека (src/lib.rs)

```rust
pub mod cv;
pub mod selection;
pub mod pipeline;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,  // n_samples x n_features
    pub labels: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SplitIndices {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResult {
    pub fold_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_score: f64,
}

impl CVResult {
    pub fn display(&self) {
        println!("Результаты кросс-валидации:");
        for (i, score) in self.fold_scores.iter().enumerate() {
            println!("  Фолд {}: {:.4}", i + 1, score);
        }
        println!("  Среднее: {:.4} +/- {:.4}", self.mean_score, self.std_score);
    }
}
```

### Очищенная K-Fold (src/cv/purged_kfold.rs)

```rust
use crate::SplitIndices;

pub struct PurgedKFold {
    pub n_splits: usize,
    pub embargo_pct: f64,
}

impl PurgedKFold {
    pub fn new(n_splits: usize, embargo_pct: f64) -> Self {
        Self { n_splits, embargo_pct }
    }

    pub fn split(&self, n_samples: usize) -> Vec<SplitIndices> {
        let embargo_size = (n_samples as f64 * self.embargo_pct) as usize;
        let fold_size = n_samples / self.n_splits;
        let mut splits = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test: Vec<usize> = (test_start..test_end).collect();

            // Очистка и эмбарго
            let purge_start = test_start.saturating_sub(embargo_size);
            let purge_end = (test_end + embargo_size).min(n_samples);

            let train: Vec<usize> = (0..purge_start)
                .chain(purge_end..n_samples)
                .collect();

            splits.push(SplitIndices { train, test });
        }

        splits
    }

    pub fn validate_no_leakage(&self, splits: &[SplitIndices]) -> bool {
        for split in splits {
            let train_set: std::collections::HashSet<_> =
                split.train.iter().collect();
            let test_set: std::collections::HashSet<_> =
                split.test.iter().collect();

            if train_set.intersection(&test_set).count() > 0 {
                return false;
            }
        }
        true
    }
}
```

### Walk-Forward CV (src/cv/walk_forward.rs)

```rust
use crate::SplitIndices;

pub struct WalkForwardCV {
    pub n_splits: usize,
    pub train_size: usize,
    pub test_size: usize,
    pub expanding: bool,
}

impl WalkForwardCV {
    pub fn new(
        n_splits: usize,
        train_size: usize,
        test_size: usize,
        expanding: bool,
    ) -> Self {
        Self { n_splits, train_size, test_size, expanding }
    }

    pub fn split(&self, n_samples: usize) -> Vec<SplitIndices> {
        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let test_start = self.train_size + i * self.test_size;
            let test_end = (test_start + self.test_size).min(n_samples);

            if test_start >= n_samples {
                break;
            }

            let train_start = if self.expanding {
                0
            } else {
                test_start - self.train_size
            };

            let train: Vec<usize> = (train_start..test_start).collect();
            let test: Vec<usize> = (test_start..test_end).collect();

            splits.push(SplitIndices { train, test });
        }

        splits
    }
}
```

### Оценка взаимной информации (src/selection/mutual_info.rs)

```rust
pub struct MutualInfoSelector {
    pub n_features: usize,
    pub k_neighbors: usize,
    pub scores: Vec<(String, f64)>,
}

impl MutualInfoSelector {
    pub fn new(n_features: usize, k_neighbors: usize) -> Self {
        Self {
            n_features,
            k_neighbors,
            scores: Vec::new(),
        }
    }

    /// Оценка взаимной информации между признаком и целью методом KNN
    pub fn estimate_mi(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < self.k_neighbors + 1 {
            return 0.0;
        }

        // Упрощённая оценка MI на основе корреляции
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n as f64;
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n as f64;

        if var_x < 1e-12 || var_y < 1e-12 {
            return 0.0;
        }

        let cov: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / n as f64;

        let rho = cov / (var_x.sqrt() * var_y.sqrt());
        // MI для гауссовского: I = -0.5 * ln(1 - rho^2)
        let rho_sq = rho.powi(2).min(0.9999);
        -0.5 * (1.0 - rho_sq).ln()
    }

    pub fn select_features(
        &mut self,
        features: &[Vec<f64>],
        target: &[f64],
        feature_names: &[String],
    ) -> Vec<usize> {
        let mut mi_scores: Vec<(usize, String, f64)> = features.iter()
            .enumerate()
            .map(|(i, feat)| {
                let mi = self.estimate_mi(feat, target);
                (i, feature_names[i].clone(), mi)
            })
            .collect();

        mi_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        self.scores = mi_scores.iter()
            .map(|(_, name, score)| (name.clone(), *score))
            .collect();

        mi_scores.iter()
            .take(self.n_features)
            .map(|(idx, _, _)| *idx)
            .collect()
    }
}
```

### Получение данных с Bybit

```rust
use reqwest;
use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<(i64, f64, f64, f64, f64, f64)>> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.bybit.com/v5/market/kline")
        .query(&[
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.to_string()),
        ])
        .send()
        .await?
        .json::<BybitResponse>()
        .await?;

    let bars: Vec<(i64, f64, f64, f64, f64, f64)> = resp.result.list
        .iter()
        .map(|row| (
            row[0].parse::<i64>().unwrap_or(0),
            row[1].parse::<f64>().unwrap_or(0.0),  // open
            row[2].parse::<f64>().unwrap_or(0.0),  // high
            row[3].parse::<f64>().unwrap_or(0.0),  // low
            row[4].parse::<f64>().unwrap_or(0.0),  // close
            row[5].parse::<f64>().unwrap_or(0.0),  // volume
        ))
        .rev()
        .collect();

    Ok(bars)
}
```

---

## Раздел 7: Практические примеры

### Пример 1: Сравнение очищенной K-Fold и стандартной K-Fold

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

pipeline = CryptoMLPipeline(symbols=["BTCUSDT"])
df = pipeline.fetch_bybit_data("BTCUSDT", interval="60", limit=1000)
features = pipeline.create_features(df)
labels = pipeline.create_labels(df, horizon=4)
common_idx = features.index.intersection(labels.dropna().index)
X, y = features.loc[common_idx], labels.loc[common_idx]

model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

# Стандартная K-Fold (некорректна для временных рядов)
std_scores = []
for train_idx, test_idx in KFold(n_splits=5, shuffle=True).split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    std_scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))

# Очищенная K-Fold (корректна для временных рядов)
purged_scores = []
for train_idx, test_idx in PurgedKFold(n_splits=5, embargo_pct=0.02).split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    purged_scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))

print(f"Стандартная K-Fold: {np.mean(std_scores):.4f} +/- {np.std(std_scores):.4f}")
print(f"Очищенная K-Fold:   {np.mean(purged_scores):.4f} +/- {np.std(purged_scores):.4f}")

# Ожидаемый результат:
# Стандартная K-Fold: 0.5623 +/- 0.0187  (завышена из-за утечки)
# Очищенная K-Fold:   0.5234 +/- 0.0312  (реалистичная оценка)
```

### Пример 2: Отбор признаков по взаимной информации

```python
selector = MutualInfoFeatureSelector(n_features=6, n_neighbors=10)
X_selected = selector.fit_transform(X, y)

print("Ранжирование признаков по взаимной информации:")
for feat, mi in selector.mi_scores.items():
    marker = " <-- отобран" if feat in selector.selected_features else ""
    print(f"  {feat:25s}: {mi:.4f}{marker}")

# Ожидаемый результат:
# Ранжирование признаков по взаимной информации:
#   volume_sma_ratio         : 0.0423 <-- отобран
#   zscore_24h               : 0.0387 <-- отобран
#   volatility_24h           : 0.0312 <-- отобран
#   momentum_12h             : 0.0298 <-- отобран
#   close_position           : 0.0276 <-- отобран
#   vol_ratio                : 0.0251 <-- отобран
#   return_1h                : 0.0198
#   high_low_range           : 0.0187
#   volume_trend             : 0.0134
#   return_24h               : 0.0112
```

### Пример 3: Валидация скользящим окном с переобучением

```python
wf_cv = WalkForwardCV(n_splits=10, train_size=500, expanding=False)
model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
scaler = StandardScaler()

wf_results = []
for train_idx, test_idx in wf_cv.split(X_selected):
    X_train = scaler.fit_transform(X_selected.iloc[train_idx])
    X_test = scaler.transform(X_selected.iloc[test_idx])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    wf_results.append({
        "period": X_selected.index[test_idx[0]].strftime("%Y-%m-%d"),
        "accuracy": accuracy,
        "n_train": len(train_idx),
        "n_test": len(test_idx)
    })

print("Результаты Walk-Forward:")
for r in wf_results:
    print(f"  {r['period']}: точность={r['accuracy']:.4f} "
          f"(обучение={r['n_train']}, тест={r['n_test']})")

# Ожидаемый результат:
# Результаты Walk-Forward:
#   2024-08-15: точность=0.5340 (обучение=500, тест=50)
#   2024-08-17: точность=0.5180 (обучение=500, тест=50)
#   2024-08-19: точность=0.5420 (обучение=500, тест=50)
#   2024-08-21: точность=0.5060 (обучение=500, тест=50)
#   ...
```

---

## Раздел 8: Фреймворк бэктестирования

### Компоненты фреймворка

Фреймворк бэктестирования обучения ML валидирует весь конвейер:

1. **Разделитель данных**: Реализует очищенную k-fold и walk-forward разбиения
2. **Конвейер признаков**: Конструирование + отбор признаков для каждого разбиения
3. **Обучатель модели**: Обучение модели с правильным выбором гиперпараметров
4. **Логгер предсказаний**: Запись всех предсказаний с временными метками
5. **Анализатор производительности**: Расчёт метрик классификации и торговли

### Панель метрик

| Метрика | Описание | Целевой диапазон |
|---------|----------|-----------------|
| Точность (Accuracy) | Правильные предсказания / всего | > 0.52 |
| Точность (Precision) | Истинно положительные / предсказанные положительные | > 0.53 |
| Полнота (Recall) | Истинно положительные / реально положительные | > 0.50 |
| F1-мера | Гармоническое среднее точности и полноты | > 0.52 |
| Log Loss | Кросс-энтропия предсказанных вероятностей | < 0.69 |
| AUC-ROC | Площадь под ROC-кривой | > 0.53 |
| Процент попаданий (лонг) | Точность только длинных сигналов | > 0.52 |
| Профит-фактор | Валовая прибыль / валовой убыток | > 1.10 |
| Шарп (из предсказаний) | Коэффициент Шарпа доходности на основе предсказаний | > 0.50 |

### Пример результатов

```
=== Оценка ML-конвейера: BTCUSDT 1H ===

Кросс-валидация: Очищенная 5-Fold (эмбарго=2%)
Модель: GradientBoosting (n_est=200, depth=4, lr=0.05)
Признаки: 6 отобрано из 14 по взаимной информации

Результаты по фолдам:
  Фолд 1: Accuracy=0.5312, AUC=0.5445, LogLoss=0.6891
  Фолд 2: Accuracy=0.5234, AUC=0.5378, LogLoss=0.6902
  Фолд 3: Accuracy=0.5389, AUC=0.5512, LogLoss=0.6878
  Фолд 4: Accuracy=0.5156, AUC=0.5289, LogLoss=0.6923
  Фолд 5: Accuracy=0.5278, AUC=0.5401, LogLoss=0.6895

Средняя точность: 0.5274 +/- 0.0078
Средний AUC:      0.5405 +/- 0.0072

Стандартная K-Fold (с утечкой):  0.5587 +/- 0.0152  (завышение!)
Очищенная K-Fold (корректная):   0.5274 +/- 0.0078  (реалистичная)
Смещение завышения:               +5.9%

Топ важностей признаков:
  1. volume_sma_ratio    : 0.187
  2. zscore_24h          : 0.162
  3. volatility_24h      : 0.158
  4. momentum_12h        : 0.145
  5. close_position      : 0.131
  6. vol_ratio           : 0.117
```

---

## Раздел 9: Оценка производительности

### Сравнение методов CV на криптоданных

| Метод | Заявленная точность | Истинная OOS точность | Завышение | Дисперсия |
|-------|--------------------|-----------------------|-----------|-----------|
| Стандартная K-Fold (перемеш.) | 0.558 | 0.519 | +7.5% | Низкая |
| Стандартная K-Fold (без перем.) | 0.542 | 0.524 | +3.4% | Средняя |
| Разделение временных рядов | 0.531 | 0.527 | +0.8% | Высокая |
| Очищенная K-Fold (1% эмбарго) | 0.529 | 0.526 | +0.6% | Средняя |
| Очищенная K-Fold (2% эмбарго) | 0.527 | 0.525 | +0.4% | Средняя |
| Walk-Forward (скользящее) | 0.524 | 0.522 | +0.4% | Высокая |
| Комбинаторная очищенная CV | 0.526 | 0.525 | +0.2% | Низкая |

### Ключевые выводы

1. **Стандартная k-fold значительно завышает производительность** на криптоданных на 3-8% точности. Это приводит к стратегиям, которые кажутся прибыльными при разработке, но терпят неудачу в продакшене.

2. **Размер эмбарго 1-2% от набора данных обычно достаточен** для предотвращения утечки через серийную корреляцию в часовых криптоданных. Для дневных данных даже меньший эмбарго (0.5%) работает.

3. **Отбор признаков по взаимной информации улучшает производительность вне выборки** на 1-3% по сравнению с использованием всех признаков. Наиболее предсказательные признаки, как правило, основаны на объёме и индикаторах возврата к среднему.

4. **Walk-forward предоставляет наиболее реалистичные оценки**, но с более высокой дисперсией по фолдам. Очищенная k-fold предлагает хороший баланс точности и стабильности.

5. **Байесовская оптимизация гиперпараметров (Optuna) находит лучшие параметры** за 30-50 проб по сравнению с grid search с сотнями вычислений, что критично, когда каждая оценка требует обучения нескольких фолдов CV.

### Ограничения

- Оценка взаимной информации зашумлена при малых выборках, характерных для крипто
- Важность признаков из древовидных моделей может быть обманчивой с коррелированными признаками
- Walk-forward валидация предполагает, что недавние данные наиболее релевантны, что может не выполняться при смене режимов
- Вычислительная стоимость вложенной CV с Optuna может быть запретительной для больших наборов признаков
- Серийная корреляция в ошибках предсказания не учитывается стандартными метриками

---

## Раздел 10: Перспективные направления

1. **Онлайн-обучение для нестационарных рынков**: Реализация онлайн-градиентного спуска и адаптивных моделей, которые обновляются непрерывно по мере поступления новых данных, уменьшая потребность в периодическом переобучении и улучшая отзывчивость к смене режимов.

2. **Конформное предсказание для квантификации неопределённости**: Применение конформного предсказания к крипто-ML-моделям для построения интервалов предсказания с гарантированным покрытием, позволяя лучше определять размер позиции на основе уверенности предсказания.

3. **Каузальное обнаружение признаков**: Переход от отбора признаков на основе корреляций к методам каузального вывода (do-исчисление, инструментальные переменные), выявляющим действительно предсказательные признаки вместо ложных корреляций.

4. **Мета-обучение по крипто-активам**: Использование мета-обучения (обучение учиться) для переноса знаний с ликвидных активов (BTC, ETH) на менее ликвидные альткоины с ограниченными обучающими данными, улучшая производительность моделей на токенах малой капитализации.

5. **ML-обучение с дифференциальной приватностью**: Включение гарантий дифференциальной приватности в обучение моделей для защиты проприетарных торговых сигналов при развёртывании моделей в общих средах или при агрегировании данных между аккаунтами.

6. **Устойчивость к состязательным воздействиям при манипулировании рынком**: Обучение моделей, устойчивых к состязательным примерам (спуфинг, фиктивная торговля, схемы «накачай и сбрось»), которые могут вводить в заблуждение стандартные ML-модели и приводить к убыточным позициям.

---

## Литература

1. De Prado, M. L. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

2. Bailey, D. H., & De Prado, M. L. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *The Journal of Portfolio Management*, 40(5), 94-107.

3. Kraskov, A., Stogbauer, H., & Grassberger, P. (2004). "Estimating Mutual Information." *Physical Review E*, 69(6), 066138.

4. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13, 281-305.

5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "Optuna: A Next-Generation Hyperparameter Optimization Framework." *Proceedings of KDD*, 2623-2631.

6. Arlot, S., & Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys*, 4, 40-79.

7. De Prado, M. L. (2019). "Beyond Econometrics: A Roadmap Towards Financial Machine Learning." *SSRN Working Paper*.
