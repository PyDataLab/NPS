import joblib
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split

# Загрузка данных из файла Excel
df = pd.read_excel('data.xlsx')

# Преобразование данных из широкого формата в длинный формат
df_melt = df.melt(id_vars='Unnamed: 0', var_name='date', value_name='nps')

# Отбор только тех строк, где 'Unnamed: 0' равно 'NPS'
df_melt = df_melt[df_melt['Unnamed: 0'] == 'NPS']

# Удаление столбца 'Unnamed: 0'
df_melt = df_melt.drop(columns='Unnamed: 0')

# Получение списка значений NPS
nps_values = df_melt['nps'].tolist()

# Создание нового DataFrame с датами и значениями NPS
data = {
    'date': pd.date_range(start='2021-01', periods=len(nps_values), freq='ME'),  # создание диапазона дат с месячной частотой
    'nps': nps_values  # использование полученного списка значений NPS
}
df = pd.DataFrame(data)  # создание нового DataFrame

# Преобразуем столбец 'date' в DataFrame df в формат datetime
df['date'] = pd.to_datetime(df['date'])

# Удаляем информацию о часовом поясе из столбца 'date'
df['date'] = df['date'].dt.tz_localize(None)

# Создаем копию DataFrame
df_copy = pd.DataFrame(data)

# Устанавливаем столбец 'date' в качестве индекса DataFrame
df_copy.set_index('date', inplace=True)

# Производим сезонное разложение данных NPS с периодом в 12 месяцев
decomposed = seasonal_decompose(df_copy['nps'], period=12)

# Извлекаем сезонную компоненту из разложения
seasonal = decomposed.seasonal

# Добавляем сезонную компоненту в DataFrame df_copy как новый столбец 'seasonality'
df_copy['seasonality'] = decomposed.seasonal

# Преобразуем значения сезонности в список
seasonality_values = df_copy['seasonality'].tolist()

# Добавляем значения сезонности в исходный DataFrame df как новый столбец 'seasonality'
df['seasonality'] = seasonality_values

# Добавляем новый столбец 'adjustment' в DataFrame df и заполняем его нулями
df['adjustment'] = 0.0

'''
В этой части кода выше применяем всё то же самое, что и в файле model_training.ipynb для изменения периодичности.
'''

# Функция для создания признаков в датафрейме
def make_features(df, max_lag, rolling_mean_size):
    # Создание признаков 'year' и 'month' из даты
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Создание лаговых признаков
    for lag in range(1, max_lag + 1):
        df['lag_{}'.format(lag)] = df['nps'].shift(lag)

    # Создание скользящего среднего
    df['rolling_mean'] = df['nps'].shift().rolling(window=rolling_mean_size).mean()

# Применение функции для создания признаков
make_features(df, 7, 6)

# Разделение данных на обучающую и тестовую выборки
train, test = train_test_split(df, shuffle=False, test_size=0.2)
# Удаление пропусков в обучающей выборке
train = train.dropna()

# Создание матриц признаков и векторов целей
X_train = train.drop('nps', axis=1)
y_train = train['nps']
X_test = test.drop('nps', axis=1)
y_test = test['nps']

# Добавление признаков 'year' и 'month' в обучающую выборку
X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month

# Удаление признака 'date' из обучающей и тестовой выборок
X_train = X_train.drop('date', axis=1)
X_test = X_test.drop('date', axis=1)

# Загрузка обученной модели
best_rf = joblib.load('linear_regression_model.pkl')

# Создание датафрейма для предсказания следующего месяца
next_month = pd.DataFrame({
    'date': pd.to_datetime(['2024-04']),
    'year': [2024],
    'month': [4]
})

# Создание лаговых признаков для следующего месяца
for lag in range(1, 8):
    next_month['lag_{}'.format(lag)] = df['nps'].iloc[-lag]

# Создание скользящего среднего для следующего месяца
next_month['rolling_mean'] = df['nps'].iloc[-6:].mean()

# Удаление признака 'date' из датафрейма следующего месяца
next_month = next_month.drop('date', axis=1)

# Добавление признака 'adjustment' в датафрейм следующего месяца
next_month['adjustment'] = 0.0

'''
В этом месте происходит ручное влияние на NPS на следующий месяц: если ожидается рост NPS на 10%, то ставим next_month['adjustment'] = 0.1.
Или, если, например, ожидается падение  NPS на 3% в следующем месяце, то ставим next_month['adjustment'] = -0.03
'''

# Получение номера следующего месяца
next_month_num = pd.to_datetime('2024-04').month

# Получение сезонности следующего месяца
next_month_seasonality = df.loc[df['date'].dt.month == next_month_num, 'seasonality'].values[-1]

# Добавление признака 'seasonality' в датафрейм следующего месяца
next_month['seasonality'] = next_month_seasonality

# Приведение датафрейма следующего месяца к формату обучающей выборки
next_month = next_month[X_train.columns]

# Предсказание NPS на следующий месяц
prediction = best_rf.predict(next_month)

# Вывод предсказания на экран
print("Предсказание NPS на следующий месяц:", prediction[0])

# Создание списка будущих месяцев
future_months = pd.date_range(start='2024-05', periods=12, freq='ME')

'''
Здесь можем изменить количество месяцев (или периодов) на долгосрочный прогноз:
future_months = pd.date_range(start='2024-05', periods=12, freq='ME')
Сейчас установлен долгосрочный прогноз (от 2 и более месяцев или периодов), начиная с 2024-05 на ближайшие 12 месяцев.
'''

# Список для хранения предсказаний
predictions = []

# Цикл по будущим месяцам
for date in future_months:
    # Создание датафрейма для каждого будущего месяца
    next_month = pd.DataFrame({
        'date': pd.to_datetime([date]),
        'year': [date.year],
        'month': [date.month]
    })

    # Создание лаговых признаков для каждого будущего месяца
    for lag in range(1, 8):
        next_month['lag_{}'.format(lag)] = df['nps'].iloc[-lag]

    # Создание скользящего среднего для каждого будущего месяца
    next_month['rolling_mean'] = df['nps'].iloc[-6:].mean()

    # Удаление признака 'date' из датафрейма каждого будущего месяца
    next_month = next_month.drop('date', axis=1)

    # Добавление признака 'adjustment' в датафрейм каждого будущего месяца
    next_month['adjustment'] = 0.0

    # Получение номера каждого будущего месяца
    next_month_num = date.month

    # Получение сезонности каждого будущего месяца
    next_month_seasonality = df.loc[df['date'].dt.month == next_month_num, 'seasonality'].values[-1]

    # Добавление признака 'seasonality' в датафрейм каждого будущего месяца
    next_month['seasonality'] = next_month_seasonality

    # Приведение датафрейма каждого будущего месяца к формату обучающей выборки
    next_month = next_month[X_train.columns]

    # Предсказание NPS на каждый будущий месяц
    prediction = best_rf.predict(next_month)
    predictions.append(prediction[0])

# Вывод предсказаний на экран
print("Предсказания NPS на следующие 12 месяцев:", predictions)
