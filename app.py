import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Carrega o arquivo csv para um pandas dataset
dataset = pd.read_csv('petr4.csv')
# Converte o campo Date de string para o formato datetime
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Separa Valores lidos, e valor a ser previsto
atributos = ['Open', 'High', 'Low', 'Volume']
# Variável a ser prevista
atrib_prev = ['Close']
# Criando objetos
x = dataset[atributos].values
y = dataset[atrib_prev].values

# Divide 75% dos dados aleatoriamente para treino e o restante para teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, random_state=42)

##Treinamento da máquina
# Modelo de regressão linear
modelo = LinearRegression()
# Treina o modelo
modelo.fit(X_treino, y_treino)


# Predizer 10 resultados
print(modelo.predict(X_teste[:10]))

# Validando o modelo
# usando mean_squared_error
RMSE = mean_squared_error(y_teste, modelo.predict(X_teste))**0.5
print("Média de erro {0}".format(RMSE))
