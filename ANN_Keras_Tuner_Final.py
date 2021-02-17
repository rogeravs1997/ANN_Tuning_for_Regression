import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation
from tensorflow import keras
from keras.models import Sequential
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
import kerastuner



# LISTS TO SAVE METRICS FOR FINAL COMPARISON
R2_values_train=[]
RMSE_values_train=[]
R2_values_test=[]
RMSE_values_test=[]
index=[]

####################### ANN ####################### 

# LOADING THE DATA FROM A .MAT FILE (MATLAB/OCTAVE FILE)
file_name = "limpiado"
main_path = ("D:\Desktop\Modelo Predictivo PPV\database")
file_path = (file_name + ".xlsx")
sheet_name = "data"
dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name)
dataset=dataset.sample(n=200,random_state=1)

# DEFINING FEATURES AND LABELS

X=dataset[["Diferencia_X","Diferencia_Y","Diferencia_Z","RQD",
            "UCS","# Taladros","Kg de columna explosiva"]]
# X=dataset[["Distancia (m)","Kg de columna explosiva"]]

y=dataset["PPV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3,shuffle=True)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
y_train.shape, y_test.shape = (-1, 1), (-1, 1)

# FEATURING SCALER
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)




# FUNCTION TO OPTIMIZE WITH HP
def build_model(hp):  # random search passes this hyperparameter() object

    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units=hp.Int('input_units',
                                min_value=2,
                                max_value=512,
                                step=2), 
                    activation = hp.Choice("activation",
                                           values=["linear",
                                                   "relu",
                                                   "sigmoid",
                                                   "softmax",
                                                   "softplus",
                                                   "softsign",
                                                   "tanh",
                                                   "selu",
                                                   "elu"]), 
                    use_bias=hp.Boolean("use_bias"),
                    input_dim = 7))
    
    # Adding n hidden layers
    for i in range(hp.Int("n_layers",0,10)):
        model.add(Dense(units=hp.Int(f"Layer_{i}_units",
                                    min_value=2,
                                    max_value=512,
                                    step=2), 
                        activation = hp.Choice(f"Layer_{i}_activation",
                                               values=["linear",
                                                       "relu",
                                                       "sigmoid",
                                                       "softmax",
                                                       "softplus",
                                                       "softsign",
                                                       "tanh",
                                                       "selu",
                                                       "elu"]), 
                        use_bias=hp.Boolean(f"Layer_{i}_use_bias")))
    

    
    # Adding the output layer
    model.add(Dense(units = 1,activation='linear'))
    
    # Optimizing the optimizer xD
    optimizer = hp.Choice("optimizador",values=["adam",
                                                "adamax",
                                                "adadelta",
                                                "adagrad",
                                                "ftrl",
                                                "nadam",
                                                "rmsprop",
                                                "sgd"])


    if optimizer=="adam":
        

        model.compile(optimizer =keras.optimizers.Adam(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1e-1,
                                                                                sampling="LOG",
                                                                                default=1e-3),
                                                        beta_1=hp.Float("beta_1",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.9),
                                                        beta_2=hp.Float("beta_2",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.999),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7),
                                                        amsgrad=hp.Boolean("amsgrad",
                                                                          default=False), clipvalue=5.0),
                      loss = 'mse')  

    elif optimizer=="adamax":
        

        model.compile(optimizer =keras.optimizers.Adamax(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1e-1,
                                                                                sampling="LOG",
                                                                                default=1e-3),
                                                        beta_1=hp.Float("beta_1",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.9),
                                                        beta_2=hp.Float("beta_2",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.999),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7), clipvalue=5.0),
                      loss = 'mse')  
    
    
    elif optimizer=="adadelta":
        

        model.compile(optimizer =keras.optimizers.Adadelta(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1,
                                                                                sampling="LOG",
                                                                                default=1),
                                                        rho=hp.Float("rho",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.95),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7), clipvalue=5.0),
                      loss = 'mse')  
    
    
    elif optimizer=="adagrad":
        

        model.compile(optimizer =keras.optimizers.Adagrad(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1,
                                                                                sampling="LOG",
                                                                                default=1),
                                                        initial_accumulator_value=hp.Float("initial_accumulator_value",
                                                                                            min_value=0,
                                                                                            max_value=1,
                                                                                            step=0.001,
                                                                                            default=0.1),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7), clipvalue=5.0),
                      loss = 'mse')  
    

    elif optimizer=="ftrl":
        

        model.compile(optimizer =keras.optimizers.Ftrl(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1e-1,
                                                                                sampling="LOG",
                                                                                default=1e-3),
                                                        learning_rate_power=hp.Float("learning_rate_power",
                                                                                    min_value=-1,
                                                                                    max_value=0,
                                                                                    step=0.001,
                                                                                    default=-0.5),
                                                        initial_accumulator_value=hp.Float("initial_accumulator_value",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.1),
                                                        l1_regularization_strength=hp.Float("l1_regularization_strength",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0),
                                                        l2_regularization_strength=hp.Float("l2_regularization_strength",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0),
                                                        l2_shrinkage_regularization_strength=hp.Float("l2_shrinkage_regularization_strength",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0), clipvalue=5.0),
                      loss = 'mse')  

        
    elif optimizer=="nadam":
        

        model.compile(optimizer =keras.optimizers.Nadam(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1e-1,
                                                                                sampling="LOG",
                                                                                default=1e-3),
                                                        beta_1=hp.Float("beta_1",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.9),
                                                        beta_2=hp.Float("beta_2",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.999),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7), clipvalue=5.0),
                      loss = 'mse')  
        
    elif optimizer=="rmsprop":
        

        model.compile(optimizer =keras.optimizers.RMSprop(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-3,
                                                                                max_value=1e-1,
                                                                                sampling="LOG",
                                                                                default=1e-3),
                                                        rho=hp.Float("rho",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0.9),
                                                        momentum=hp.Float("momentum",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0),
                                                        epsilon=hp.Float("epsilon",
                                                                        min_value=1e-9,
                                                                        max_value=1e-3,
                                                                        sampling="LOG",
                                                                        default=1e-7),
                                                        centered=hp.Boolean("centered",
                                                                          default=False), clipvalue=5.0),
                      loss = 'mse')  
    
    elif optimizer=="sgd":
        

        model.compile(optimizer =keras.optimizers.SGD(learning_rate=hp.Float("learning_rate",
                                                                                min_value=1e-5,
                                                                                max_value=1,
                                                                                sampling="LOG",
                                                                                default=1e-2),
                                                        momentum=hp.Float("momentum",
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        step=0.001,
                                                                        default=0),
                                                        nesterov=hp.Boolean("nesterov",
                                                                          default=False), clipvalue=5.0),
                      loss = 'mse')  
        
    

    return model

# CLASS TO OPTIMIZE BATCH SIZE
class MyTuner(kerastuner.tuners.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 200, 200, step=10)

    super(MyTuner, self).run_trial(trial, *args, **kwargs)


tuner=MyTuner(build_model,
                   objective="val_loss",
                   max_trials=20, 
                   executions_per_trial=1, 
                   overwrite=True,
                   directory="D:\Desktop\Modelo Predictivo PPV\KERAS TUNER")

tuner.search(x=X_train,
              y=y_train,
              verbose=1,
              epochs=500, 
              callbacks=[EarlyStopping(monitor='val_loss', 
                                       patience=2,
                                       mode="min",
                                       restore_best_weights=True,
                                       min_delta=1)],
              validation_data=(X_test, y_test))


print(tuner.get_best_hyperparameters()[0].values)
model=tuner.get_best_models(num_models=1)[0]

# saving the model
# model.save("directory")

pred_test = model.predict(X_test)

pred_train=model.predict(X_train)



RMSE_values_test.append(MSE(y_test,pred_test)**(1/2))
R2_values_test.append(r2_score(y_test,pred_test))

RMSE_values_train.append(MSE(y_train,pred_train)**(1/2))
R2_values_train.append(r2_score(y_train,pred_train))
index.append("ANN")

data={'RMSE Test':RMSE_values_test,'R2 test':R2_values_test,'RMSE_train':RMSE_values_train,"R2_train":R2_values_train}

table=pd.DataFrame(data,index=index)

print(table)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(pred_test, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


