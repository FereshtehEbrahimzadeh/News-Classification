from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd



def simple_model():
    import keras
    import pandas as pd
    
    from keras.models import Sequential
    from keras.layers import Dense,Activation,BatchNormalization,Dropout
    import matplotlib.pyplot as plt
    train_data = pd.read_excel (r'train.xlsx')
    #df_train = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])

    x_train=train_data.iloc[:,1:]
    y_train=train_data.iloc[:,0:1]


    test_data = pd.read_excel (r'test.xlsx')
    #df_test = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])

    x_test=test_data.iloc[:,1:]
    y_test=test_data.iloc[:,0:1]

    # make model & compile
    model=Sequential()
    model.add(Dense(15, input_dim=9))
    #model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10 ))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Dense(5))
    #model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
   
    o=keras.optimizers.RMSprop(0.0001)
    model.compile(optimizer=o, loss='mse',metrics=['mean_absolute_error','mean_squared_error'])



    return model
# read file

#y=keras.utils.to_categorical(t)

# train model


#################
def hybrid_model():
    import pandas as pd
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import matplotlib.pyplot as plt
    from keras import optimizers
    from keras import  regularizers

    from keras.layers import Dense,Activation,BatchNormalization,Dropout

    train_data = pd.read_excel (r'train.xlsx')
    #df_train = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])

    x_train=train_data.iloc[:,1:]
    y_train=train_data.iloc[:,0:1]


    test_data = pd.read_excel (r'test.xlsx')
    #df_test = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])

    x_test=test_data.iloc[:,1:]
    y_test=test_data.iloc[:,0:1]

    encoding_dim =2

    input_ = Input(shape=(9,))
    #1 15
    encoded = Dense(5, activation='relu')(input_)
    #encoded = Dropout(0.33)(encoded)
    #encoded=Dropout(0.5)(encoded)
    #encoded=Dropout(0.9)(encoded)
    #encoded = Dense(15, activation='relu')(encoded)
    #encoded=Dropout(0.7)(encoded)
    #encoded=BatchNormalization()(encoded)
    encoded = Dense(15, activation='relu')(encoded)
    #encoded=BatchNormalization()(encoded)
    #2 10
    encoded = Dense(10, activation='linear')(encoded)
    #encoded=Dropout(rate=0.9)(encoded)
    #encoded=BatchNormalization()(encoded)
    #encoded=Dropout(0.8)(encoded)
    #encoded=BatchNormalization()(encoded)
    #3 5

    encoded = Dense(5, activation='relu')(encoded)
    encoded=BatchNormalization()(encoded)
    #encoded=Dropout(0.5)(encoded)
    #encoded=Dropout(0.5)(encoded)

    #encoded = Dense(, activation='sigmoid')(encoded)
    #encoded=BatchNormalization()(encoded)
    #encoded = Dense(4, activation='relu')(encoded)
    #encoded=BatchNormalization()(encoded)
    #4 2
    mlp = Dense(1,activation='relu')(encoded)
    encoder = Model(input_, mlp)


    o=keras.optimizers.RMSprop(0.0001)


    encoder.compile(optimizer='sgd', loss='mse',metrics=['mean_absolute_error','mean_squared_error'])
    



    return encoder





train_data = pd.read_excel (r'train.xlsx')
#df_train = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])
x_train=train_data.iloc[:,1:]
y_train=train_data.iloc[:,0:1]


test_data = pd.read_excel (r'test.xlsx')
#df_test = pd.DataFrame(train_data,columns= ['y','x1','x2','x3','x4','x5','x6','x7'])

x_test=test_data.iloc[:,1:]
y_test=test_data.iloc[:,0:1]

    

from sklearn.metrics import roc_curve, auc
x_test_1 = test_data.iloc[:,1:]
y_test_1 = test_data.iloc[:,0]
x_train_1 =train_data.iloc[:,1:]
y_train_1 =train_data.iloc[:,0]
y_pred_1_train = [0,0,0]
fpr = [0,0,0]
tpr = [0,0,0]
thresholds = [0,0,0]
roc_auc = [0,0,0]
predlst=[]
i = 0
for model in [hybrid_model,simple_model]:
    mod=model()
    mod.fit(x_train_1, y_train_1,
                        epochs=100,
                  batch_size=10240,
                     shuffle=False,
                    callbacks=None,
                         verbose=1,
                    validation_data=(x_test_1, y_test_1))
    y_pred_1_train[i] = mod.predict(x_train_1)

    
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_train_1, y_pred_1_train[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    i+=1
    
#plt.title('Receiver Operating Characteristic - Test')


plt.title('Receiver Operating Characteristic - Train')
plt.plot(fpr[0], tpr[0], 'b', label = 'Autoencoder-MLP-AUC = %0.3f' % roc_auc[0])
plt.plot(fpr[1], tpr[1], 'g', label = 'MLP-AUC = %0.3f' % roc_auc[1])
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Roc_train.png')
plt.close()

    


    
