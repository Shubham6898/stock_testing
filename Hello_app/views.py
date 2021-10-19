from django.shortcuts import render,HttpResponse


# Create your views here.
def index(request):
    #return HttpResponse("This is home page")

    return render(request,'index.html') 

def about(request):
    #return HttpResponse("This is about page.")
    return render(request,'about.html') 
def contact(request):
    #return HttpResponse("This is contact page.")
    return render(request,'contact.html') 

def prediciton(request):
    
    #return HttpResponse("This is service page.")
    return render(request,'prediciton.html') 






def apple(request):
    """Shubham Agawane - 2019"""
    from pandas_datareader import data as pdr

    import pandas as pd
    import numpy as np
    import numpy
        

    def fetch_stock_data():
        key="7ff53969f09e08e6564882582847ee3cbacc55d8"
        df = pdr.get_data_tiingo('AAPL', api_key=key)
        return df

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    """LSTM model development"""
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from sklearn.preprocessing import MinMaxScaler


    df = fetch_stock_data()
        
    df1 = df.reset_index()['close']
        
    scaler=MinMaxScaler(feature_range=(0,1))
        
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
        
    training_size=int(len(df1)*0.65)
        
    test_size=len(df1)-training_size
        
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

        # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
        # deploying model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

        # fitting model
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=30,batch_size=64,verbose=1)

        ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

        ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
        ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    math.sqrt(mean_squared_error(y_train,train_predict))
        ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))
        # plotting
    import matplotlib.pyplot as plt
    look_back=100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
    '''plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()'''
    
    x_input=test_data[341:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

        # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=99
    i=0
    while(i<30):
            
        if(len(temp_input)>99):
                #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
                #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    scaler.inverse_transform(lst_output)

    print(scaler.inverse_transform(lst_output))

    '''plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.show()'''
    plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    fig=plt.gcf()
    import io
    import urllib, base64
    buf=io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string=base64.b64encode(buf.read())
    url=urllib.parse.quote(string)
    return render(request, 'apple.html', {'data':url})



def microsoft(request):
    
    from pandas_datareader import data as pdr

    import pandas as pd
    import numpy as np
    import numpy
        

    def fetch_stock_data():
        key="7ff53969f09e08e6564882582847ee3cbacc55d8"
        df = pdr.get_data_tiingo('MSFT', api_key=key)
        return df

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    """LSTM model development"""
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from sklearn.preprocessing import MinMaxScaler


    df = fetch_stock_data()
        
    df1 = df.reset_index()['close']
        
    scaler=MinMaxScaler(feature_range=(0,1))
        
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
        
    training_size=int(len(df1)*0.65)
        
    test_size=len(df1)-training_size
        
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

        # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
        # deploying model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

        # fitting model
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=45,batch_size=50,verbose=1)

        ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

        ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
        ### Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    math.sqrt(mean_squared_error(y_train,train_predict))
        ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))
        # plotting
    import matplotlib.pyplot as plt
    look_back=100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
    '''plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()'''
    
    x_input=test_data[341:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

        # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=99
    i=0
    while(i<30):
            
        if(len(temp_input)>99):
                #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
                #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    scaler.inverse_transform(lst_output)

    print(scaler.inverse_transform(lst_output))

    '''plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.show()'''
    '''plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))'''
    df3=df1.tolist()
    df3.extend(lst_output)
    plt.plot(df3[1200:])
    fig=plt.gcf()
    import io
    import urllib, base64
    buf=io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string=base64.b64encode(buf.read())
    url=urllib.parse.quote(string)
    return render(request, 'microsoft.html', {'data':url})


    

    

	