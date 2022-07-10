# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:23:54 2021

@author: 94541
"""
### training in groups
for i in range(2):  
  a="{}".format(i)
  print(a)
  checkpoint_path = r"C:/Users/94541/Desktop/check/training"
  file_name = checkpoint_path + a + "\\cp1-{epoch:04d}.ckpt"


for m in range(2):
    print("{}次训练".format(m))
    model.load_weights(r'C:\Users\94541\Desktop\checkpoint./normal')
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['chang'+str(m)]=[]

    for n in range(10):  
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        
        w,b=model.layers[0].get_weights()
        w_,b_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,b,w_,b_))
        locals()['chang'+str(m)]=np.hstack((locals()['chang'+str(m)],temp))
        
for i in range(2):  
  a="{}".format(i)
  print(a)
  checkpoint_path = r"C:/Users/94541/Desktop/check/training"
  file_name = checkpoint_path + a + "\\cp1-{epoch:04d}.ckpt"


for m in range(2):
    print("{}次训练".format(m))
    model.load_weights(r'C:\Users\94541\Desktop\checkpoint./normal')
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['chang'+str(m)]=[]

    for n in range(10):  
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        
        w,b=model.layers[0].get_weights()
        w_,b_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,b,w_,b_))
        locals()['chang'+str(m)]=np.hstack((locals()['chang'+str(m)],temp))


for m in range(2):
    print("{}次训练".format(m))
    model.load_weights(r"C:\Users\94541\Desktop\check\trainning\cp1-0100.ckpt")
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['chang'+str(m)]=[]

    for n in range(10):  
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        
        w,b=model.layers[0].get_weights()
        w_,b_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,b,w_,b_))
        locals()['chang'+str(m)]=np.hstack((locals()['chang'+str(m)],temp))





for m in range(1):
    print("{}次训练".format(m))
    model=build_model()
    model.load_weights(r"C:\Users\94541\Desktop\check\trainning\cp1-0100.ckpt")
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['sum_of_weights'+str(m)]=[]
    for n in range(10):          
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        w,b=model.layers[0].get_weights()
        w_,b_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,b,w_,b_))
        print(temp)
        locals()['sum_of_weights'+str(m)]=np.hstack((locals()['sum_of_weights'+str(m)],temp))
        
        
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(6,)),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
df=pd.read_excel(r"C:/Users/94541/Desktop/flooding.xlsx")
df_labels=df.pop('target')      
df_normalization=normalization(df)
for i in range(1):
    min_df= df_normalization[i*10:(i+1)*10]
    locals()['normal_train_dataset'+str(i)]=min_df
for i in range(1):
    min_labels= df_labels[i*10:(i+1)*10]
    locals()['normal_train_labels'+str(i)]=min_labels
for m in range(10):
    print("{}次训练".format(m))
    model=build_model()
    model.load_weights(r'C:\Users\94541\Desktop\checknormal./1')
    a= locals()['normal_train_dataset'+str(m)]
    b= locals()['normal_train_labels'+str(m)]
    locals()['sum_of_weights'+str(m)]=[]
    for n in range(10):          
        model.fit(a[n:n+1],b[n:n+1],
                  epochs=1, verbose=1)
        w,c=model.layers[0].get_weights()
        w_,c_=model.layers[1].get_weights()
        w=w.flatten()
        w_=w_.flatten()
        temp=np.hstack((w,c,w_,c_))
        locals()['sum_of_weights'+str(m)]=np.hstack((locals()['sum_of_weights'+str(m)],temp))

com=sum_of_weights0
for x in range(1):
    com=np.vstack((com,locals()['sum_of_weights'+str(x)]))
com=pd.DataFrame(com)
com.to_excel(r'C:\Users\94541\Desktop\floodingwithoutnoise.xlsx')


