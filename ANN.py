import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("campaign_data.csv")  # pandas kütüphanesi aracılığıyla veriler okunur
print(df.head()) # kontrol için pandas tan head fonksiyonu ile verinin ilk bir kaç satırı döndürülür(özellikler gösterilir)

df.drop(columns = ['CampaignID','RaisedAmount'], inplace= True ) # pandas tan drop fonksiyonu ile işe yaramayan özellikler çıkartıldı
print(df.head())                                                   # peki işe yaramayanı hangi referansa göre belirledin?
                                                                 # analizini yapmak istediğim şey şu: ! belirli özelliklere göre destekleyen kişi sayısını tahmin etmek !

print(df.isnull().sum()) # yine pandastan isnull fonksiyonunu çağırıyoruz, bu fonksiyon df nin her bir hücresinde eksik değer olup olmadığını sorgular,
                         # eksikse true eksik değilse false değeri döndürür isnull, sum fonksiyonu True değerlerini 1 olarak sayar ve her sütub için eksik değerlerin toplamını verir
                         # hepsinin değeri 0 çıktı, yani her şey normal eksik hücre bulunmuyor devam edilebilir
                         # peki, eğer birinde 1 değeri dönerse ne olur? demekki bir hücre boş, bu eksik veriyi silebiliriz::satır silmek için::[df.dropna(inplace=True)]::sütün silmek için::[df.drop(columns=['column_name'], inplace=True)], doldurabiliriz::[df.fillna(value=0, inplace=True)]
print(df.describe()) # count: Her sütundaki geçerli (eksik olmayan) veri sayısını gösterir.
                     # mean: Her sütunun ortalama değerini hesaplar.
                     # std: Her sütunun standart sapmasını hesaplar.
                     # min: Her sütunun minimum değerini gösterir.
                     # 25% (1. çeyrek): Verilerin yüzde 25'inin altında olduğu değeri gösterir.
                     # 50% (medyan): Verilerin yüzde 50'sinin altında olduğu değeri gösterir.
                     # 75% (3. çeyrek): Verilerin yüzde 75'inin altında olduğu değeri gösterir.
                     # max: Her sütunun maksimum değerini gösterir.
print(df.info())  # boş hücreleri, boş olmayan hücre sayısını vererek belirtir
                  # veri tipini döndürür

print(df["Country"].value_counts()) # value_counts fonksiyonu, seçtiğimiz bir özelliğin farklı değerlerinin sayısını çıkartır

# görselleştirme
plt.figure(figsize=(10, 6))
df['Country'].value_counts().plot(kind='bar')
plt.xlabel('Counrties Names')
plt.ylabel('Numbers of Campaign')
plt.title("Comparsion betweeen Countries and Campaign")
plt.savefig('output.png')
from IPython.display import Image, display
display(Image(filename='output.png'))


df= pd.get_dummies(df, columns= ['Category', 'LaunchMonth', 'Country', 'Currency', 'VideoIncluded'], drop_first=True) # get_dummies kategorik verileri sayıya dönştürür
                                                                          # kategorik değişkenlerin her bir benzersiz değeri için bir sütun oluşturur ve bu sütunlara 1 veya 0 değerleri atar
                                                                          # "one_hot_encoding" denir buna
print(df.head())                                                          # yeni veri ayrımını yaptık, çıktıda şu göze çarpıyor: artık 31 sütun var. neden? kategorik verileri makinanın anlayabileceği değerlere dönüştürmke için belirli özellikleri parçalara ayırıp True False değerlerine atadık



X=df.drop(columns=['IsSuccessful']) # modelin özellikleri için yeni bir DataFrame(X)
y=df['IsSuccessful']                # hedef değişkeni seri olarak saklar(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=3) # veri kümesini test ve eğitim setlerine ayırıyoruz

from sklearn.preprocessing import StandardScaler
sc=StandardScaler() # özellikleri ölçeklendirdik
X_train=sc.fit_transform(X_train) # eğitim veri kümesinin ortalama ve standart sapmasını hesaplar ve bnlarla x_train i standartlaştırır, en sonda her bır özellıgın ortalaması 0 standart sapması 1 olarak ayarlanır
X_test=sc.transform(X_test)  # eğitim verilerinde hesaplanan ortalama ve standart sapmayı kullanarak X_test veri kümesini standartlaştırır
print(X_train)
print(y_train)

import tensorflow as tf
from tensorflow.keras.models import Sequential # ANN modelini başlatmak için kullanılır
from tensorflow.keras.layers import Dense, InputLayer # farklı katman yapısı için kullanılırr

classifier = Sequential()

classifier.add(InputLayer(input_shape=(X_train.shape[1],)))

classifier.add(Dense(50,activation='relu'))
classifier.add(Dense(50,activation='relu'))
classifier.add(Dense(50,activation='relu'))
classifier.add(Dense(50,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
print(classifier.summary())



classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])


history = classifier.fit(X_train,y_train,batch_size=32,epochs=100,verbose=1,validation_split=0.25)


   
