import numpy as np  #bilimsel hesaplamalar için özellikle veri manipülasyonu ve olusturma işlemlerinde
import matplotlib.pyplot as plt #grafik çizim kütüphansei  modelın kaybının gorsellestırılmesı ıcın kullanılack

import tensorflow as tf  #derın ogrenme kutuphanesı burdan derın ogrenme modellerı olusturup egıtıcez
from tensorflow.keras import layers #layers modulunu sınır agı katmanları olusturmak ıcın kullanıca<

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #MNIST veri setini yükledik.60000 train verisi ve 10000 test verisi var. pixeli 28*28
x_train = x_train/255
x_test = x_test/255  #gorutnu verılerını 0 ile 1 arasında normalıze eder
#fikir şu verı setınde goruntuler 28*28 piksel boyutunda gelıyor her pıksel degerı 0 ıle 255 arasında bır tam sayıdır bu degerlerı 255 e bolerek her degerı 0 ile 1 arasında bır degere donusturur
#daha hızlı öğrenmesine ve daha iyi genelleştirme 

"""print(x_train.shape) # bu verilerin nasıl yapılandırıldığına bakmamamızı .shape saglar.                                                                        
                    #cıktı:(60000,28,28) 28*28 pıksellık 60000 verı var
                    #sunu fark et burda 3 tane ozellıgı var yanı bu 3 boyutlu bır nesnedır 
print(y_train.shape)  # bununciktisi (60000) yanı bu kadar label oldugunu gosterır  

plt.imshow(x_train[0], cmap='Greys') # 0. ındextekı goruntuyu gorsellıstırme 
                    #yukarıda x_train 3 boyutlu bı nesnedır dedık ve x_train[0] ifadesi ile elde edılen şey ıkı boyutlu bı nesne oluyır .28*28 bıyutunda bı nesne                    
plt.show() """               


#asagıdakı sınır agı modelını olusturmak ıcın kullanılır.modelın katmanlarını sıralı bır sekılde olusturmak ıcın kullanılır
#her bır katman yalnızca bır oncekı katmanın cıktısını alır. basit ve sıralı veri işleme için uygun
model_lr = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    layers.Flatten(),#çok boyutlu gırdı versını tek boyutlu bır vektore donusturur.
                     #ve burda 28x28 piksellik bir görüntüyü 784 boyutlu bir vektöre dönüştürur
                     #sonrakı fully coneccted katmanlarda gereklıdır.
    layers.Dense(10, activation='softmax')#bu katmanda 10 noron var  yanı 10 cıkıs uretıcek. sınıf sayısı kadar noron genellıkle.
                                          #her sınıfın olasılık degerını uretır.
])                                        #ozetle:. dense katmanı flatten katmanının duzlestırdıgı verıyı(28*28=784 boyutlu) alır 10 cıktı degerıne donusturur 
   #Dense katmanı tam baglı fully connected bır katmandır. bunun olayı şu her bir noronun bır oncekı katmandakı tum noronlara baglı oldugu anlamına gelır.                    
   #bu katmanın parametrelerı : units(noron sayısı), activation(aktıvasyon fonksiyonu , input_shape(gırdı seklı))               
   #                        katmandakı noron sayısı(ne kadar noron varsa o kadar cıkıs uretıcek), noronların cıksıını donusturmek ıcın kullanılan fonksıyon(agın ogrenme yetenegını artırır),ilk katmanda kullanılır ve gıtfı verısının seklını belırtır.
   #                   (sınıflandırma problemlerinde sınıf sayısı kadar nöron kullanılır)                    
   #                    (softmax bu bır aktıvasyon fonksıyonu  ve cok sınıflı sınıflandırma problemlerınde kullanılır. her bır sınıfın olasılık degerını uretir
   #                    ()
model_lr.compile(     #egitim baslamadan once gereklı yapılandırmaları yapar
    optimizer='adam', #optımızasyon algorıtması
    loss='categorical_crossentropy',#kayıp fonksıyonu  gercek etıketler ıle model tahmınlerı arasındakı farkı olceer:loss
    metrics=['accuracy'])  #performans metrıgı   
model_lr.summary()                

y_onehot_train = tf.one_hot(y_train, 10)# Bir veri kümesindeki her sınıf, tüm sınıflar için ayrı bir pozisyonda '1' değerine sahip ve diğer pozisyonlarda '0' değerine sahip bir vektörle temsil edilir. <== onehot bunu saglıyor
history_lr = model_lr.fit(x_train, y_onehot_train, epochs=10, batch_size=128, validation_split=0.2, verbose=False)  #modelı egıtım verılerı x_train yanı  ve one-hot encoded etıketlerı yanı y_onehot_train ile egitir                     
                                   #kac kez verı uzerınden gececegını belırtır :epochs: daha fazla epochs daha ıyı ogrenme ama bbı noktada overflıttıng de olabılır!
                                   #her bır guncelleme ıcın kac ornek kullanılacak :batch_size: daha kucuk batch_size daha sık guncelleme yapar ama egıtım suresını uzatabılır
                                   #egıtım verısının bır kısmını dogrulama ıcın ayrılmasını saglar :validation_split:       
#print(history_lr.history)  bununla accuracy loss val accuracy val loss bunların degerlerı verılır                 


y_onehot_test = tf.one_hot(y_test, 10)
print(model_lr.evaluate(x_test, y_onehot_test)) #burda test verılerını onehot ile dounusturmeden once calıstırdım valueerro verdı    

print(model_lrpredict(x_test[:5]))

                                                        
