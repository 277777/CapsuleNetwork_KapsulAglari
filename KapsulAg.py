
Deep Learning Türkiye Topluluğu için Merve Ayyüce Kızrak tarafından hazırlanmıştır.
       ...
Hinton’ın “Dynamic Routing Between Capsules” makalesindeki CapsNet algoritmasının Keras Uygulamasıdır.
Geçerli sürümü TensorFlow’da hazırlanmıştır. Keras sürümünden farklı sürümlere kolaylıkla dönüştürülüp yeniden yazılabilir.
Amaç: Kapsül ağının el yazısı rakamları tanımaktaki performansını değerlendirmek.
Veriseti: MNIST (http://yann.lecun.com/exdb/mnist/)
Algoritma: Kapsül Ağları (Capsule Networks-CapsNet)
Microsoft Azure Notebook: https://notebooks.azure.com/deeplearningturkiye/libraries/pratik-derin-ogrenme/html/kapsul_rakam_tanima_CapsNet-Keras_MNIST.py
Kullanım:
       python CapsNet.py
       python CapsNet.py --epochs 50
       python CapsNet.py --epochs 50 --num_routing 3
       ... ...  
Sonuç:
    Validasyon Başarımı > 20 epoch sonrasında %99.5. 
     50 epoch sonrasında %99.66’ya yükselmektedir.
    Her epoch işlemi tek bir GTX1070 GPU kart üzerinde 110 saniye sürmektedir.
    """
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')
MNIST veriseti için Kapsül Ağları.
    :param input_shape: veri şekli, 3d, [genişlik, yükseklik, kanal]
    :param n_class: sınıf sayısı
    :param num_routing: dinamik yönlendirme (dynamic routing) iterasyon sayısı
    :return: iki Keras model, birincisi eğitim için, ikincisi değerlendirme için (evalaution).
            `eval_model` aynı zamanda eğitim için de kullanılabilir.
    """
    x = layers.Input(shape=input_shape)

    # Katman 1: Geleneksel Evrişimli Sinir Ağı Katmanı (Conv2D)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Katman 2: Conv2D katmanı ile ezme (squash) aktivasyonu, [None, num_capsule, dim_capsule]’e yeniden şekil veriyoruz. 
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Katman 3: Kapsül Katmanı. Dinamik Yönlendirme algoritması burada çalışıyor.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    # Katman 4: Her kapsülün uzunluğunu yeniden düzenleyen yardımcı bir katmandır. Doğru etiketle eşleşmesi için bu işlem yapılır.
    # Eğer Tensorflow kullanıyorsanız bu işleme gerek yoktur. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Kodçözücü Ağ.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # Doğru etiket, kapsül katmanın çıkışını maskelemek için kullanılır (Eğitim için).
    masked = Mask()(digitcaps)  # Maske, kapsülün maksimal uzunluğu ile kullanılır (Kestirim için). 

    # Eğitim ve Kestirimde Kodçözücü modelin paylaşımı
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Eğitim ve Değerlendirme (Kestirim) için Modeller
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Denklem(4) için hata değeri. y_true[i, :] sadece `1` içermediğinde, bu kayıp çok uzun çalışmalıdır. 
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: skaler kayıp değeri.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
Kapsül Ağının Eğitilmesi    
    :param model: CapsNet (Kapsül Ağ) Modeli 
    :param data: eğitim ve test verisinden bir grup içerir, örneğin; `((x_train, y_train), (x_test, y_test))`
    :param args: bağımsız değişkenler
    :return: eğitilmiş model
    """
    # verilerin kullanıma hazır hale gelmesi
    (x_train, y_train), (x_test, y_test) = data

    # geri çağırmalar
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # Modelin derlenmesi
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Veri Büyütme yapmadan modelin eğitimi:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Başlangıç: Veri Büyütme yaparak Modelin eğitimi ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # MNIST veri setini 2 piksel yukarı kaydırır
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Veri Büyütme yaparak Modelin Eğitimi. Eğer shift_fraction=0., Bu durumda da veri büyütme olmaz.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # Son: Veri Büyütme Yaparak Modelin Eğitilmesi -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()


def load_mnist():
    # Veri, eğitim ve test setleri arasında bölünüp karıştırılır
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    # Hiperparametrelerin ayarlanması
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, makalede SE hesaplanmıştır, burada MSE hesaplanıyor
    parser.add_argument('--num_routing', default=3, type=int)  # yönlendirme sayısı > 0 olmalı
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 TensorFlowBoard’ta ağırlıklar tutulur.
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Veri Yüklenir
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Model Tanımlanır
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    # Eğitim yada Test
    if args.weights is not None:  # Model ağırlıkları verilir
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # Ağırlıkları verildiği sürece test işlemi yapılır.
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test))

