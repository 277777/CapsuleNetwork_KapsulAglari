"""
Deep Learning Türkiye Topluluğu için Merve Ayyüce Kızrak tarafından hazırlanmıştır. (http://www.ayyucekizrak.com/)

Bazı temel katmanlar bir Kapsül Ağ oluşturmak için kullanılır. Kapsül ağ modeli (CapsNet) oluşturmak için kullanılan katmanlar 
farklı veri setleri üzerinde de kullanılabilir, sadece MNIST seti için tasarlanmamıştır.
*NOT*: Bazı fonksiyonlar birden fazla şekilde uygulanabilir. Bunları kendiniz test edebilirsiniz ve yorum olarak ekleyebilirsiniz.

"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Vektör uzunluklarının hesaplanır. Bu, hata değerindeki (margin_loss) y_true ile aynı boyutta Tensor hesaplamak için kullanılır.
    Bu katmanı kullanarak modelin çıkışı direkt olarak etiketleri kestirebilir. ( `y_pred = np.argmax(model.predict(x), 1)` ) kullanarak.
    girişler    : shape=[None, num_vectors, dim_vector]
    çıkış       : shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    shape=[None, num_capsule, dim_vector]  Bu Tensor maske ya maksimum uzunluğuyla kapsül ya da ek bir giriş maskesidir.
    
    Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8,her bir iterasyonda "8" resim alınsın 
                                                   her örnek 3 kapsül içersin  dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # Doğru etiketler. 8 örnek, 3 sınıf, (one-hot coding).
        out = Mask()(x)  # out.shape=[8, 6]
        # ya da
        out2 = Mask()([x, y])  # out2.shape=[8,6]. y'nin doğru etiketleri ile maskelenir. Tabi ki y manipüle edilebilir.
        `
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # doğru etiket shape = [batch_size, n_classes], ile sağlanı.  (örneğin: one-hot code.)
            assert len(inputs) == 2
            inputs, mask = inputs
            mask = K.expand_dims(mask, -1)
        else:  # eğer doğru etiket yoksa, kapsüller maksimum uzunluklarıyla maskelenir. Temel olrak kestirim için kullanılır.
            # kapsül uzunluğu hesaplanır.
            x = K.sqrt(K.sum(K.square(inputs), -1, True))
            # x aralığını max(new_x[i,:])=1 ve diğerleri << 0 yapmak için büyütür. 
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            # x'teki bu maksimum değer 1 yapılır diğerleri 0 yapılır.
            # the max value in x clipped to 1 and other to 0. Böylece `maske` bir one-hot coding olur.
            mask = K.clip(x, 0, 1)

        return K.batch_flatten(inputs * mask)  # maskelenmiş girişler, shape = [None, num_capsule * dim_capsule]

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # deoğru değerler sağlanır
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # doğru olmayan değerler sağlanır
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    Kapsülde lineer olmayan aktivasyon kullanılır. Böylece büyük vektörün uzunluğu 1'e küçük vektör 0'a yaklaşır.
    :param vectors: bazı vektörler ezilir (squashed), N-dim tensor
    :param axis: eksen ezilir (squash)
    :return: giriş vektörleri ile aynı uzunluklu bir Tensor
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    Kapsül Katmanı. Dense katmanıyla benzerdir. Dense katmanı `in_num` girişlere sahiptir. Her biri skalardır. önceki katmandan 
    gelen nöron çıkıştır. Çıkış nöronları `out_num` ile gösterilir. Kapsül katmanı (CapsuleLayer) çıkış nöronlarının genişletilmiş
    skalar bir vektör halidir.
    Giriş: shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. Dense katman için, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: her katmandaki kapsül sayısı 
    :param dim_capsule: ilgili katmandaki kapsülün çıkış vektörünün boyutu 
    :param num_routing: yönlendirme (routing) algoritmasının iterasyon sayısı
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "Input Tensorunun olması gereken boyutu shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Matris Dönüştürme
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # W değerleriyle çarpmaya hazırlamak için num_capsule boyutunu çoğaltır  
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        """
        # Begin: routing algorithm V1, dynamic ------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        b = K.zeros(shape=[self.batch_size, self.num_capsule, self.input_num_capsule])
        def body(i, b, outputs):
            c = tf.nn.softmax(b, dim=1)  # dim=2 is the num_capsule dimension
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i != 1:
                b = b + K.batch_dot(outputs, inputs_hat, [2, 3])
            return [i-1, b, outputs]
        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), b, K.sum(inputs_hat, 2, keepdims=False)]
        shape_invariants = [tf.TensorShape([]),
                            tf.TensorShape([None, self.num_capsule, self.input_num_capsule]),
                            tf.TensorShape([None, self.num_capsule, self.dim_capsule])]
        _, _, outputs = tf.while_loop(cond, body, loop_vars, shape_invariants)
        # End: routing algorithm V1, dynamic ------------------------------------------------------------#
        """
        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # In forward pass, `inputs_hat_stopped` = `inputs_hat`;
        # In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
        inputs_hat_stopped = K.stop_gradient(inputs_hat)
        
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule]. It's equivalent to
        # `b=K.zeros(shape=[batch_size, num_capsule, input_num_capsule])`. I just can't get `batch_size`
        b = K.stop_gradient(K.sum(K.zeros_like(inputs_hat), -1))

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.num_routing - 1:
                # c.shape =  [batch_size, num_capsule, input_num_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
                # outputs.shape=[None, num_capsule, dim_capsule]
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]
            else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))

                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


"""
# The following is another way to implement primary capsule layer. This is much slower.
# Apply Conv2D `n_channels` times and concatenate all capsules
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_capsule])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""
