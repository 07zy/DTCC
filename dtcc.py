# -*-coding:utf-8-*-

import tensorflow as tf
import utils
import math
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
import copy
import drnn
import drnn
import rnn_cell_extensions
import util.augmentation as aug
from tsnes import tsne,middle_tsne
import warnings
warnings.filterwarnings("ignore", category=Warning)

class Config(object):
    """Train config."""
    batch_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 4, 16]  # 我们设置将层数和每一层的膨胀数为3，1，4，16，编码器每层的单位数为ｍ１，ｍ２，ｍ３
    num_steps = None
    embedding_size = None
    learning_rate = 5e-3
    cell_type = 'GRU'  # decode
    lamda = 1
    class_num = None
    denosing = True  # False
    sample_loss = True  # False

    hidden_norm = False
    temperature = 1.0


def cluster_loss(out, out_aug,class_num,hidden_norm,temperature):#(batch_size,cluster_num)

    out=tf.nn.softmax(out,dim=1)#
    out_aug=tf.nn.softmax(out_aug,dim=1)


    p_i = tf.reduce_sum(out, axis=0)  # .view(-1)
    p_i = tf.reshape(p_i, [-1])
    p_i = p_i / tf.reduce_sum(p_i)
    ne_i = math.log(float(p_i.shape.as_list()[0])) + tf.reduce_sum((p_i * tf.log(p_i)))

    p_j = tf.reduce_sum(out_aug, axis=0)  # .view(-1)
    p_j = tf.reshape(p_j, [-1])
    p_j = p_j / tf.reduce_sum(p_j)
    ne_j = math.log(float(p_j.shape.as_list()[0])) + tf.reduce_sum((p_j * tf.log(p_j)))

    ne_loss = ne_i + ne_j

    out = tf.transpose(out)
    out_aug = tf.transpose(out_aug)


    LARGE_NUM = 1e9
    if hidden_norm:
        out = tf.nn.l2_normalize(out, -1)  # t(28,400)-----outaug(28,400)
        out_aug = tf.nn.l2_normalize(out_aug, -1)
    labels = tf.one_hot(tf.range(class_num), class_num * 2)  # [batch_size,2*batch_size]
    masks = tf.one_hot(tf.range(class_num), class_num) # [batch_size,batch_size]

    logits_aa =tf.keras.losses.cosine_similarity(tf.expand_dims(out,1),tf.expand_dims(out,0),axis=2) /temperature  # (28,28)[batch_size,batch_size]
    logits_aa = logits_aa - masks * LARGE_NUM  # remove the same samples in out

    logits_bb = tf.keras.losses.cosine_similarity(tf.expand_dims(out_aug,1), tf.expand_dims(out_aug,0),axis=2) /temperature  # [batch_size,batch_size]
    logits_bb = logits_bb - masks * LARGE_NUM  # remove the same samples in out_aughiddena=tf.reshape(hidden_a,[self.batch_size,-1])

    logits_ab = tf.keras.losses.cosine_similarity(tf.expand_dims(out,1), tf.expand_dims(out_aug,0),axis=2) /temperature  # (28,28)logits_aa=tf.clip_by_value(logits_aa, 1e-8,tf.reduce_max(logits_aa))
    logits_ba = tf.keras.losses.cosine_similarity(tf.expand_dims(out_aug,1), tf.expand_dims(out,0),axis=2) /temperature
    loss_a = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], axis=1))  # labels(28,56)----logits_aa(28,28)
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], axis=1))

    loss = loss_a + loss_b+ne_loss

    return loss







def contrastive_loss(out, out_aug, batch_size, hidden_norm, temperature):
    LARGE_NUM = 1e9
    if hidden_norm:
        out = tf.nn.l2_normalize(out, -1)  # t(28,400)-----outaug(28,400)
        out_aug = tf.nn.l2_normalize(out_aug, -1)

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)  # [batch_size,2*batch_size]
    masks = tf.one_hot(tf.range(batch_size), batch_size)  # [batch_size,batch_size]
    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature  # (28,28)[batch_size,batch_size]
    logits_aa = logits_aa - masks * LARGE_NUM  # remove the same samples in out
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature  # [batch_size,batch_size]
    logits_bb = logits_bb - masks * LARGE_NUM  # remove the same samples in out_aughiddena=tf.reshape(hidden_a,[self.batch_size,-1])

    logits_ab = tf.matmul(out, out_aug,
                          transpose_b=True) / temperature  # (28,28)logits_aa=tf.clip_by_value(logits_aa, 1e-8,tf.reduce_max(logits_aa))
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature
    loss_a = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], axis=1))  # labels(28,56)----logits_aa(28,28)
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], axis=1))
    loss = loss_a + loss_b

    return loss


class RNN_clustering_model(object):

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.dilations = config.dilations
        self.num_steps = config.num_steps
        self.embedding_size = config.embedding_size

        self.cell_type = config.cell_type
        self.lamda = config.lamda
        self.class_num = config.class_num  # numbers of clusters
        self.denosing = config.denosing
        self.sample_loss = config.sample_loss
        self.K = config.class_num  # numbers of clusters

        self.temperature = config.temperature
        self.hidden_norm = config.hidden_norm

    def build_model(self):
        input = tf.placeholder(tf.float32, [None, self.num_steps],name='inputs')  # 这表示该维度待定，你的输入是什么长度，它就是多少。一般来说第一个维度是指 batch_size
        # 定义如下:[2,3], [None, 3]等。其中 [None, 3]表示列是3，行数不定。
        noise = tf.placeholder(tf.float32, [None, self.num_steps], name='noise')

        input_a = tf.placeholder(tf.float32, [None, self.num_steps], name='input_a')
        # input_b = tf.placeholder(tf.float32, [None, self.num_steps], name='input_b')

        F_new_value = tf.placeholder(tf.float32, [None, self.K], name='F_new_value')  # k=2

        F = tf.get_variable('F', shape=[self.batch_size, self.K],
                            initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32),
                            trainable=False)

        F_aug_new = tf.placeholder(tf.float32, [None, self.K], name='F_aug_new')  # k=2

        F_aug = tf.get_variable('F_aug', shape=[self.batch_size, self.K],
                            initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32),
                            trainable=False)

        # inputs has shape (batch_size, n_steps, embedding_size)
        inputs = tf.reshape(input, [-1, self.num_steps, self.embedding_size])
        noises = tf.reshape(noise, [-1, self.num_steps, self.embedding_size])

        inputs_a = tf.reshape(input_a, [-1, self.num_steps, self.embedding_size])

        # a list of 'n_steps' tenosrs, each has shape (batch_size, embedding_size)
        # encoder_inputs = utils._rnn_reformat(x = inputs, input_dims = self.embedding_size, n_steps = self.num_steps)

        # noise_input has shape (batch_size, n_steps, embedding_size)
        if self.denosing:
            print('Noise')
            noise_input = inputs + noises  # (?,286,1)

            noise_a = inputs_a + noises
        else:
            print('Non_noise')
            noise_input = inputs

            noise_a = inputs_a

        reverse_noise_input = tf.reverse(noise_input, axis=[1])  # (?,286,1)
        reverse_noise_a = tf.reverse(noise_a, axis=[1])

        decoder_inputs = utils._rnn_reformat(x=noise_input, input_dims=self.embedding_size,
                                             n_steps=self.num_steps)  # a list of 'n_steps' tenosrs,each has shape (batch_size, input_dims)list:286----tensor(28,1)
        targets = utils._rnn_reformat(x=inputs, input_dims=self.embedding_size,
                                      n_steps=self.num_steps)  # embedding_size=1,num_steps = train_data.shape[1]

        recon_inputs = utils._rnn_reformat(x=noise_a, input_dims=self.embedding_size,
                                             n_steps=self.num_steps)  # a list of 'n_steps' tenosrs,each has shape (batch_size, input_dims)list:286----tensor(28,1)
        target_a= utils._rnn_reformat(x=inputs_a, input_dims=self.embedding_size,
                                      n_steps=self.num_steps)  # embedding_size=1,num_steps = train_data.shape[1]

        # targets------list:286

        if self.cell_type == 'LSTM':
            raise ValueError('LSTMs have not support yet!')

        elif self.cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(np.sum(self.hidden_size) * 2)

        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.embedding_size)

        lf = None
        if self.sample_loss:
            print('Sample Loss')

            def lf(prev, i):
                return prev

        # encoder_output has shape 'layer' list of tensor [batch_size, n_steps, hidden_size]
        with tf.variable_scope('fw'):  # 返回一个用于定义创建variable（层）的op的上下文管理器。
            _, encoder_output_fw = drnn.drnn_layer_final(noise_input, self.hidden_size, self.dilations, self.num_steps,
                                                         self.embedding_size, self.cell_type)  # 输出drnn结构
        with tf.variable_scope('bw'):
            _, encoder_output_bw = drnn.drnn_layer_final(reverse_noise_input, self.hidden_size, self.dilations,
                                                         self.num_steps, self.embedding_size, self.cell_type)

        with tf.variable_scope('fa'):  # 返回一个用于定义创建variable（层）的op的上下文管理器。
            _, encoder_output_fa = drnn.drnn_layer_final(noise_a, self.hidden_size, self.dilations, self.num_steps,
                                                         self.embedding_size, self.cell_type)

        with tf.variable_scope('ba'):
            _, encoder_output_ba = drnn.drnn_layer_final(reverse_noise_a, self.hidden_size, self.dilations,
                                                         self.num_steps, self.embedding_size, self.cell_type)


        if self.cell_type == 'LSTM':
            raise ValueError('LSTMs have not support yet!')
        elif self.cell_type == 'GRU':
            fw = []
            bw = []

            fa = []
            ba = []

            for i in range(len(self.hidden_size)):
                fw.append(encoder_output_fw[i][:, -1, :])  # encoder_output_fw[i][:, -1, :]---(28,100),(28,50),(28,50),,,encoder_output_fw[i]----shape=(?, 286, 100),(28,286,50),(28,286,50)
                bw.append(encoder_output_bw[i][:, -1, :])

                fa.append(encoder_output_fa[i][:, -1, :])
                ba.append(encoder_output_ba[i][:, -1, :])


            encoder_state_fw = tf.concat(fw, axis=1)  # shape=(?, 200)200=100+50+50
            encoder_state_bw = tf.concat(bw, axis=1)  # shape=(?, 200)

            encoder_state_fa = tf.concat(fa, axis=1)  # tensor(28,200)
            encoder_state_ba = tf.concat(ba, axis=1)  # tensor(28,200)



            encoder_state = tf.concat([encoder_state_fw, encoder_state_bw], axis=1)  # shape=(?, 400)
            encoder_state_a = tf.concat([encoder_state_fa, encoder_state_ba], axis=1)  # (28,400)

            decoder_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                                       initial_state=encoder_state, cell=cell,
                                                                       loop_function=lf)  # list(286)----tensor(28,1)
            decoder_aug,_=tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=recon_inputs,
                                                                       initial_state=encoder_state_a, cell=cell,
                                                                       loop_function=lf)

        if self.cell_type == 'LSTM':
            hidden_abstract = encoder_state.h
            hidden_a = encoder_state_a.h

        elif self.cell_type == 'GRU':
            hidden_abstract = encoder_state
            hidden_a = encoder_state_a  # tensor(28,286)


        # F_update
        F_update = tf.compat.v1.assign(F, F_new_value)
        real_hidden_abstract = hidden_abstract  # 原始数据作一个k-means损失
        # W has shape [sum(hidden_size)*2, batch_size]
        W = tf.transpose(real_hidden_abstract)  # (400,28)
        WTW = tf.matmul(real_hidden_abstract, W)  # (28,28)
        FTWTWF = tf.matmul(tf.matmul(tf.transpose(F), WTW), F)  # （2，2）

        #F_aug
        F_update_a = tf.compat.v1.assign(F_aug, F_aug_new)
        W_aug= tf.transpose(hidden_a)  # (400,28)
        W_augTW_aug = tf.matmul(hidden_a, W_aug)  # (28,28)
        F_augTW_augTW_augF_aug = tf.matmul(tf.matmul(tf.transpose(F_aug), W_augTW_aug), F_aug)  # （2，2）


        with tf.name_scope("loss_reconstruct"):
            loss_reconstruct = tf.compat.v1.losses.mean_squared_error(labels=targets , predictions=decoder_outputs)
            loss_recona=tf.compat.v1.losses.mean_squared_error(labels=target_a , predictions=decoder_aug)
        with tf.name_scope("k-means_loss"):
            loss_k_means = tf.linalg.trace(WTW) - tf.linalg.trace(FTWTWF)
            loss_kmeans_aug=tf.linalg.trace(W_augTW_aug) - tf.linalg.trace(F_augTW_augTW_augF_aug)
        with tf.name_scope("contrastive_loss"):
            hiddena = tf.reshape(hidden_a, [self.batch_size, -1])
            contrastive_l = contrastive_loss(hiddena, hidden_abstract, self.batch_size, self.hidden_norm,self.temperature)
            cluster_l=cluster_loss(F_update,F_update_a,self.class_num,self.hidden_norm,self.temperature)

        with tf.name_scope("loss_total"):
            loss=loss_reconstruct+loss_recona + self.lamda / 2 * (loss_k_means+loss_kmeans_aug)+ contrastive_l#\loss_kmeans_aug


        regularization_loss = 0.0
        for i in range(len(tf.trainable_variables())):
            regularization_loss += tf.nn.l2_loss(tf.trainable_variables()[i])
        loss = loss + 1e-4 * regularization_loss
        input_tensors = {
            'inputs': input,
            'input_a': input_a,
            'noise': noise,
            'F_new_value': F_new_value,
            'F_aug_new':F_aug_new,
        }
        loss_tensors = {
            'loss_reconstruct': loss_reconstruct,
            'loss_recona':loss_recona,
            'loss_k_means': loss_k_means,
            'loss_kmeans_aug':loss_kmeans_aug,
            'regularization_loss': regularization_loss,
            'contrastive_loss': contrastive_l,
            'cluster_loss':cluster_l,
            'loss': loss
        }

        return input_tensors, loss_tensors, real_hidden_abstract,hidden_a, F_update,F_update_a


def run_model(dataset,train_data,train_label, config):
    best_nmi=0
    best_ri=0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_config = tf.ConfigProto()  # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
    gpu_config.gpu_options.allow_growth = True  # 动态申请显存

    config.batch_size =int(train_data.shape[0])
    config.num_steps = train_data.shape[1]# 这里不包括label,故为286，而不是287
    config.embedding_size = 1

    train_label, num_classes = utils.transfer_labels(train_label) # num_classes=2(0 or 1)
    config.class_num = num_classes

    print('samples:',train_data.shape[0])
    print('batch_size',config.batch_size)
    print('time_length:', config.num_steps)
    print('Label:', np.unique(train_label))

    with tf.Session(config=gpu_config) as sess:
        model = RNN_clustering_model(config=config)
        input_tensors, loss_tensors, real_hidden_abstract,hidden_a, F_update,F_update_a = model.build_model()
        train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_tensors['loss']+loss_tensors['cluster_loss'])#+loss_tensors['cluster_loss']  # 使用优化器，并且更新op
        sess.run(tf.global_variables_initializer())
        print('========================================================================================================')
        saver = tf.train.Saver(max_to_keep=10)
        Epoch = 200
        for i in range(Epoch):
            # shuffle data and label
            loss_val,recon_l, recona_l,kmeans,kmeans_aug,clu,con=0,0,0,0,0,0,0

            indices = np.random.permutation(train_data.shape[0])  # 随机排序
            shuffle_data = train_data[indices]  # 28,286
            shuffle_label = train_label[indices]  # (28,)

            row = train_data.shape[0]
            batch_len = int(row / config.batch_size)
            left_row = row - batch_len * config.batch_size# 定位到某个序列的起始位置

            if left_row != 0:
                need_more = config.batch_size - left_row
                rand_idx = np.random.choice(np.arange(batch_len * config.batch_size), size=need_more)
                shuffle_data = np.concatenate((shuffle_data, shuffle_data[rand_idx]), axis=0)
                shuffle_label = np.concatenate((shuffle_label, shuffle_label[rand_idx]), axis=0)
            assert(shuffle_data.shape[0] % config.batch_size == 0)

            noise_data = np.random.normal(loc=0, scale=0.1, size=[shuffle_data.shape[0], shuffle_data.shape[1]])
            total_abstract = []
            total_a=[]
            print('----------Epoch %d----------' % i)
            k = 0

            for input, _ in utils.next_batch(config.batch_size, shuffle_data):#contrastive_l, loss_tensors['contrastive_loss'],
                noise = noise_data[k * config.batch_size: (k + 1) * config.batch_size, :]  # ndarray(28,286)
                input_a = np.reshape(input, [-1, config.num_steps, config.embedding_size])  # 28,286,1  dtype=float64
                input_a = aug.jitter(input_a, sigma=0.03)  # 28,286,1
                input_aa = np.reshape(input_a, [-1, config.num_steps])#k_means,loss_tensors['loss_k_means']
                val,recon,recona,contrastive_l,k_means,kmeans_a,abstract,abstract_a,_ = sess.run(
                    [loss_tensors['loss'],loss_tensors['loss_reconstruct'], loss_tensors['loss_recona'],loss_tensors['contrastive_loss'],loss_tensors['loss_k_means'],loss_tensors['loss_kmeans_aug'], real_hidden_abstract,hidden_a,train_op],
                    feed_dict={input_tensors['inputs']: input,
                               input_tensors['input_a']: input_aa,
                               input_tensors['noise']: noise,
                               })
                total_abstract.append(abstract)
                total_a.append(abstract_a)

                loss_val+=val
                recon_l+=recon
                recona_l+=recona
                kmeans+=k_means
                kmeans_aug+=kmeans_a
                con+=contrastive_l
                k += 1

                if i % 10 == 0 and i != 0:
                    part_hidden_val = np.array(abstract).reshape(-1, np.sum(config.hidden_size) * 2)  # 20,400
                    W = part_hidden_val.T
                    U, sigma, VT = np.linalg.svd(W)
                    sorted_indices = np.argsort(sigma)
                    topk_evecs = VT[sorted_indices[:-num_classes - 1:-1], :]  # (2,20)
                    F_new = topk_evecs.T  # (20,2)

                    part = np.array(abstract_a).reshape(-1, np.sum(config.hidden_size) * 2)  # 20,400
                    W_aug = part.T
                    U, sig, V_augT = np.linalg.svd(W_aug)
                    sorted = np.argsort(sig)
                    topk = V_augT[sorted[:-num_classes - 1:-1], :]  # (2,20)
                    F_aug_new = topk.T  # (20,2)
                    print('F_aug_new:',F_aug_new)

                    cluster_l = sess.run([loss_tensors['cluster_loss']],
                                         feed_dict={input_tensors['F_aug_new']: F_aug_new,
                                                    input_tensors['F_new_value']: F_new})

                    print('clu_loss:', cluster_l)


            print('loss_val:{},recon:{},recona:{},kmeans:{},kmeans_aug:{},con:{}'.format(loss_val / k, recon_l / k,
                                                                                              recona_l / k, kmeans / k,
                                                                                              kmeans_aug / k, con / k))#con:{}, con / k



            # if i==0:
            #     # ml=[]
            #     # part_hidden_val = np.array(total_abstract).reshape(-1, np.sum(config.hidden_size) * 2)
            #     # t_l = np.array(shuffle_label)
            #     # for k in range(t_l.shape[0]):
            #     #     ml.append([t_l[k]])
            #     # ml = np.array(ml)
            #     # middle_tsne(dataset, part_hidden_val, ml)
            #     saver.save(sess, 'NMI_model/middle_results/{}/epoch{}'.format(dataset, i))
            #     saver.save(sess, 'RI_model/middle_results/{}/epoch{}'.format(dataset, i))
            #
            # if i==10 or i==30 or i==50:
            #     saver.save(sess, 'NMI_model/middle_results/{}/epoch{}'.format(dataset,i))
            #     saver.save(sess, 'RI_model/middle_results/{}/epoch{}'.format(dataset,i))
            #


            # # keshihua--origin
            # ml=[]
            # part_hidden_val = np.array(total_abstract).reshape(-1, np.sum(config.hidden_size) * 2)
            # t_l=np.array(shuffle_label)
            # for k in range(t_l.shape[0]):
            #     ml.append([t_l[k]])
            # ml=np.array(ml)



            test_hidden_val = np.array(total_abstract).reshape(-1, np.sum(config.hidden_size) * 2)  # test_total_abstract(28,400),test_hidden_val(28,400)
            km = KMeans(n_clusters=num_classes)
            km_idx = km.fit_predict(test_hidden_val)  # km_idx:ndarray(28,)
            ri, nmi, ari, acc = utils.evaluation(prediction=km_idx, label=shuffle_label)  # prediction:ndarray(28,)
            print('acc:', acc)
            print('nmi:', nmi)
            print('ari:', ari)
            print('ri:', ri)
            if nmi > best_nmi:
                best_nmi = nmi
                best_epoch = i
                # tsne(dataset,part_hidden_val,ml)
                # saver.save(sess, 'NMI_model/{}/'.format(dataset))# global_step=0000
            if ri>best_ri:
                best_ri=ri
                ri_epoch=i
                # saver.save(sess, 'RI_model/{}/'.format(dataset))
    return best_nmi,best_ri, best_epoch,ri_epoch



def main():
    config = Config()
    # input your filename
    dataset_name='Meat'


    '''dataset setting'''
    train_name ='./UCRArchive_2018/{0}/{0}_TRAIN.tsv'.format(dataset_name)  # re-config the path
    test_name ='./UCRArchive_2018/{0}/{0}_TEST.tsv'.format(dataset_name)  # re-config the path



    train_data, train_label = utils.load_data(train_name)
    test_data,test_label=utils.load_data(test_name)
    data=np.concatenate([train_data,test_data],axis=0)
    label=np.concatenate([train_label,test_label],axis=0)

    '''dataset setting'''
    # filename = './UCRArchive_2018/{0}/{0}_TRAIN.tsv'.format(dataset_name)  # re-config the path
    dilations = [1, 4, 16]
    config.dilations = dilations

    for encoder_config in [[100, 50, 50]]:  # [100,50,50], [50,30,30]
        config.hidden_size = encoder_config  # list:3[100,50,50]

        for lambda_1 in [1e-3]:  # 1, 1e-1, 1e-2, 1e-3print(sess.run(p_i))

            config.lamda = lambda_1
            best_nmi,best_ri, best_epoch,ri_epoch = run_model(dataset_name,data,label, config)

            # log
            log_file = os.path.join('Ablation/wo_kmeans_aug', '{}_log.txt'.format(dataset_name))
            if os.path.exists(log_file) == False:
                f = open(log_file, 'w')
                f.close()
            f = open(log_file, 'a')
            print('dataset: {}\t'.format(dataset_name), file=f)
            print('network config:\nencoder_hidden_units = {}, lambda = {},batch_size={}'.format(config.hidden_size, config.lamda,config.batch_size),
                  file=f)
            print('best\t{} = {},epoch = {}\nbest\t{} = {},epoch = {}\n\n'.format('NMI', best_nmi, best_epoch,'RI',best_ri,ri_epoch), file=f)

            f.close()


if __name__ == "__main__":
    main()
