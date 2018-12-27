import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from SyncNetModel_MDMN import SyncNetModel_MDMN
import utils
import signalloader as dataloader
import tensorflow.contrib.slim as slim
import scipy.io
import os 
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 


batch_size = 8
num_steps = 2000
valid_steps = 400


test_name = 'SEED'
test_method = 'MDMN'
best_params = None
options = {}
options['batch_size'] = batch_size
options['G_iter'] = 1
options['D_iter'] = 1
options['l'] = 0.01
options['K'] = 15
options['Nt'] = 40
options['pool_size'] = 40
options['dropout_rate'] = 0.2
save_path = './' + test_name + '_' + test_method + '_' + str(options['l']) + '_Diter' + str(options['D_iter']) +  '_K' + str(options['K']) +'/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
record = []
for test_idx in range(15):
    best_valid = -1
    best_test = -1
    best_acc = -1
    train,valid,test = dataloader.load_SEED_dataset('/home/yl353/SEAD/Result_zscore_200/', test_idx)   
    options['sample_shape'] = (test['images'].shape[1],test['images'].shape[2])
    options['num_domains'] = len(train.keys()) + 1 #len(sources) + len(targets)
    options['num_labels'] = test['labels'].shape[1]
    options['source_num'] = batch_size * ( options['num_domains'] - 1)
    options['target_num'] = 32 #* ( options['num_domains'] - 1)      
    options['C'] = test['images'].shape[2]
    options['T'] = test['images'].shape[1]   
    options['t_idx'] = np.argmax(test['domains'][0])   
    options['cl_Wy'] = int(np.ceil(float(options['T'])/float(options['pool_size'])) * options['K'])
    description = test_name + '_' + test_method + '_test_' + str(test_idx) + '_' + str(options['l'])
           
    tf.reset_default_graph() 
    graph = tf.get_default_graph()
    
    model = SyncNetModel_MDMN(options)
    sess =  tf.Session(graph = graph, config=tf.ConfigProto(gpu_options=gpu_options)) 
    tf.global_variables_initializer().run(session = sess)
    
    gen_source_batches = []
    for key in train.keys():
        gen_source_batches.append(utils.batch_generator([train[key]['images'], train[key]['labels'], train[key]['domains']], options['batch_size']))
    gen_target_batch = utils.batch_generator([test['images'], test['labels'], test['domains']], options['target_num'])
    
    d_pred = None
    print('Training...')
    for i in range(1, num_steps + 1):          
        p = float(i) / num_steps
        l = options['l'] * (2. / (1. + np.exp(-10. * p)) - 1)
        lr = 0.0002 / (1. + 10 * p)**0.75
        #lr = 0.002
        X0 = []
        y0 = []
        d0 = []
        for j in range(len(train.keys())):
            x_temp, y_temp, d_temp = gen_source_batches[j].next()
            X0.append(x_temp)
            y0.append(y_temp)
            d0.append(d_temp)
        X0 = np.concatenate(X0, axis = 0)
        y0 = np.concatenate(y0, axis = 0)
        d0 = np.concatenate(d0, axis = 0)
        X1, y1, d1 = gen_target_batch.next()
        X = np.concatenate([X0, X1], axis = 0)
        d = np.concatenate([d0, d1], axis = 0)
    
        # Update Feature Extractor & Lable Predictor
        for j in range(options['G_iter']):
            weights, domain_weights = model.get_weights_soft(d,d_pred)
            _, batch_loss, d_pred, tploss, tdloss, tp_acc, tpenalty= \
                sess.run([model.train_feature_ops, model.total_loss, model.d_pred, model.y_loss, model.d_loss, model.y_acc, model.adversary_penalty],
                         feed_dict={model.X: X, model.y: y0, model.domains: d, model.domain_weights:domain_weights[:, options['t_idx']], model.weights: weights, model.l: l, model.lr: lr, model.train: True})
            _ = sess.run(model.beta_op, feed_dict = {})
            
        for j in range(options['D_iter']):
            weights, domain_weights = model.get_weights_soft(d,d_pred)
            _, d_pred, dloss, dpenalty = \
                sess.run([model.train_adversary_ops, model.d_pred, model.d_loss, model.adversary_penalty],
                         feed_dict={model.X:X, model.domains: d, model.domain_weights:domain_weights[:,options['t_idx']], model.weights: weights, model.l: l, model.lr: lr, model.train: True})
            _ = sess.run(model.beta_op, feed_dict = {})
        
        if i % 200 == 0:
            print '%s iter %d  loss: %f  d_loss: %f  p_acc: %f  p: %f  l: %f  lr: %f' % \
                    (description, i, batch_loss, dloss, tp_acc, p, l, lr)
            
         
        if i % valid_steps == 0:
            train_pred = []
            labtrain = []
            for key in train.keys():
                train_pred.append(utils.get_data_pred(sess, model, 'y', train[key]['images'], train[key]['labels']))
                labtrain.append(train[key]['labels'])
            train_pred = np.concatenate(train_pred,axis = 0)
            labtrain = np.concatenate(labtrain, axis = 0)
            train_acc = utils.get_acc(train_pred, labtrain)
            
            valid_pred = []
            labvalid = []
            for key in valid.keys():
                valid_pred.append(utils.get_data_pred(sess, model, 'y', valid[key]['images'], valid[key]['labels']))
                labvalid.append(valid[key]['labels'])
            valid_pred = np.concatenate(valid_pred,axis = 0)
            labvalid = np.concatenate(labvalid,axis = 0)
            valid_acc = utils.get_acc(valid_pred, labvalid)
    
            test_pred = utils.get_data_pred(sess, model, 'y', test['images'], test['labels'])
            test_acc = utils.get_acc(test_pred, test['labels'])
           
                     
            print 'train: %.4f  valid: %.4f  test: %.4f ' % \
                    (train_acc, valid_acc, test_acc)
            if test_acc > best_test:
                best_test = test_acc
            if valid_acc > best_valid:
                best_params = utils.get_params(sess)
                best_valid = valid_acc
                best_acc = test_acc   
                confusion = metrics.confusion_matrix(np.argmax(test['labels'],axis = 1),np.argmax(test_pred,axis = 1))                  
    print('Best performance for ' + str(test_idx) + ' is ' + str(best_acc))
    n_test = 20
    test_samples = []
    test_labels = []
    test_domains = []
    for key in train.keys():
        shuffle = np.random.permutation(train[key]['domains'].shape[0])    
        test_samples.append(train[key]['images'][shuffle[:n_test]])
        test_labels.append(train[key]['labels'][shuffle[:n_test]])   
        test_domains.append( utils.to_one_hot(0 * np.ones((n_test,)),2))   
    for key in test.keys():   
        n_test = 100
        shuffle = np.random.permutation(test['labels'].shape[0])
        test_samples.append(test['images'][shuffle[:n_test]])
        test_labels.append(test['labels'][shuffle[:n_test]])   
        test_domains.append( utils.to_one_hot(1 * np.ones((n_test,)), 2))    
    
    test_samples = np.concatenate(test_samples, axis = 0)
    test_labels = np.concatenate(test_labels, axis = 0)
    test_domains = np.concatenate(test_domains, axis = 0)
    test_emb = sess.run(model.features, feed_dict={model.X: test_samples, model.train: False})
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(test_emb)
    names  = []
    names.append('sources')
    names.append('targets')  
    scipy.io.savemat(save_path +  'result_' + '_' + str(test_idx) + '_' + str(best_acc) + '_' + str(best_test) + '.mat',
                         { 'confusion':confusion})#, 'dann_tsne':dann_tsne, 'test_labels':test_labels, 'test_domains':test_domains}) # 'd_pred': test_dpred * labd,
#    utils.plot_embedding(dann_tsne, test_labels.argmax(1), test_domains.argmax(1), names, 'Domain Adaptation on EEG data (DANN)')