import tensorflow as tf
import utils
import numpy as np

class SyncNetModel_MDMN(object):
    def __init__(self, options):
        self.l = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.sample_type = tf.float32
        self.num_domains = options['num_domains']
        self.num_labels = options['num_labels']
        self.sample_shape = options['sample_shape']
        self.batch_size = options['batch_size']
        self.source_num = options['source_num']
        self.target_num = options['target_num']
        self.t_idx = options['t_idx']
        self.dropout_rate = options['dropout_rate']
        self.X = tf.placeholder(tf.as_dtype(self.sample_type), [None] + list(self.sample_shape), name="input_X")
        self.domains = tf.placeholder(tf.float32, [None, self.num_domains], name="input_domains")
        self.domain_weights = tf.placeholder(tf.float32, [self.num_domains], name = 'domain_weights')
        self.weights = tf.placeholder(tf.float32, [None, self.num_domains], name = 'weights')
        
        self.y = tf.placeholder(tf.float32, [None, self.num_labels], name="input_labels")
        self.train = tf.placeholder(tf.bool, [], name = 'train')
        self._build_model(options)
        self._setup_train_ops()
    
    def get_mask(self, d):
        masks = np.zeros((d.shape[0], d.shape[1]), dtype = np.float32)
        for i in range(self.num_domains):
            sum_d = max(np.sum(d[:,i]),1)
            masks[:,i] = d[:,i]/sum_d
        return masks
    
    def get_fspecific_domain_weights(self, d, pred, target_idx): 
        masks = self.get_mask(d)
        fdomain_weights = np.zeros((self.num_domains,), dtype = np.float32)
        for i in range(self.num_domains):
            if pred is None:
                fdomain_weights[i] = 1.
            else:
                fdomain_weights[i] = np.sum((masks[:,i] - masks[:,target_idx]) * pred)
        #fdomain_weights[target_idx] = -1000
        #fdomain_weights = utils.softmax(fdomain_weights)
        #fdomain_weights[target_idx] = 0
        #fdomain_weights = fdomain_weights/np.linalg.norm(fdomain_weights,2)
        fdomain_weights = utils.sigmoid(fdomain_weights)
        fdomain_weights[target_idx] = 1
        return fdomain_weights.astype('float32')      
             
    def get_weights_soft(self, d, pred = None):   
        ones = np.ones((self.source_num + self.target_num,), dtype = np.float32)
        if pred is None:     
            domain_weights = np.ones((self.num_domains, self.num_domains), dtype = np.float32)
            for i in range(self.num_domains):
                domain_weights[:,i] = self.get_fspecific_domain_weights(d, pred, i)
            weights = np.zeros((self.source_num + self.target_num, self.num_domains), dtype = np.float32)
            for i in range(self.num_domains):
                temp = ones - d[:,i]
                weights[:,i] = d[:,i]/max(np.sum(d[:,i]), 1) - temp/max(np.sum(temp), 1)
            return -weights, domain_weights
        
        domain_weights = np.zeros((self.num_domains,self.num_domains), dtype = np.float32)
        for i in range(self.num_domains):
            domain_weights[:,i] = self.get_fspecific_domain_weights(d, pred[:,i], i)
        f_weights = np.zeros((self.num_domains,), dtype = np.float32)
        for i in range(self.num_domains):
            temp = np.repeat(np.reshape(domain_weights[:,i],(1,self.num_domains)), self.source_num + self.target_num, axis = 0)
            masks = self.get_mask(d)
            masks[:,i] = -masks[:,i]
            temp = np.sum(temp * masks, axis = 1)
            f_weights[i] = np.sum(temp * pred[:,i].reshape(-1))
        f_weights[self.t_idx] = -1000
        f_weights = utils.softmax(f_weights)
        #f_weights = utils.sigmoid(f_weights)
        f_weights[self.t_idx] = 1
        weights = np.zeros(d.shape, dtype = np.float32)
        for i in range(self.num_domains):
            domain_weights_repeat = np.repeat(np.reshape(domain_weights[:,i],(1,self.num_domains)), self.source_num + self.target_num, axis = 0)
            masks = self.get_mask(d)
            masks[:,i] = -masks[:,i]
            temp = d * masks * domain_weights_repeat
            weights[:,i] =  f_weights[i] * np.sum(temp, axis = 1)
        return weights.astype('float32'), domain_weights.astype('float32')
        
    
    def SyncNetFilters(self, options):
        b=tf.get_variable(name = 'b', shape = [1,1,options['C'],options['K']], initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05))
        omega=tf.get_variable(name = 'omega', shape = [1,1,1,options['K']], initializer =tf.random_uniform_initializer(minval = 0., maxval = 1.))
        zero_pad = tf.zeros((1, 1, 1, options['K']), dtype = tf.float32, name ='zero_pad')
        phi_ini=tf.get_variable(name = 'phi', shape = [1,1,options['C']-1, options['K']], initializer =tf.random_normal_initializer(mean=0.0, stddev=0.05))
        phi = tf.concat([zero_pad, phi_ini], axis = 2)
        beta=tf.get_variable(name = 'beta', shape = [1,1,1,options['K']], initializer =tf.random_uniform_initializer(minval = 0., maxval = 0.05))
        #t=np.reshape(np.linspace(-options['Nt']/2.,options['Nt']/2.,options['Nt']),[1,options['Nt'],1,1])
        t=np.reshape(range(-options['Nt']/2,options['Nt']/2),[1,options['Nt'],1,1])
        tc=tf.constant(np.single(t),name='t')
        W_osc=tf.multiply(b,tf.cos(tc*omega+phi))
        W_decay=tf.exp(-tf.pow(tc,2)*beta)
        W=tf.multiply(W_osc,W_decay)
        self.beta_op = tf.assign(beta, tf.clip_by_value(beta, 0, np.infty))
        return W

    def feature_extractor(self, X, options, reuse = False):
        self.dropout_x = utils.channel_dropout(X, self.dropout_rate)
        X = tf.expand_dims(self.dropout_x, axis = 1, name = 'reshaped_input')
        with tf.variable_scope('feature_extractor',reuse = reuse):
            W = self.SyncNetFilters(options)       
            bias = tf.get_variable(name = 'bias', shape = [options['K']], initializer = tf.constant_initializer(0.0))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME') + bias)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, options['pool_size'], 1], strides=[1, 1, options['pool_size'], 1], padding='SAME')            
            h_pool1 = tf.reshape(h_pool1, [-1, options['cl_Wy']])   
        with tf.variable_scope('feature_extractor_fc1', reuse = reuse):
            features = tf.nn.relu(utils.fully_connected_layer(h_pool1, 100))     
        return features
            
    def label_predictor(self, features):
        #self.classify_domains= tf.cond(self.train, lambda: tf.slice(self.domains, [0, 0], [self.source_num, -1]), lambda: self.domains)    
        #weights = tf.reduce_sum(self.domain_weights * self.classify_domains, axis = 1)
        with tf.variable_scope('label_predictor_logits'):
            logits = utils.fully_connected_layer(features, self.num_labels)    
        self.y_pred = tf.nn.softmax(logits)
        self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y))
        self.y_acc = utils.predictor_accuracy(self.y_pred,self.y)
    
    def domain_predictor(self, features, reuse = False):       
        with tf.variable_scope('domain_predictor_fc1', reuse = reuse):
            d_h_fc1 = tf.nn.relu(utils.fully_connected_layer(features, 100))
        with tf.variable_scope('domain_predictor_logits', reuse = reuse):
            d_logits = utils.fully_connected_layer(d_h_fc1, self.num_domains)   
        return d_logits
      

    def adversary_loss(self, options):   
        shuffled = self.feature_extractor(tf.random_shuffle(self.X), options, reuse = True)
        epsilon = tf.random_uniform([], 0.0, 1.0)
        interpolated = epsilon * self.features + (1 - epsilon) * shuffled
        fs_interpolated = tf.boolean_mask(self.domain_predictor(interpolated, reuse = True), tf.cast(self.domains, tf.bool))
        penalty_coefficient = 10 #* self.num_domains

        self.adversary_gradient = tf.norm(tf.gradients(fs_interpolated, interpolated)[0], axis=1)
        self.adversary_penalty = tf.reduce_mean(tf.square(self.adversary_gradient - 1.0) * penalty_coefficient)                           
        self.wass_approx = tf.reduce_mean(self.weights * self.d_pred)
        
    def _build_model(self, options):     
        self.features = self.feature_extractor(self.X, options, reuse = False)   
        self.features_for_prediction = tf.cond(self.train,  lambda: tf.slice(self.features, [0, 0], [self.source_num, -1]), lambda: self.features)              
        self.label_predictor(self.features_for_prediction)        
        self.d_pred = self.domain_predictor(self.features, reuse = False)
        self.adversary_loss(options)
        self.d_loss = self.adversary_penalty - self.wass_approx
        self.total_loss = self.y_loss + self.l*self.wass_approx
        
    def _setup_train_ops(self):
        label_vars = utils.vars_from_scopes(['label_predictor', 'feature_extractor'])
        domain_vars = utils.vars_from_scopes(['domain_predictor'])
        self.train_feature_ops = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, var_list = label_vars)
        self.train_adversary_ops = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list = domain_vars)
        