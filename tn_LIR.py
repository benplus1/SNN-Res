import numpy as np
import tensorflow as tf
import os
import pickle as pk
import sys
import itertools


class neurons:
    def __init__(self, classs, trial):
        self.classs = classs

        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.dataPath = self.rootPath + '/dataFiles/Trial_' + str(trial)

        self.codePath = self.rootPath + '/codeFiles'
        self.check_create_save_dir(self.dataPath)

        self.v_rest_e = -65.
        self.v_reset_e = -65.
        self.v_thresh_e = -55.
        self.refrac_e = 5.
        self.equil_e = -57.

        self.v_rest_i = -60.
        self.v_reset_i = -60.
        self.v_thresh_i = -53.
        self.refrac_i = 2.
        self.equil_i = -53.

        self.init_ge = 0.1

        self.init_wee = 0.1
        self.init_wei = 0.1

        self.v_time = 100.
        self.g_time = 30
        self.tr_pre = 20.

        self.th_pre = 15.

        self.lrate = 0.01
        self.exp_lrate = 0.001

        self.wmax = 1.0

        self.xtar = 0.

        self.theta_init = 0.01

        self.spikes_amount = 5

        self.sess = None

    def assign_sess(self, sess):
        self.sess = sess

    def check_create_save_dir(self, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print('SAVE DIRECTORY ALREADY EXISTS, TERMINATE PROGRAM IMMEDIATELY TO PREVENT LOSS OF EXISTING DATA!')

    def unpack_file(self, filename):

        names = []
        data = []

        f = open(filename, 'rb')
        # filename_orig = pk.load(f)

        read = True
        while read:
            dat_temp = pk.load(f)
            if dat_temp == 'end':
                read = False
            else:
                print(isinstance(dat_temp, str))
                if isinstance(dat_temp, str):
                    names.append(dat_temp)
                    data.append(pk.load(f))
                    # print(data)
        f.close()
        # names.append(filename_orig)

        return data, names

    def block_diagonal(self, matrices, dtype=tf.float32):
        """
        Constructs block-diagonal matrices from a list of batched 2D tensors.

        Args:
          matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
            matrices with the same batch dimension).
          dtype: Data type to use. The Tensors in `matrices` must match this dtype.
        Returns:
          A matrix with the input matrices stacked along its main diagonal, having
          shape [..., \sum_i N_i, \sum_i M_i].

        """
        matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
        blocked_rows = tf.Dimension(0)
        blocked_cols = tf.Dimension(0)
        batch_shape = tf.TensorShape(None)
        for matrix in matrices:
            full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
            batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
            blocked_rows += full_matrix_shape[-2]
            blocked_cols += full_matrix_shape[-1]
        ret_columns_list = []
        for matrix in matrices:
            matrix_shape = tf.shape(matrix)
            ret_columns_list.append(matrix_shape[-1])
        ret_columns = tf.add_n(ret_columns_list)
        row_blocks = []
        current_column = 0
        for matrix in matrices:
            matrix_shape = tf.shape(matrix)
            row_before_length = current_column
            current_column += matrix_shape[-1]
            row_after_length = ret_columns - current_column
            row_blocks.append(tf.pad(
                tensor=matrix,
                paddings=tf.concat(
                    [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                     [(row_before_length, row_after_length)]],
                    axis=0)))
        blocked = tf.concat(row_blocks, -2)
        blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
        return blocked

    def initialize_LIR_neurons(self, num_neurons_list):

        int_num_neurons = sum(num_neurons_list)

        v = tf.Variable(tf.ones(int_num_neurons) * self.v_rest_e, dtype=tf.float32, expected_shape=[int_num_neurons, 1],
                        name='v', use_resource=True)

        # Weights and conductances
        tensor_weight_list = []
        tensor_ge_list = []

        w_in_to_e = tf.Variable(tf.ones([num_neurons_list[1], num_neurons_list[0]]) * self.init_wee, dtype=tf.float32)
        g_input_to_e = tf.Variable(tf.ones([num_neurons_list[1], num_neurons_list[0]]) * self.init_ge, dtype=tf.float32)

        tensor_weight_list.append(w_in_to_e)
        tensor_ge_list.append(g_input_to_e)

        w_e_to_i = tf.linalg.tensor_diag([self.init_wei] * num_neurons_list[1])
        g_e_to_i = tf.linalg.tensor_diag([self.init_ge] * num_neurons_list[1])

        tensor_weight_list.append(w_e_to_i)
        tensor_ge_list.append(g_e_to_i)

        non_variable_W = self.block_diagonal(matrices=tensor_weight_list)
        non_variable_g = self.block_diagonal(matrices=tensor_ge_list)

        w_padded_one = tf.pad(non_variable_W, tf.constant([[num_neurons_list[0], 0], [0, 0]]), "CONSTANT")
        g_padded_one = tf.pad(non_variable_g, tf.constant([[num_neurons_list[0], 0], [0, 0]]), "CONSTANT")

        casted_w_p_one = tf.cast(w_padded_one, tf.float32)
        casted_g_p_one = tf.cast(g_padded_one, tf.float32)

        w_i_to_e = tf.Variable(tf.ones([num_neurons_list[1], num_neurons_list[2]]) * self.init_wei, dtype=tf.float32)
        g_i_to_e = tf.Variable(tf.ones([num_neurons_list[1], num_neurons_list[2]]) * self.init_ge, dtype=tf.float32)

        wie_diag = tf.linalg.set_diag(w_i_to_e, [0] * num_neurons_list[1])
        gie_diag = tf.linalg.set_diag(g_i_to_e, [0] * num_neurons_list[1])

        w_padded_two = tf.pad(wie_diag, tf.constant([[num_neurons_list[0], num_neurons_list[2]], [0, 0]]), "CONSTANT")
        g_padded_two = tf.pad(gie_diag, tf.constant([[num_neurons_list[0], num_neurons_list[2]], [0, 0]]), "CONSTANT")

        casted_w_p_two = tf.cast(w_padded_two, tf.float32)
        casted_g_p_two = tf.cast(g_padded_two, tf.float32)

        w_concatted = tf.concat([casted_w_p_one, casted_w_p_two], 1)
        g_concatted = tf.concat([casted_g_p_one, casted_g_p_two], 1)

        W = tf.Variable(tf.zeros((int_num_neurons, int_num_neurons)), dtype=tf.float32,
                        expected_shape=[int_num_neurons, int_num_neurons], name='W', use_resource=True)
        W_up = tf.assign(W, w_concatted)

        g = tf.Variable(tf.zeros((int_num_neurons, int_num_neurons)),
                        dtype=tf.float32, expected_shape=[int_num_neurons, int_num_neurons], name='g', use_resource=True)
        g_up = tf.assign(g, g_concatted)

        # Traces
        xpres = tf.Variable(tf.zeros((int_num_neurons, int_num_neurons)), dtype=tf.float32,
                            expected_shape=[int_num_neurons, int_num_neurons], name='xpres', use_resource=True)

        # Create spikes variable
        fired = tf.Variable(np.zeros(int_num_neurons), dtype=tf.float32, expected_shape=[int_num_neurons, 1],
                            name='fired', use_resource=True)

        # Create cum_sum variable
        cum_sum = tf.Variable(np.zeros((self.spikes_amount, num_neurons_list[0])),
                              dtype=tf.float32, expected_shape=[self.spikes_amount, num_neurons_list[0]], use_resource=True)

        theta = tf.Variable(tf.ones(num_neurons_list[1]) * self.theta_init, dtype=tf.float32,
                            expected_shape=[int_num_neurons, 1], name='theta', use_resource=True)

        output_spikes = tf.Variable(np.zeros((10, num_neurons_list[1])), dtype=tf.float32,
                                    expected_shape=[10, num_neurons_list[1]], name="output_spikes", use_resource=True)

        sum_curr_spikes = tf.Variable(np.zeros(num_neurons_list[1]), dtype=tf.float32,
                                      expected_shape=[num_neurons_list[1], 1], name='sum_curr_spikes', use_resource=True)

        classes = tf.Variable(np.zeros(num_neurons_list[1]), dtype=tf.float32,
                              expected_shape=[num_neurons_list[1], 1], name='classes', use_resource=True)

        prediction = tf.Variable(0., dtype=tf.float32, expected_shape=[], name='prediction', use_resource=True)

        return v, W, xpres, g, fired, cum_sum, theta, output_spikes, W_up, g_up, classes, sum_curr_spikes, prediction

    def update_v_for_LIR_neurons(self, v, g, tensor_len):

        const = tf.constant(self.equil_e, shape=[tensor_len, 1])

        temp_v = tf.reshape(v, shape=[tensor_len, 1])

        that_tensor = tf.add(const, tf.negative(temp_v))

        ge = tf.matrix_band_part(g, -1, 0)
        gi = tf.matrix_band_part(g, 0, -1)

        this_tensor = tf.matmul(ge, that_tensor)

        inhib_subtract = tf.add(tf.constant(self.equil_i, shape=[tensor_len, 1]), tf.negative(temp_v))

        those_tensor = tf.matmul(gi, inhib_subtract)

        other_const = tf.constant(self.v_rest_e, shape=[tensor_len, 1])

        subtraction = tf.add(other_const, tf.negative(temp_v))

        dv = tf.multiply(tf.add(tf.add(subtraction, this_tensor), those_tensor),
                         tf.constant(1./self.v_time, shape=[tensor_len, 1]))

        v_up = tf.assign(v, tf.add(v, tf.reshape(dv, [-1])))

        return v_up

    def check_spikes_train(self, v_update, fired, theta, num_neurons_list):
        input_neurons = tf.slice(v_update, [0], [num_neurons_list[0]])
        excitatory = tf.slice(v_update, [num_neurons_list[0]], [num_neurons_list[1]])
        inhibatory = tf.slice(v_update, [num_neurons_list[0] + num_neurons_list[1]], [num_neurons_list[2]])

        # Check if spiked, and clip to form tensors with 1s and 0s
        checked_input = tf.subtract(input_neurons, self.v_rest_e)
        checked_excitatory = tf.subtract(excitatory, tf.add(theta, self.v_thresh_e))
        checked_inhibatory = tf.subtract(inhibatory, self.v_thresh_i)
        concatted = tf.concat([checked_input, checked_excitatory, checked_inhibatory], 0)

        spikes = tf.ceil(tf.clip_by_value(concatted, 0.0, 1.0))

        # Count num of spikes
        # num_spikes = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.equal(fired, spikes)), tf.float32))

        # Create the new fired Variable
        fired_update = tf.assign(fired, spikes)

        return fired_update

    def check_spikes_test(self, v_update, fired, num_neurons_list):
        input_neurons = tf.slice(v_update, [0], [num_neurons_list[0]])
        excitatory = tf.slice(v_update, [num_neurons_list[0]], [num_neurons_list[1]])
        inhibatory = tf.slice(v_update, [num_neurons_list[0] + num_neurons_list[1]], [num_neurons_list[2]])

        checked_input = tf.subtract(input_neurons, self.v_rest_e)
        checked_excitatory = tf.subtract(excitatory, self.v_thresh_e)
        checked_inhibatory = tf.subtract(inhibatory, self.v_thresh_i)
        concatted = tf.concat([checked_input, checked_excitatory, checked_inhibatory], 0)

        spikes = tf.ceil(tf.clip_by_value(concatted, 0.0, 1.0))

        # Count num of spikes
        # num_spikes = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.equal(fired, spikes)), tf.float32))

        # Create the new fired Variable
        fired_update = tf.assign(fired, spikes)

        return fired_update

    # Updates traces and g_e based on presynaptic spikes
    def update_g(self, w, g, spikes, tensor_len):
        # Make spikes 1d array into 2d array, where we tile the spikes array column by column (we care about presynp)
        re_spikes = tf.reshape(spikes, shape=[tensor_len, 1])

        transpose_spikes = tf.transpose(re_spikes)

        tiled_spikes = tf.tile(transpose_spikes, [tensor_len, 1])

        where = tf.equal(tiled_spikes, tf.ones_like(tiled_spikes))

        # need to adjust for different g_time
        new_g = tf.assign(g, tf.where(where, tf.add(w, g), tf.add(g, tf.divide(tf.negative(g), self.g_time))))

        return new_g

    def update_trace(self, xpres, spikes, tensor_len):
        re_spikes = tf.reshape(spikes, shape=[tensor_len, 1])

        transpose_spikes = tf.transpose(re_spikes)

        tiled_spikes = tf.tile(transpose_spikes, [tensor_len, 1])

        where = tf.equal(tiled_spikes, tf.ones_like(tiled_spikes))

        new_xpres = tf.assign(xpres, tf.where(where, tf.add(xpres, 1.),
                                              tf.add(xpres, tf.divide(tf.negative(xpres), self.tr_pre))))

        return new_xpres

    # Updates weights based on postsynpatic spikes
    def update_weights_train(self, w, xpres, spikes, tensor_len):
        multiply = tf.constant([tensor_len])

        tiled_spikes = tf.transpose(tf.reshape(tf.tile(spikes, multiply), [multiply[0], tf.shape(spikes)[0]]))

        where = tf.equal(tiled_spikes, tf.ones_like(tiled_spikes))

        shp = [tensor_len, tensor_len]

        # calculate x_tar
        re_spikes = tf.reshape(spikes, shape=[tensor_len, 1])

        transpose_spikes = tf.transpose(re_spikes)

        tiled_spikes = tf.tile(transpose_spikes, [tensor_len, 1])

        where_tar = tf.equal(tiled_spikes, tf.ones_like(tiled_spikes))

        xtarget = tf.where(where_tar, tf.add(xpres, 1.), tf.add(xpres, tf.divide(tf.negative(xpres), self.tr_pre)))

        this_w = tf.where(
                where, tf.add(w, tf.multiply(tf.multiply(tf.constant(self.lrate, shape=shp),
                                                         tf.subtract(xpres, xtarget)),
                                             tf.pow(tf.subtract(tf.constant(self.wmax, shape=shp), w),
                                                    tf.constant(self.exp_lrate, shape=shp)))), w)

        new_w = tf.assign(w, this_w)

        return new_w

    def reset_neurons_voltage(self, spikes, v, tensor_len):

        # Create the boolean equivalent of fired_update to be used when resetting spiked values
        where = tf.equal(spikes, tf.ones_like(spikes))

        reset_v = tf.assign(v, tf.where(where, tf.constant(self.v_reset_e, shape=[tensor_len]), v))

        return reset_v

    def calculate_theta_train(self, spikes, num_neurons_list, theta):
        ex_spikes = tf.slice(spikes, [num_neurons_list[0]], [num_neurons_list[1]])

        where_theta = tf.equal(ex_spikes, tf.ones_like(ex_spikes))

        new_theta = tf.assign(theta, tf.where(where_theta, tf.add(theta, 1.),
                                              tf.add(theta, tf.divide(tf.negative(theta), self.th_pre))))

        return new_theta

    def set_training_voltage(self, currSpikes, v, tensor_len):
        where = tf.equal(currSpikes, tf.ones_like(currSpikes))

        trained_v = tf.assign(v, tf.where(where, tf.constant(self.v_thresh_e, shape=[tensor_len]), v))

        return trained_v

    def add_to_sum_curr_spikes(self, spikes, sum_curr_spikes, num_neurons_list):
        sliced = tf.slice(spikes, [num_neurons_list[0]], [num_neurons_list[1]])
        new_sum = tf.add(sliced, sum_curr_spikes)

        assign_sum_curr_spikes = tf.assign(sum_curr_spikes, new_sum)

        return assign_sum_curr_spikes

    def reset_sum_curr_spikes(self, sum_curr_spikes, num_neurons_list):
        reset_output = tf.assign(sum_curr_spikes, tf.constant(0., shape=[num_neurons_list[1]]))

        return reset_output

    def finalize_output_spikes(self, output_spikes, sum_curr_spikes, batch_y):
        shp1 = output_spikes.get_shape()
        unpacked_tensor = tf.unstack(output_spikes, axis=0)
        new_tensor_list = []

        for iiR in list(range(shp1[0])):
            new_tensor_list.append(tf.where(tf.equal(batch_y, iiR), sum_curr_spikes, unpacked_tensor[iiR]))

        new_tensor = tf.stack(new_tensor_list, axis=0)

        new_output = tf.assign(output_spikes, new_tensor)

        return new_output

    def reset_output_spikes(self, output_spikes, num_neurons_list):
        reset_output = tf.assign(output_spikes, tf.constant(0., shape=[10, num_neurons_list[1]]))

        return reset_output

    def assign_class(self, output_spikes, classes):
        transposed = tf.transpose(output_spikes)
        thisthing = tf.argmax(transposed, axis=1)
        assigned = tf.assign(classes, tf.cast(thisthing, tf.float32))

        return assigned

    def reset_class(self, classes):
        reset_output = tf.assign(classes, tf.constant(0., shape=[25]))

        return reset_output

    def gen_all_spikes(self, cum_sum, curr_flat_freqs):
        random = tf.random.poisson(curr_flat_freqs, [self.spikes_amount])
        new_cum_sum = tf.cumsum(random, axis=0)
        new_new_cum_sum = tf.squeeze(new_cum_sum)
        updated_cs = tf.assign(cum_sum, new_new_cum_sum)

        return updated_cs

    def get_prediction(self, classes, sum_spikes, prediction):
        running_value = tf.constant(0.)
        the_prediction = tf.constant(0.)

        for i in range(10):
            fl = float(i)
            where = tf.equal(tf.constant([fl] * 25), classes)
            values = tf.where(where, sum_spikes, tf.constant([0.] * 25))
            value = tf.reduce_sum(values)
            condition = tf.greater(value, running_value)
            the_prediction = tf.cond(condition, lambda: tf.constant(fl), lambda: the_prediction)
            running_value = tf.cond(condition, lambda: value, lambda: running_value)

        new_predict = tf.assign(prediction, the_prediction)

        return new_predict

    def reset_prediction(self, prediction):
        rest = tf.assign(prediction, tf.constant(0.))

        return rest

    def get_spikes_at_time_step(self, cum_sum, time_step, fired, length):
        minus_t = tf.subtract(cum_sum, time_step)
        absolute = tf.abs(minus_t)
        clipped = tf.clip_by_value(absolute, 0, 1)
        summ = tf.reduce_sum(clipped, axis=0)
        subtracted = tf.subtract(summ, self.spikes_amount)
        final = tf.abs(subtracted)
        concat_this = tf.constant(0., shape=[length])
        concatted = tf.concat([final, concat_this], axis=0)
        fired_updated = tf.assign(fired, concatted)

        return fired_updated

    def reset_theta(self, theta, num_neurons_list):
        reset_theta = tf.ones(num_neurons_list[1]) * self.theta_init
        new_thing = tf.assign(theta, reset_theta)

        return new_thing
