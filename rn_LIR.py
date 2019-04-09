import numpy as np
import tensorflow as tf
import os
import sys
import pickle as pk
import tn_LIR as tn
import time
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


class run_neurons:
    def __init__(self, trial):
        self.trial = trial

        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.dataPath = self.rootPath + '/dataFiles/Trial_' + str(trial)

        self.codePath = self.rootPath + '/codeFiles'
        self.check_create_save_dir(self.dataPath)

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.fire_rate = 4.
        self.max_intensity = 255.

    def check_create_save_dir(self, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print('SAVE DIRECTORY ALREADY EXISTS, TERMINATE PROGRAM IMMEDIATELY TO PREVENT LOSS OF EXISTING DATA!!!!')

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

    def generate_neuron_ranges(self, num_neurons_list):
        begin_range = 0
        end_range = num_neurons_list[0]

        layer_ranges = []
        layer_number = len(num_neurons_list)

        for index, i in enumerate(num_neurons_list):
            layer_ranges.append(range(begin_range, end_range))
            begin_range = begin_range + i
            if (index + i) <= layer_number:
                end_range = num_neurons_list[index + 1]

        return layer_ranges

    def gen_input_spike_array(self, neurons_spiking_at_curr_time, tensor_len, input_len):
        spike_array = []
        past_neuron = 0
        if not neurons_spiking_at_curr_time:
            return [0.0] * tensor_len
        else:
            for neuron in neurons_spiking_at_curr_time:
                neuron_difference = neuron - past_neuron - 1
                temp = [0.0] * neuron_difference
                spike_array.extend(temp)
                spike_array.append(1.0)
                past_neuron = neuron
            final_diff = input_len - neurons_spiking_at_curr_time[-1]
            spike_array.extend([0.0] * final_diff)
            b = [0.0] * (tensor_len - input_len)
            spike_array.extend(b)
            return spike_array

    def run_neurons(self, per_train_iter, per_test_iter):

        neuron = tn.neurons('nnnn', 0)

        num_neurons_list = [784, 25, 25]
        tensor_len = sum(num_neurons_list)
        concat_train = tensor_len-num_neurons_list[0]

        v, w, xpres, g, fired, cum_sum, theta, \
            output_spikes, W_up, g_up, classes, \
            sum_curr_spikes, prediction = neuron.initialize_LIR_neurons(num_neurons_list)

        curr_flat_freqs = tf.placeholder(tf.float32, shape=(784, 1))
        curr_time_step = tf.placeholder(tf.float32, shape=(5, 784))
        batch_y = tf.placeholder(tf.float32, shape=())

        accuracy = []

        # Generate all spikes
        spikes_array = neuron.gen_all_spikes(cum_sum, curr_flat_freqs)

        # Get spikes at time step
        spikes_curr_time = neuron.get_spikes_at_time_step(spikes_array, curr_time_step, fired, concat_train)

        # Updated V based on spikes
        trained_v = neuron.set_training_voltage(spikes_curr_time, v, tensor_len)

        # 1 iteration per time step
        v_up = neuron.update_v_for_LIR_neurons(v, g, tensor_len)

        # check for spikes
        fired_up_t = neuron.check_spikes_train(v, fired, theta, num_neurons_list)

        fired_up_te = neuron.check_spikes_test(v, fired, num_neurons_list)

        # presynaps, update pretraces and g_e
        new_g = neuron.update_g(w, g, fired, tensor_len)
        new_xpres = neuron.update_trace(xpres, fired, tensor_len)

        # update weight matrix for train
        new_w_t = neuron.update_weights_train(w, xpres, fired, tensor_len)

        # reset v if passed threshold
        reset_v = neuron.reset_neurons_voltage(fired, v, tensor_len)

        new_theta_t = neuron.calculate_theta_train(fired, num_neurons_list, theta)

        reset_theta = neuron.reset_theta(theta, num_neurons_list)

        # predict
        get_predict = neuron.get_prediction(classes, sum_curr_spikes, prediction)

        rest_pred = neuron.reset_prediction(prediction)

        # add to output spikes
        add_spikes = neuron.add_to_sum_curr_spikes(fired, sum_curr_spikes, num_neurons_list)

        reset_add_spikes = neuron.reset_sum_curr_spikes(sum_curr_spikes, num_neurons_list)

        finalize_os = neuron.finalize_output_spikes(output_spikes, sum_curr_spikes, batch_y)

        reset_os = neuron.reset_output_spikes(output_spikes, num_neurons_list)

        assign_cls = neuron.assign_class(output_spikes, classes)

        reset_cls = neuron.reset_class(classes)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        neuron.assign_sess(sess)
        sess.run(W_up)
        sess.run(g_up)

        for j in range(int(60000/per_train_iter)):

            for i in range(per_train_iter):
                tt_tt = time.time()
                batch_xs, batch_ys = self.mnist.train.next_batch(1)
                feed_flat_freqs = (self.max_intensity * batch_xs / self.fire_rate).flatten()
                n_feed_ff = np.reshape(feed_flat_freqs, [num_neurons_list[0], 1])
                sess.run(spikes_array, feed_dict={curr_flat_freqs: n_feed_ff})

                for curr_time in range(350):
                    a_time = curr_time+1
                    time_arr = [a_time, a_time, a_time, a_time, a_time] * num_neurons_list[0]
                    shaped_time = np.reshape(time_arr, [5, num_neurons_list[0]])

                    sess.run(spikes_curr_time, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})

                    sess.run(trained_v, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})

                    sess.run(v_up)

                    sess.run(fired_up_t)

                    sess.run(new_g)

                    sess.run(new_w_t)

                    sess.run(new_xpres)

                    sess.run(reset_v)

                    sess.run(new_theta_t)

                for curr_time in range(150):
                    sess.run(v_up)
                    sess.run(fired_up_t)
                    sess.run(new_g)
                    sess.run(new_w_t)
                    sess.run(new_xpres)
                    sess.run(reset_v)
                    sess.run(new_theta_t)

                print("Done" + str(i) + " " + str(time.time() - tt_tt))

                # plt.eventplot(voltage_array)
                # plt.show()
                # plt.plot(range(500), voltage_array)
                # plt.show()

            sess.run(reset_theta)

            saver = tf.train.Saver()

            save_path = saver.save(sess, "/tmp6/model_new.ckpt")

            print("Model saved in path: %s" % save_path)

            # validate and assign labels
            validate = [False] * 10
            while True:
                batch_xs, batch_ys = self.mnist.validation.next_batch(1)

                curr = np.nonzero(batch_ys[0])[0][0]

                if not validate[curr]:
                    validate[curr] = True

                    feed_flat_freqs = (self.max_intensity * batch_xs / self.fire_rate).flatten()
                    n_feed_ff = np.reshape(feed_flat_freqs, [num_neurons_list[0], 1])
                    sess.run(spikes_array, feed_dict={curr_flat_freqs: n_feed_ff})

                    for curr_time in range(350):
                        a_time = curr_time+1
                        time_arr = [a_time, a_time, a_time, a_time, a_time] * num_neurons_list[0]
                        shaped_time = np.reshape(time_arr, [5, num_neurons_list[0]])
                        sess.run(spikes_curr_time, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})

                        sess.run(trained_v, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})
                        # print("Trained_v " + str(time.time() - curr_time))
                        sess.run(v_up)
                        # print("v_up " + str(time.time() - curr_time))
                        sess.run(fired_up_te)
                        # print("fired_up " + str(time.time() - curr_time))
                        # sess.run(num_spikes)
                        sess.run(new_g)
                        # print("new_g " + str(time.time() - curr_time))
                        sess.run(new_xpres)
                        # print("new_xpres " + str(time.time() - curr_time))
                        # sess.run(new_w)
                        # print("new_w " + str(time.time() - curr_time))
                        sess.run(reset_v)
                        # print("full time for " + str(curr_time) + " " + str(time.time() - t_tt))
                        # sess.run(new_theta)

                        # add_spikes
                        sess.run(add_spikes)

                    for curr_time in range(150):
                        sess.run(v_up)
                        sess.run(fired_up_te)
                        sess.run(new_g)
                        sess.run(new_xpres)
                        sess.run(reset_v)

                        # add_spikes
                        sess.run(add_spikes)

                    # substitute output_spikes row
                    sess.run(finalize_os, feed_dict={batch_y: curr})

                    # reset spikes added
                    sess.run(reset_add_spikes)

                    print('validated')

                if sum(validate) == 10:
                    # assign classes
                    sess.run(assign_cls, feed_dict={batch_y: curr})

                    # reset output_spikes
                    sess.run(reset_os)

                    break

            batch_success = 0.
            # Test 20-30 images
            for k in range(per_test_iter):
                tt_tt = time.time()
                batch_xs, batch_ys = self.mnist.test.next_batch(1)
                feed_flat_freqs = (self.max_intensity * batch_xs / self.fire_rate).flatten()
                n_feed_ff = np.reshape(feed_flat_freqs, [num_neurons_list[0], 1])
                sess.run(spikes_array, feed_dict={curr_flat_freqs: n_feed_ff})

                for curr_time in range(350):
                    a_time = curr_time+1
                    time_arr = [a_time, a_time, a_time, a_time, a_time] * num_neurons_list[0]
                    shaped_time = np.reshape(time_arr, [5, num_neurons_list[0]])
                    sess.run(spikes_curr_time, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})

                    sess.run(trained_v, feed_dict={curr_flat_freqs: n_feed_ff, curr_time_step: shaped_time})
                    # print("Trained_v " + str(time.time() - curr_time))
                    sess.run(v_up)
                    # print("v_up " + str(time.time() - curr_time))
                    sess.run(fired_up_te)
                    # print("fired_up " + str(time.time() - curr_time))
                    # sess.run(num_spikes)
                    sess.run(new_g)
                    # print("new_g " + str(time.time() - curr_time))
                    sess.run(new_xpres)
                    # print("new_xpres " + str(time.time() - curr_time))
                    # print("new_w " + str(time.time() - curr_time))
                    sess.run(reset_v)
                    # print("full time for " + str(curr_time) + " " + str(time.time() - t_tt))

                    # add_spikes
                    sess.run(add_spikes)

                for curr_time in range(150):
                    sess.run(v_up)
                    sess.run(fired_up_te)
                    sess.run(new_g)
                    sess.run(new_xpres)
                    sess.run(reset_v)

                    # add_spikes
                    sess.run(add_spikes)

                # gauge accuracy with classes and number of spikes found
                sess.run(get_predict)

                if sess.run(prediction) == curr:
                    batch_success = batch_success + 1.

                # reset_spikes
                sess.run(reset_add_spikes)

                sess.run(rest_pred)

                print("Done Test" + str(k) + " " + str(time.time() - tt_tt))

            accuracy.append(batch_success/per_test_iter)

            print(accuracy)

            # reset pregenerated classes
            sess.run(reset_cls)


trial1 = run_neurons(0)
trial1.run_neurons(1, 1)
