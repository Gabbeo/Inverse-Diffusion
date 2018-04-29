import tensorflow as tf
import numpy as np
import time, datetime, os
from PIL import Image
from GaussianFilter import Gauss

# Suppress TensorFlow messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

setup_start_time = time.clock()

# Run specifications.
iterations = 10000
save_plots = False
save_graph_information = False

# Load the image using Pillow (ImageMagick) and convert it to a NumPy array.
cell_number = "250"
image = Image.open("image_" + cell_number + ".tiff").convert('L', [0, 1, 0, 0])
D_obs_array = np.array(image).astype(np.float32)

# Sigma values that have been optimized for this specific implementatation.
sigmas = [2.3, 5, 9, 13, 23, 33, 43, 53, 67]
K = len(sigmas) - 1

# Creates the inverse diffusion Gaussian matrices.
gauss_filters_horizontal = Gauss.weighted_filters_1d(sigmas, "x")
gauss_filters_vertical = Gauss.weighted_filters_1d(sigmas, "y")

##################################################################
'''
    In the following steps we will construct the TensorFlow graph that we will later on feed to a TensorFlow session.
    Before we feed it to a session it will not do any kind of calculations and we are just building the computation
    structure.
'''

with tf.name_scope('image') as scope1:
    # Convert the image array to a tensor.
    input_image = tf.convert_to_tensor(D_obs_array)
    reshaped_image = tf.reshape(input_image, [image.size[0], image.size[1], 1])

    # Stack the image.
    D_obs = tf.stack([reshaped_image])

    image_run = [D_obs]

with tf.name_scope('inverse') as scope2:
    with tf.name_scope('variables') as scope2_1:
        # Get the dimensions of the diffused image.
        diffused_image_shape = D_obs.get_shape()
        M = diffused_image_shape[1]
        N = diffused_image_shape[2]

        # Create all variables used by the algorithm.
        A = tf.Variable(tf.zeros([1, M, N, K], dtype=tf.float32), dtype=tf.float32)
        A_old = tf.Variable(tf.zeros([1, M, N, K], dtype=tf.float32), dtype=tf.float32)
        B = tf.Variable(tf.zeros([1, M, N, K], dtype=tf.float32))
        D = tf.Variable(tf.zeros(diffused_image_shape, dtype=tf.float32), dtype=tf.float32)
        eta = tf.constant(1 / (max(sigmas) - sigmas[0]), dtype=tf.float32)
        Ones = tf.ones([1, M, N, K], dtype=tf.float32)
        P = tf.Variable(tf.zeros([1, M, N, K], dtype=tf.float32), dtype=tf.float32)
        tf_lambda = tf.constant(0.5, dtype=tf.float32)
        W = tf.Variable(tf.ones(diffused_image_shape, dtype=tf.float32), dtype=tf.float32)

        with tf.name_scope('filters') as scope2_1_1:
            # The TensorFlow constant that holds the Gaussian filter.
            tf_gauss_filters_horizontal = []
            for i in range(K):
                tf_gauss_filters_horizontal.append(tf.constant(gauss_filters_horizontal[i], dtype=tf.float32))

            tf_gauss_filters_vertical = []
            for i in range(K):
                tf_gauss_filters_vertical.append(tf.constant(gauss_filters_vertical[i], dtype=tf.float32))

        variables_run = [D, A, P, W, eta, tf_lambda, Ones, B]

        with tf.control_dependencies(variables_run):
            with tf.name_scope('convolution') as scope2_2:
                # Changes the data structure of B to NCHW for more efficient computation on the GPU.
                B_NCHW = tf.transpose(B, [0, 3, 1, 2])
                B_convolution_layers = []

                for k in range(K):
                    B_horizontal_conv = tf.nn.conv2d(tf.expand_dims(B_NCHW[:, k, :, :], 1),
                                                     tf_gauss_filters_horizontal[k],
                                                     strides=[1, 1, 1, 1],
                                                     padding="SAME",
                                                     data_format='NCHW')
                    B_convolution_layers.append(tf.transpose(tf.nn.conv2d(B_horizontal_conv,
                                                                          tf_gauss_filters_vertical[k],
                                                                          strides=[1, 1, 1, 1],
                                                                          padding="SAME",
                                                                          data_format="NCHW"), [0, 2, 3, 1]))

                B_convolution_concatenated = tf.concat(B_convolution_layers, 3)
                B_convolution_summed = tf.reduce_sum(B_convolution_concatenated, 3)
                B_convolution_extended = tf.expand_dims(B_convolution_summed, 3)

                convolution_run = [B_convolution_concatenated, B_convolution_summed, B_convolution_extended]

            with tf.name_scope('difference') as scope2_3:
                # A factor 2 is included to counteract the factor 0.5 built into the tf.nn.l2_loss function.
                loss = tf.nn.l2_loss(tf.multiply(W, D_obs - B_convolution_extended)) * 2
                with tf.name_scope('gradient') as scope2_3_1:
                    gradient_step = tf.gradients(loss, B)[0]
                A_assign = tf.assign(A, tf.clip_by_value(tf.subtract(B, 1 / 2 * eta * gradient_step), 0.0, np.infty))

                difference_run = [loss, gradient_step, A_assign]

        with tf.control_dependencies(convolution_run + difference_run):
            with tf.name_scope('regularizer') as scope2_4:
                A_squared = tf.square(A)
                A_sum = tf.reduce_sum(A_squared, 3)
                A_sqrt = tf.sqrt(A_sum)
                A_invert = tf.reciprocal(A_sqrt)
                A_factor = tf.multiply((1 / 2) * eta * tf_lambda, A_invert)
                A_reshaped = tf.reshape(A_factor, [int(1), int(M), int(N), int(1)])
                P_assign = tf.assign(P, tf.clip_by_value(Ones - A_reshaped, 0, np.infty))

                regularizer_run = [A_squared, A_sum, A_sqrt, A_invert, A_factor, A_reshaped, P_assign]

        with tf.control_dependencies(regularizer_run):
            A_P_multiply = tf.assign(A, tf.multiply(P, A))
            multiply_run = [A_P_multiply]

    with tf.name_scope('speedup') as scope2_5:
        with tf.name_scope('t_and_alpha') as scope2_5_1:
            t = tf.Variable(1, dtype=tf.float32)
            t_old = tf.Variable(1, dtype=tf.float32)
            alpha = tf.Variable(0, dtype=tf.float32)

            # Calculates t and alpha.
            t = tf.assign(t, 1 / 2 + tf.sqrt(1 / 4 + tf.square(t_old)))
            with tf.control_dependencies([t]):
                alpha = tf.assign(alpha, tf.subtract(t_old, 1) / t)
                with tf.control_dependencies([alpha]):
                    t_old = tf.assign(t_old, t)

            t_and_alpha_run = [alpha, t_old]

        with tf.control_dependencies(t_and_alpha_run + multiply_run):
            alpha_prod = tf.multiply(alpha, tf.subtract(A, A_old))
            B_add = tf.add(A, alpha_prod)
            new_B = tf.assign(B, B_add)

            # Updates the variable A_old for the next run.
            A_to_old_A = tf.assign(A_old, A)

        # List of steps to run for the above part.
        speedup_run = [t_and_alpha_run, new_B, A_to_old_A]

    with tf.name_scope('statistics') as scope2_6:
        # Operation for retrieving the image
        output_img_op = tf.sqrt(tf.reduce_sum(tf.square(A), 3))

        # Operation for retrieving the normalized prediction's square error NSE and the group sparsity regularizer GS.
        # NSE
        NSE_NCHW = tf.transpose(A, [0, 3, 1, 2])
        NSE_convolution_layers = []

        for k in range(K):
            NSE_horizontal_conv = tf.nn.conv2d(tf.expand_dims(NSE_NCHW[:, k, :, :], 1),
                                               tf_gauss_filters_horizontal[k],
                                               strides=[1, 1, 1, 1],
                                               padding="SAME",
                                               data_format='NCHW')
            NSE_convolution_layers.append(tf.transpose(tf.nn.conv2d(NSE_horizontal_conv,
                                                                    tf_gauss_filters_vertical[k],
                                                                    strides=[1, 1, 1, 1],
                                                                    padding="SAME",
                                                                    data_format="NCHW"), [0, 2, 3, 1]))

        NSE_convolution_concatenated = tf.concat(NSE_convolution_layers, 3)
        NSE_convolution_summed = tf.reduce_sum(NSE_convolution_concatenated, 3)
        NSE_convolution_extended = tf.expand_dims(NSE_convolution_summed, 3)
        NSE = tf.divide(tf.nn.l2_loss(NSE_convolution_extended - D_obs), tf.nn.l2_loss(D_obs))

        # Cost function
        cost = tf.nn.l2_loss(tf.multiply(W, D_obs - NSE_convolution_extended)) * 2

        # GS
        GS = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(A), 3)))

##################################################################

'''
    Now that we have constructed the graph we can initialize it and feed it to a session for it to do any
    useful calculations.
'''

# Create a session and perform the setup.
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
session.run([image_run, variables_run])

# Create a file writer for TensorBoard.
current_date = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d_%H%M%S")
file_writer = tf.summary.FileWriter("logs/" + current_date, session.graph)
loss_summary = tf.summary.scalar("Loss function", loss)


# Run the image part once and extract the result as a numpy-array.
diffused_image_array = session.run(reshaped_image)

setup_end_time = time.clock()
print("Setup took", format(setup_end_time - setup_start_time, '.2f'), "s")

# Run the network.
cost_values = []
NSE_values = []
GS_values = []
current_folder = os.getcwd()
os.mkdir(os.path.dirname(os.path.join(current_folder,
                                      "runs",
                                      str(current_date) + "_" + str(iterations) + "_iterations_img" + cell_number,
                                      "")))

# Saves time consumption of different code sections to TensorBoard.
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
for iteration in range(iterations):
    if save_graph_information:
        session.run([convolution_run, difference_run, regularizer_run, multiply_run, speedup_run],
                    options=options, run_metadata=run_metadata)
    else:
        session.run([convolution_run, difference_run, regularizer_run, multiply_run, speedup_run])

    if (iteration + 1) % 10 == 0 and iteration != 0:
        # Saves the loss and current image to TensorBoard.
        loss_value = session.run(loss_summary)
        file_writer.add_summary(loss_value, iteration)

        if save_graph_information:
            file_writer.add_run_metadata(run_metadata, 'Step%d' % iteration)

        current_cost_value = session.run(cost)
        NSE_values.append(session.run(NSE))
        GS_values.append(session.run(GS))
        cost_values.append(current_cost_value)
        print("Iteration:", format(iteration + 1, '4.0f'), "| Cost:", format(current_cost_value, '.0f'),
              "(", format(10 * np.log10(current_cost_value), "2.2f"), "dB)")

    if iteration % 100 == 0 and iteration != 0:
        print("Iteration:", format(iteration + 1, '4.0f'))
        print("Total time spent iterating:", format(time.clock() - setup_end_time, '.2f'), "s")
        print("Average time per iteration:", format((time.clock() - setup_end_time) / (iteration + 1), '.2f'), "s")
        iterations_left = iterations - iteration + 1
        time_left_seconds = (time.clock() - setup_end_time) / (iteration + 1) * iterations_left
        print("Approximated time left:", format(time_left_seconds / 60, ".0f"), "min")

        # Retrieve data.
        data_output = session.run(output_img_op)
        # Change type and normalize.
        output_img_array = np.asarray(data_output[0, :, :])
        output_img_array *= (255.0 / output_img_array.max())
        output_img_array = np.resize(output_img_array.astype(np.int8),
                                     (output_img_array.shape[0], output_img_array.shape[1]))

        # Save output as an image.
        inversely_diffused_result_image = Image.fromarray(output_img_array, "L")
        inversely_diffused_result_image.save(os.path.join(current_folder,
                                                          "runs", str(current_date) + "_" + str(iterations)
                                                          + "_iterations_img" + cell_number,
                                                          "inverse_diffusion_after_" + str(
                                                              iteration) + "_iterations.tiff"))

print("Total time running time:", format(time.clock() - setup_end_time, '.2f'), "s")

##################################################################

'''
    Now that we have run the algorithm network for a certain amount of iterations we can extract the
    inversely diffused image and save/display it.
'''

# Retrieves the image from the network.
data_output = session.run(output_img_op)

# Changes type and resizes
output_img_array = np.asarray(data_output[0, :, :])
output_img_array *= (255.0 / output_img_array.max())
output_img_array = np.resize(output_img_array.astype(np.uint8), (output_img_array.shape[0], output_img_array.shape[1]))

inversely_diffused_result_image = Image.fromarray(output_img_array, "L")
inversely_diffused_result_image.show()
inversely_diffused_result_image.save(os.path.join(current_folder,
                                                  "runs",
                                                  str(current_date) + "_" + str(iterations)
                                                  + "_iterations_img" + cell_number,
                                                  "inverse_diffusion_final.tiff"))

np.save("cost_values.npy", cost_values)
np.save("NSE_values.npy", NSE_values)
np.save("GS_value.npy", GS_values)
np.save("iterations.npy", iterations)

##################################################################
'''
    Finally we close the file writer and the session just to avoid any buggy memory leaks.
'''

file_writer.close()
session.close()
