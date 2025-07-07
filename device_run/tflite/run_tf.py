import tflite_runtime.interpreter as tflite

import argparse
import time
import numpy as np


print(f"numpy version: {np.__version__}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model_file',
        default='./models/bert-base-uncased.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument(
        '-e', '--ext_delegate', help='external_delegate_library path')
    parser.add_argument(
        '-o',
        '--ext_delegate_options',
        help='external delegate options, \
                format: "option1: value1; option2: value2"')

    args = parser.parse_args()

    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
        if (len(kv) == 2):
            ext_delegate_options[kv[0].strip()] = kv[1].strip()
        else:
            raise RuntimeError('Error parsing delegate option: ' + o)

    # load external delegate
    if args.ext_delegate is not None:
        print('Loading external delegate from {} with args: {}'.format(
            args.ext_delegate, ext_delegate_options))
        ext_delegate = [
            tflite.load_delegate(args.ext_delegate, ext_delegate_options)
        ]


    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    input_ids = [[101, 1045, 2572, 1037, 8957, 1012,  102]]
    attention_mask = [[1, 1, 1, 1, 1, 1, 1]]
    input_ids_np = np.array(input_ids, dtype='int64')
    attention_mask_np = np.array(attention_mask, dtype='int64')

    interpreter.set_tensor(input_details[0]['index'], input_ids_np)
    interpreter.set_tensor(input_details[1]['index'], attention_mask_np)


    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    print(results)