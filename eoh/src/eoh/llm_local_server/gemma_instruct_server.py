import gc
import os
from argparse import ArgumentParser

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

default_model_path_path = 'gemma-2b-it'

# arguments
parser = ArgumentParser()
parser.add_argument('--d', nargs='+', default=['0'])
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--path', type=str, default=default_model_path_path)
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=11015)
args = parser.parse_args()

# cuda visible devices
cuda_visible_devices = ','.join(args.d)
# os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

# set quantization (do not quantization by default)
# set quantization (do not quantization by default)
if args.quantization:
    quantization_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        # llm_int8_enable_fp32_cpu_offload=True
    )

else:
    quantization_config = None

# CodeLlama-Python model
pretrained_model_path = args.path
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
)
# config.pretraining_tp = 1
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
    torch_dtype=torch.float16,
    config=config,
    quantization_config=quantization_config,
    # device_map='auto',
)

# tokenizer for the LLM
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
)

# Flask API
app = Flask(__name__)
CORS(app)


@app.route(f'/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']

    # due to the limitations of the GPU devices in the server, the maximum repeat prompt have to be restricted
    max_repeat_prompt = 20

    print(f'========================================== Prompt ==========================================')
    print(f'{prompt}\n')
    print(f'============================================================================================')
    print(f'\n\n')

    max_new_tokens = 512
    temperature = None
    do_sample = True
    top_k = None
    top_p = None

    if 'params' in content:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', max_new_tokens)
        temperature = params.get('temperature', temperature)
        do_sample = params.get('do_sample', do_sample)
        top_k = params.get('top_k', top_k)
        top_p = params.get('top_p', top_p)

    while True:
        inputs = tokenizer(prompt, return_tensors='pt').to('cpu')

        try:
            # LLM inference
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
            )
        except torch.cuda.OutOfMemoryError as e:
            print("out of memory error")
            # clear cache
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            # decrease repeat_prompt num
            continue

        content = []
        for i, out_ in enumerate(output):
            content.append(tokenizer.decode(output[i, len(inputs[i]):], skip_special_tokens=True))

        print(f'======================================== Response Content ========================================')
        print(f'{content}\n')
        print(f'==================================================================================================')
        print(f'\n\n')

        # clear cache
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        # Send back the response.
        return jsonify(
            {'content': content}
        )


if __name__ == '__main__':
    app.run(host=args.host, port=args.port, threaded=False)
