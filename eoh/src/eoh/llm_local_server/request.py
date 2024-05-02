import time

import requests
import json

url = 'http://127.0.0.1:11101/completions'

while True:
    prompt = '''
def priority_v0(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    penalty = np.arange(len(bins), 0, -1)
    scores = bins / (bins - item) - penalty
    max_capacity_bins = np.where(bins == bins.max())[0]
    for idx in max_capacity_bins:
        scores[idx] = -np.inf
    return scores


def priority_v1(item: float, bins: np.ndarray) -> np.ndarray:
    """Improved version of `priority_v0`."""'''

    data = {
        'prompt': prompt,
        'repeat_prompt': 20,
        'system_prompt': '',
        'stream': False,
        'params': {
            'temperature': None,
            'top_k': None,
            'top_p': None,
            'add_special_tokens': False,
            'skip_special_tokens': True,
        }
    }

    # inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021

    headers = {'Content-Type': 'application/json'}

    record_time = time.time()
    response = requests.post(url, data=json.dumps(data), headers=headers)
    durations = time.time() - record_time
    print(f'time: {durations}s')

    # def process_response_content(content: str) -> str:
    #     ret = content.split('[/INST]')[1]
    #     return ret

    if response.status_code == 200:
        print(f'Query time: {durations}')
        # print(f'Response: {response.json()}')
        content = response.json()["content"]

        for c in content:
            # content = process_response_content(content)
            print(f'{c}')
            print(f'----------------------------------------------')
    else:
        print('Failed to make the POST request.')

    break
