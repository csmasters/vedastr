import argparse
import os
import re
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2

from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    parser.add_argument('gpus', type=str, help='target gpus')
    args = parser.parse_args()

    return args


def main():
    count = 0
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")
    with open("/home/xinyudong/program/pythonProject/tools/test_null.json", 'r') as load_f:
        dicts = json.load(load_f)
    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [os.path.join(args.image, name)
                  for name in os.listdir(args.image)]
    for img in images:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_str, probs = runner(image)
        regex = re.compile(r'\d+')
        num = regex.findall(img)
        dicts['test_' + num[0] + '.JPEG'][int(num[1])]["label"] = pred_str[0]
        count = count + 1
        if count % 100 ==0:
            print(pred_str[0])
        if count % 1000 == 0:
            print("已完成" + str(count) + "次")
            # print(dicts['test_' + str(num[0]) + '.JPEG'][int(num[1])])
        # runner.logger.info('predict string: {} \t of {}'.format(pred_str, img))
    with open("/home/xinyudong/program/pythonProject/tools/test_submission.json", "w") as f:
        json.dump(dicts, f)
        print("加载入文件完成...")


if __name__ == '__main__':
    main()
