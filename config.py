import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('main', default='图像 AI 扩展')

    parser.add_argument('data_path', default='F:/')

    parser.add_argument('epoch_count', default=30)

    args = parser.parse_args()
    return args
