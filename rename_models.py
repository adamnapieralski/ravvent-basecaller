import re
from pathlib import Path


def rename_model(model_dir_path, epoch_add):
    model_dir_path = Path(model_dir_path)
    name_match = re.match(
        r'^(.*\/model\.)(\d)(.*\.)(\d+)',
        str(model_dir_path))

    if name_match is not None:
        new_path = Path(f'{name_match[1]}1{name_match[3]}{int(name_match[4]) + epoch_add}')
        print(new_path)
        model_dir_path.rename(new_path)


if __name__ == '__main__':
    for dir_path in [p for p in Path('models/snippets/mask/encd_2_decd_2').iterdir() if 'model.2.joint' in str(p)]:
        print(dir_path)
        rename_model(dir_path, 12)

