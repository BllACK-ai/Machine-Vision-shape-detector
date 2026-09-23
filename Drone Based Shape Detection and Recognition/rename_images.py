import os
import sys


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TEMP_PREFIX = "tmp_rename_images_"


def configure_output():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def get_temp_index(name):
    base_name, _ = os.path.splitext(name)
    if not base_name.startswith(TEMP_PREFIX):
        return None

    index_text = base_name[len(TEMP_PREFIX):]
    if not index_text.isdigit():
        return None

    return int(index_text)


def main():
    configure_output()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "test")

    if not os.path.isdir(test_dir):
        print("[RENAME] Error: test folder does not exist")
        sys.exit(0)

    image_files = []
    for name in os.listdir(test_dir):
        path = os.path.join(test_dir, name)
        if os.path.isfile(path):
            _, extension = os.path.splitext(name)
            if extension.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(name)

    image_files.sort()
    count = len(image_files)

    if count == 0:
        print("[RENAME] No supported images found in test/")
        sys.exit(0)

    print("[RENAME] Found {} images in test/".format(count))

    temp_files = []
    for name in image_files:
        temp_index = get_temp_index(name)
        if temp_index is not None:
            temp_files.append((temp_index, name))

    if temp_files:
        temp_files.sort()
        renamed_count = 0
        for temp_index, temp_name in temp_files:
            _, extension = os.path.splitext(temp_name)
            final_name = "test{}{}".format(temp_index, extension)
            temp_path = os.path.join(test_dir, temp_name)
            final_path = os.path.join(test_dir, final_name)
            try:
                os.rename(temp_path, final_path)
                renamed_count += 1
                print("[RENAME] {} \u2192 {}".format(temp_name, final_name))
            except OSError as error:
                print("[RENAME] Error renaming {} to {}: {}".format(temp_name, final_name, error))

        print("[RENAME] Complete \u2014 {} files renamed".format(renamed_count))
        return

    rename_plan = []
    for index, old_name in enumerate(image_files, start=1):
        _, extension = os.path.splitext(old_name)
        temp_name = "tmp_rename_images_{}{}".format(index, extension)
        final_name = "test{}{}".format(index, extension)
        rename_plan.append((old_name, temp_name, final_name))

    successful_temp_renames = []
    for old_name, temp_name, final_name in rename_plan:
        old_path = os.path.join(test_dir, old_name)
        temp_path = os.path.join(test_dir, temp_name)
        try:
            os.rename(old_path, temp_path)
            successful_temp_renames.append((old_name, temp_name, final_name))
        except OSError as error:
            print("[RENAME] Error renaming {} to {}: {}".format(old_name, temp_name, error))

    renamed_count = 0
    for old_name, temp_name, final_name in successful_temp_renames:
        temp_path = os.path.join(test_dir, temp_name)
        final_path = os.path.join(test_dir, final_name)
        try:
            os.rename(temp_path, final_path)
            renamed_count += 1
            print("[RENAME] {} \u2192 {}".format(old_name, final_name))
        except OSError as error:
            print("[RENAME] Error renaming {} to {}: {}".format(temp_name, final_name, error))

    print("[RENAME] Complete \u2014 {} files renamed".format(renamed_count))


if __name__ == "__main__":
    main()
