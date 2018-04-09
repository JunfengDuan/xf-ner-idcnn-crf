import os
import xlrd


def create_file(dir_name):
    """
    创建语料空文件：训练集、验证集、测试集
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    xf_train_file = os.path.join(dir_name, 'xf_train.txt')
    xf_val_file = os.path.join(dir_name, 'xf_val.txt')
    xf_test_file = os.path.join(dir_name, 'xf_test.txt')

    if os.path.exists(xf_train_file):
        os.remove(xf_train_file)
    os.mknod(xf_train_file)
    if os.path.exists(xf_val_file):
        os.remove(xf_val_file)
    os.mknod(xf_val_file)
    if os.path.exists(xf_test_file):
        os.remove(xf_test_file)
    os.mknod(xf_test_file)


def get_column_index(table, column_name):
    column_index = None
    for i in range(table.ncols):
        if table.cell_value(0, i) == column_name:
            column_index = i
            break
    return column_index


def read_xls_file(file_name):
    """
    读取文件并转换为一行（去空格）
    :param file_name:
    :return:
    """
    workbook = xlrd.open_workbook(file_name)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    print(file_name, worksheet.name, worksheet.nrows, worksheet.ncols)

    contents_column_index = get_column_index(worksheet, '投诉内容')
    location_column_index = get_column_index(worksheet, '问题属地')

    contents = worksheet.col_values(contents_column_index)
    locations = worksheet.col_values(location_column_index)

    locations_contents = list(zip(contents, locations))
    items = []
    for item in locations_contents[1:len(locations_contents)]:
        clean_content = str(item[0]).replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', '')
        if len(clean_content) < 10 or len(item[1]) < 1:
            continue
        items.append([clean_content, item[1]])

    return items


def process_data(location_content):
    iobes_item = []
    for lc in location_content:
        characters = list(lc[0])
        for c in characters:
            if c not in lc[1]:
                tag = 'O'
            elif c == lc[1][0]:
                tag = 'B-LOC'
            else:
                tag = 'I-LOC'
            char_tag = c + ' ' + tag
            iobes_item.append(char_tag)
        iobes_item.append('\n')
    return iobes_item


def save_file(dir_name):
    """
    构造训练需要的3个文件
    文件内容格式:  类别\t内容
    :param dir_name: 原数据目录
    :return:
    """
    f_train = open('../data/xf_train.txt', 'w', encoding='utf-8')
    f_val = open('../data/xf_val.txt', 'w', encoding='utf-8')
    f_test = open('../data/xf_test.txt', 'w', encoding='utf-8')

    files = os.listdir(dir_name)
    for file in files:
        filename = os.path.join(dir_name, file)
        location_content = read_xls_file(filename)
        iobes = process_data(location_content)
        count = 0
        for text in iobes:
            if count < 20000:
                f_train.write(text + '\n')
            elif count < 22000:
                f_val.write(text + '\n')
            elif count < 27000:
                f_test.write(text + '\n')
            else:
                break
            count += 1

    f_train.close()
    f_val.close()
    f_test.close()
    print('Write finished')

if __name__ == '__main__':
    create_file('../data')
    save_file('../corpus')
    print(len(open('../data/xf_train.txt', 'r', encoding='utf-8').readlines()))  # 60000
    print(len(open('../data/xf_val.txt', 'r', encoding='utf-8').readlines()))  # 6000
    print(len(open('../data/xf_test.txt', 'r', encoding='utf-8').readlines()))  # 10216

