from xlrd import open_workbook
from SVMTesting import *
from Motion import *


def svm_classifier() -> svm.SVC:
    training_data = open_workbook('training.xls')
    worksheet = training_data.sheet_by_name("Sheet1")
    number_of_rows = worksheet.nrows
    number_of_columns = worksheet.ncols
    training_data_result = []
    for row in range(1, number_of_rows):
        row_data = []
        for col in range(1, number_of_columns):
            data = worksheet.cell_value(row, col)
            row_data.append(data)
        training_data_result.append(row_data)
    fake_train = []
    orig_train = []
    # get fake frames and original frames from training data
    list_row = len(training_data_result)
    list_col = len(training_data_result[0])
    for row in range(list_row):
        if training_data_result[row][0] == 'fake':
            fake_train.append(training_data_result[row][1:list_col])
        elif training_data_result[row][0] == 'original':
            orig_train.append(training_data_result[row][1:list_col])

    # get testing data
    # testing_data = open_workbook('testing.xlsx')
    # worksheet = testing_data.sheet_by_name("Sheet1")
    # number_of_rows = worksheet.nrows
    # number_of_columns = worksheet.ncols
    # testing_data_result = []
    # testing_label = []
    # testing_data = []
    # for row in range(1, number_of_rows):
    #     row_data = []
    #     for col in range(1, number_of_columns):
    #         data = worksheet.cell_value(row, col)
    #         row_data.append(data)
    #     testing_data_result.append(row_data)
    #
    # testing_col = len(testing_data_result[0])
    # for row in range(number_of_rows-1):
    #     testing_label.append(testing_data_result[row][0])
    #     testing_data.append(testing_data_result[row][1:testing_col])

    return classifyFrames(fake_train, orig_train)


clf = svm_classifier()
m = Motion(clf)
m.read_video('E:\\CS\\4th year\\GP\Papers\\video_tampering_dataset\\videos\\h264_lossless\\10_forged.mp4')
m.compute_features()
x = m.get_fake_time()
print(x)
