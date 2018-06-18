from xlrd import open_workbook
from Motion import *

class Reader:

    @staticmethod
    def readTrainingFeatures() -> (list,list):

        training_data = open_workbook('new_training.xls')       # read training excell file
        worksheet = training_data.sheet_by_name("Sheet 1")
        number_of_rows = worksheet.nrows
        number_of_columns = worksheet.ncols
        training_data_result = []
        for row in range(0, number_of_rows):
            row_data = []
            for col in range(2, number_of_columns):
                data = worksheet.cell_value(row, col)
                row_data.append(data)
            training_data_result.append(row_data)               # append each frame vector features
        fake_train = []
        orig_train = []

        list_row = len(training_data_result)
        list_col = len(training_data_result[0])
        for row in range(list_row):                    # get fake frames and original frames from training data
            if training_data_result[row][0] == 0:
                fake_train.append(training_data_result[row][1:list_col])
            elif training_data_result[row][0] == 1:
                orig_train.append(training_data_result[row][1:list_col])
        return fake_train, orig_train

    def read_video( path: str) ->(list,int):

        frames=[]                               # list to store video frames
        cap = cv2.VideoCapture(path)            # read video using openCV
        fps = cap.get(cv2.CAP_PROP_FPS)         # calculate number of frames per second
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()             # read video frame sequentially
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # convert frame to gray scale
                gray = np.array(gray).astype(int)
                frames.append(gray)                                # add frame pixels to list
            else:
                break                        # break when all video frames are loaded
        #  release the capture
        cap.release()
        cv2.destroyAllWindows()
        return (frames, fps)





