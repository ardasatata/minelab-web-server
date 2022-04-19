import sys
from ErhuPrediction3DCNNLSTM_class import main_predict

inputVideo = sys.argv[1]

# main_predict('/home/minelab/dev/erhu-project/upload/04_19_2022_02_38_16_04_15_2022_10_04_25_01.mp4')
main_predict(inputVideo)