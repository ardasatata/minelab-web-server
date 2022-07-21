import sys
from classifier.ErhuPrediction3DCNNLSTM_class import check_player_postition

inputVideo = sys.argv[1]

# main_predict('/home/minelab/dev/erhu-project/upload/04_19_2022_02_38_16_04_15_2022_10_04_25_01.mp4')
print('Checker')
check = check_player_postition(inputVideo)

print(check)

if check:
    pass
else:
    raise Exception("Video doesn't meet criteria")
