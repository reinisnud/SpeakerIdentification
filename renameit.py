import os
dirsa=[]
result_files = []
testfold = 'C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted\\test'
for root, dirs, files in os.walk(testfold):
    if files:
        result_files.append(sorted(files)[0])
        # dirsa.append(dirs)
    if dirs:
        dirsa.append(dirs)

for filename in result_files:
    print(filename)
for direct in dirsa:
    for dd in direct:
        print(dd)
i=0
for direct in dirsa:
    for dd in direct:
        os.rename(testfold + "\\" + dd + "\\" + result_files[i], testfold + "\\" + dd + "\\" + "enroll.p")
        i=i+1
# files=[]

# for folder in os.listdir('C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted\\test'):
#     print(folder)
#     i=0

#     for filename in os.listdir('C:\\Users\\reini\\Desktop\\SpeakerRecognition_tutorial-master\\sorted\\test' + "\\" + folder):
#         if filename.endswith('.p'): # only get MFCCs from .wavs
#             files.append(folder, filename)
# print(files)