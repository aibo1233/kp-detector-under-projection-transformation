import os
import shutil


path_dataset = '../../dataset/'
target_path = 'dataset/'
result_path = 'output/'
img_list_file = 'image_list.txt'


# pass
# Resave the results back to the corresponding folder
file_list = []
for filename in os.listdir(result_path):
    if(filename[-1]!='y'):
        continue
    file_list.append(filename)

    name_list = filename.split('_')
    if(len(name_list)==5):
        # str_dir = filename[:-22]
        # str_type = filename[-21:-4]
        str_dir = filename[:-30]
        str_type = filename[-29:-12]
    elif(len(name_list)==3):
        str_dir = name_list[0]+'_'+name_list[1]
        # str_type = name_list[2][:-4]
        str_type = name_list[2][:-12]


    print(str_dir)
    print(str_type)

    path2save = path_dataset+str_dir+'/'+str_type+'/keynet_result/'
    if(not os.path.exists(path2save)):
        os.makedirs(path2save)
    if(str_type=='src'):
        shutil.copy(result_path+filename,path2save+'kp_src.txt')
    else:
        shutil.copy(result_path+filename,path2save+'kp_dst.txt')