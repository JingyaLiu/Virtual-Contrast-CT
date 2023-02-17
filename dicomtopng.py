import cv2
import os,os.path,io,glob
import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import os
import shutil
import pydicom.uid
import sys
import PIL.Image as Image
from skimage import io, transform

from shutil import copyfile
#################################### window size ####################################

win_dict = {'abdomen':
            {'wl/wc': 40, 'ww': 400},
            'angio':
            {'wl': 300, 'ww': 600},
            'bone':
            {'wl': 300, 'ww': 1500},
            'brain':
            {'wl': 40, 'ww': 80},
            'chest':
            {'wl': 40, 'ww': 400},
            'lungs':
            {'wl': -400, 'ww': 1500}}


#################################### function to process DICOM ####################################


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = img_temp.min()
    max = img_temp.max()
    # min = (2 * wincenter - winwidth) / 2.0 + 0.5
    # max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp



sys_is_little_endian = (sys.byteorder == 'little')

NumpySupportedTransferSyntaxes = [
    pydicom.uid.ExplicitVRLittleEndian,
    pydicom.uid.ImplicitVRLittleEndian,
    pydicom.uid.DeflatedExplicitVRLittleEndian,
    pydicom.uid.ExplicitVRBigEndian,
]


def supports_transfer_syntax(dicom_dataset):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.
        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return (dicom_dataset.file_meta.TransferSyntaxUID in
            NumpySupportedTransferSyntaxes)


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    return False



def get_pixeldata(dicom_dataset):
    """If NumPy is available, return an ndarray of the Pixel Data.
    Raises
    ------
    TypeError
        If there is no Pixel Data or not a supported data type.
    ImportError
        If NumPy isn't found
    NotImplementedError
        if the transfer syntax is not supported
    AttributeError
        if the decoded amount of data does not match the expected amount
    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.
    """
    if (dicom_dataset.file_meta.TransferSyntaxUID not in
            NumpySupportedTransferSyntaxes):
        raise NotImplementedError("Pixel Data is compressed in a "
                                  "format pydicom does not yet handle. "
                                  "Cannot return array. Pydicom might "
                                  "be able to convert the pixel data "
                                  "using GDCM if it is installed.")


    #dicom_dataset.

    # if not have_numpy:
    #     msg = ("The Numpy package is required to use pixel_array, and "
    #            "numpy could not be imported.")
    #     raise ImportError(msg)
    if 'PixelData' not in dicom_dataset:
        raise TypeError("No pixel data found in this dataset.")

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.BitsAllocated == 1:
        # single bits are used for representation of binary data
        format_str = 'uint8'
    elif dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_dtype = np.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    if dicom_dataset.is_little_endian != sys_is_little_endian:
        numpy_dtype = numpy_dtype.newbyteorder('S')

    pixel_bytearray = dicom_dataset.PixelData

    if dicom_dataset.BitsAllocated == 1:
        # if single bits are used for binary representation, a uint8 array
        # has to be converted to a binary-valued array (that is 8 times bigger)
        try:
            pixel_array = np.unpackbits(
                np.frombuffer(pixel_bytearray, dtype='uint8'))
        except NotImplementedError:
            # PyPy2 does not implement numpy.unpackbits
            raise NotImplementedError(
                'Cannot handle BitsAllocated == 1 on this platform')
    else:
        pixel_array = np.frombuffer(pixel_bytearray, dtype=numpy_dtype)
    length_of_pixel_array = pixel_array.nbytes
    expected_length = dicom_dataset.Rows * dicom_dataset.Columns
    if ('NumberOfFrames' in dicom_dataset and
            dicom_dataset.NumberOfFrames > 1):
        expected_length *= dicom_dataset.NumberOfFrames
    if ('SamplesPerPixel' in dicom_dataset and
            dicom_dataset.SamplesPerPixel > 1):
        expected_length *= dicom_dataset.SamplesPerPixel
    if dicom_dataset.BitsAllocated > 8:
        expected_length *= (dicom_dataset.BitsAllocated // 8)
    padded_length = expected_length
    if expected_length & 1:
        padded_length += 1
    if length_of_pixel_array != padded_length:
        raise AttributeError(
            "Amount of pixel data %d does not "
            "match the expected data %d" %
            (length_of_pixel_array, padded_length))
    if expected_length != padded_length:
        pixel_array = pixel_array[:expected_length]
    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"
    if dicom_dataset.Modality.lower().find('ct') >= 0:
        intercept = dicom_dataset.RescaleIntercept if 'RescaleIntercept' in dicom_dataset else -1024
        slope = dicom_dataset.RescaleSlope if 'RescaleSlope' in dicom_dataset else 1
        pixel_array = pixel_array * slope + intercept
        # print('slope: '+ str(dicom_dataset.RescaleSlope) +' intercept' + str(dicom_dataset.RescaleIntercept))
    pixel_array = pixel_array.reshape(dicom_dataset.Rows, dicom_dataset.Columns*dicom_dataset.SamplesPerPixel)
    return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns



#################################### process DICOM for whole  ####################################

# print('test')
# for root, dirs, files in os.walk('/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/9_30/'):
#     # print root
#     # for dir_f in dirs:
#         # print dir_f
#         file_list = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.dcm')]
#         print(file_list)
#         if len(file_list)>30:
#             print('processing:'+ root +': length '+str(len(file_list)))
#             save_path = file_list[0].split('.dcm')[0][:-9]+'_png/'
#             if not os.path.exists(save_path):
#                 os.mkdir(save_path)
#                 for filename in file_list:
#                     ww = 400
#                     wl = 40
#                     # print(filename.split('.dcm')[:-8])
#                     dcm = pydicom.dcmread(filename)
#                     img,rows, cols = get_pixeldata(dcm)
#                     img = setDicomWinWidthWinCenter(img, ww, wl, rows, cols)
#                     patient_id = file_list[0].split(' ')[0].split('/')[-1]
#                     print('save file:' + save_path + filename[-12:-4]+'_'+ patient_id+'.png')
#                     scipy.misc.imsave(save_path + filename[-12:-4]+'_'+ patient_id+'.png',img)

#################################### print Directory and files  ####################################
# print all folder
# for root, dirs, files in os.walk('../DICOM/'):
#         file_list = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.dcm')]
#         if len(file_list)>30:
#             if 'CT PORTAL' in root:
#                 print(root)
            # print('processing:'+ root +': length '+str(len(file_list)))

# # print current  folder
# dirs = os.listdir('../DICOM/')
# for dir_single in dirs:
#     print(dir_single)
#################################### delete diff file  ####################################
# find difference
# diff -r folder1 folder2
def delete_diff():

    path2 = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN/'
    path1 = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN_LABEL_NEW/'
    # path2 = ''

    list1 = os.listdir(path1)
    list2 = os.listdir(path2)

    for file in list1:
        if file not in list2:
            print(file)
            os.remove(path1+file)
# delete_diff()
#################################### check diff for one file  ####################################
# # find difference
# file_one = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/9_30/19091414460159507 JANE112 DOE112/100112 Abdomen DE_DINAMIKBT Adult/CT 400PRE_png/CT000001_19091414460159507.png'
# file_two = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/9_30/19091414460159507 JANE112 DOE112/100112 Abdomen DE_DINAMIKBT Adult/CT PRECONTRAST 3.0_png/CT000012_19091414460159507.png'
#
# file_one_content = Image.open(file_one)
# file_two_content = Image.open(file_two)
#
# # print(test)
# plt.hist(file_one_content, bins = 3)
# plt.show()
# plt.hist(file_two_content, bins = 3)
# plt.show()
#################################### process DICOM for one folder  ####################################

# # dicom_id name
# name = '19092316070077242'
# fdir = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/9_24/19092316070077242 JANE94 DOE94/100094 Abdomen DE_MEZENTERIKANJIOTORAKSBT Adult/'
# file = ['pre_png','early_png', 'middle_png', 'late_png']
# for i in range(1):
#     i = i+1
#     inputdir = fdir + file[i]+'/'
#     outdir = fdir + file[i]+'_png'+'/'
#
#     if not os.path.isdir(outdir):
#         os.mkdir(outdir)
#
#     file_list = [f for f in os.listdir(inputdir) if f.endswith('.dcm')]
#     print('Processing file:', len(file_list))
#
#     for f in file_list:   # remove "[:10]" to convert all images
#
#
#         ############################################################
#         ww = 400
#         wl = 40
#
#         dcm = pydicom.dcmread(inputdir+f)
#         img,rows, cols = get_pixeldata(dcm)
#         img = setDicomWinWidthWinCenter(img, ww, wl, rows, cols)
#         ct_id = int(f[2:2+6])
#         if ct_id > -1: #and ct_id<67:
#             print(ct_id)
#             scipy.misc.imsave(outdir +'CT'+ str(ct_id-8).zfill(6)+ '_'+ name +'.png',img)


#################################### plot four stages  ####################################
def get_png(filepath):
    # ww = 400
    # wl = 40
    ww = 400
    wl = 50
    dcm = pydicom.dcmread(filepath)
    # print(dcm.dir(), dcm.PatientPosition, abs(int(dcm.ImagePositionPatient[2])))
    table_id = abs(int(dcm.ImagePositionPatient[2]))
    img,rows, cols = get_pixeldata(dcm)
    img = setDicomWinWidthWinCenter(img, ww, wl, rows, cols)
    return img, table_id


#################### from dicom to png #####################################
def savepng():
    for i in range(len(file_save)):
        savedir = fdir + file_save[i]+'/'
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        inputdir = fdir + file_dicom[i]+'/'
        file_list = [f for f in os.listdir(inputdir) if f.endswith('.dcm')]
        print('Processing file:', len(file_list))

        for f in file_list:
            # print(fdir+file_dicom[i]+'/'+f)
            img, table_id = get_png(inputdir+f)
            print(table_id)
            cv2.imwrite(savedir + name +'_'+ str(table_id-1) + '.png',np.uint8(img))
            # cv2.imwrite(savedir + name +'_'+ table_id + '.png',np.uint8(img))


######################## delete the extra pre-contrast ct images ##################################

def delete_misalign(list):
    for i in range(len(file_save)):
        inputdir = fdir + file_save[i]+'/'
        file_list = [f for f in os.listdir(inputdir) if f.endswith('.png')]
        print('Processing file:', len(file_list))
        count = 0
        for f in file_list:   # remove "[:10]" to convert all images
            if not os.path.isfile(fdir+file_save[list[0]]+'/'+f) or not os.path.isfile(fdir+file_save[list[1]]+'/'+f) or not os.path.isfile(fdir+file_save[list[2]]+'/'+f) or not os.path.isfile(fdir+file_save[list[3]]+'/'+f):
                print(f)
                count = count + 1
                for j in range(len(file_save)):
                    if os.path.isfile(fdir+file_save[j]+'/'+f):
                        os.remove(fdir+file_save[j]+'/'+f)
        print(count)
# # #
########################### main code #################################
# dicom
def main():
    # fdir = '/home/tensor-server/Downloads/20012214385669270 JOHN263 DOE263/100263 SUPRAAORTIK MRA' + '/'
    # fdir = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/19112514375425138 JOHN156 DOE156/100156 Abdomen DE_DINAMIKBT Adult' + '/'
    fdir = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/all/19080212093207479 JOHN29 DOE29' + '/'
    name = fdir.split('/')[5].split(' ')[0]
    # folder = 'CT DE_PORTAL 3mm F_0.7'
    # folder = 'CT PREKONTRAST 3.0/CT ARTER FAZ 3mm/CT DE_PORTAL 3mm F_0.7/CT GEC FAZ 3.0'
    # folder = 'CT PREKONTRAST 400/CT ARTER FAZ 3mm/CT GEC FAZ 3.0_400/CT DE_PORTAL 3mm F_0.7'
    # folder = 'MR fl3dcor_carotis_post/MR fl3dcor_carotis_post_SUB-2/MR fl3dcor_carotis_post-2/MR fl3dcor_carotis_pre/MR fl3dcor_carotis_post_SUB'
    # file_dicom = [folder.split('/')[0]]#,folder.split('/')[1],folder.split('/')[2], folder.split('/')[3]]#
    # file_save = [file_dicom[0]+'_png_table']#, file_dicom[1]+'_png_table', folder.split('/')[2]+'_png_table',folder.split('/')[3]+'_png_table']#
    file_dicom = [folder.split('/')[0],folder.split('/')[1],folder.split('/')[2], folder.split('/')[3]]#, folder.split('/')[4]]#
    file_save = [file_dicom[0]+'_png_table', file_dicom[1]+'_png_table', folder.split('/')[2]+'_png_table',folder.split('/')[3]]#+'_png_table',folder.split('/')[4]+'_png_table']#
    list=[0,1,2,3,4]
    # delete_misalign(list)

    # savepng()


########################### compare 4 slices #################################
# for i in range(4):
#     outdir = fdir + 'compare'+'/'
#     if not os.path.isdir(outdir):
#         os.mkdir(outdir)
#     inputdir = fdir + file_save[i]+'/'
#     file_list = [f for f in os.listdir(inputdir) if f.endswith('.png')]
#     print('Processing file:', len(file_list))
#     count = 0
#     for f in file_list:   # remove "[:10]" to convert all images
#         img_pre = cv2.imread(fdir+file_save[0]+'/'+f)
#         img_early = cv2.imread(fdir+file_save[1]+'/'+f)
#         img_middle = cv2.imread(fdir+file_save[2]+'/'+f)
#         img_late = cv2.imread(fdir+file_save[3]+'/'+f)
#         print(img_pre.shape)
#         print(img_early.shape)
#         print(img_middle.shape)
#         print(img_late.shape)
#         vis = np.concatenate((img_pre,img_early,img_middle,img_late),axis = 1)
#         cv2.imwrite(outdir + f,vis)
########################### create  #################################
def create_path():
    path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/'
    folders = ['TEST','TRAIN', 'VAL', 'TEST_LABEL', 'TRAIN_LABEL','VAL_LABEL']
    stages = ['EARLY_','PORTAL_','LATE_']
    for s in stages:
        for f in folders:
            print(path+s+f)
            if not os.path.exists(path+s+f):
                os.mkdir(path+s+f)


# create_path()
def get_filename(path):
    file_name = []
    file = os.listdir(path)
    for f in file:
        file_name.append(f.split('_')[0])
    # print(file_name)
    return file_name

def get_folder_query(query, img_path,path_list):
    folder_to_move = []
    for query_current in query:
        for path_current in path_list:
            # get the folder inside the path
            # print(query_current)
            path_current = os.path.join(img_path, path_current)
            for path_current_i in os.walk(path_current):
                path_current_i = path_current_i[0]
                # print('path_current_i', path_current_i)#os.path.join(path_current_i[0],path_current_i[1]))
                index = path_current_i.find(query_current) * path_current_i.find('table')
                # print(index)
                if (index > 1):
                    # print(query_current, path_current_i)
                    folder_to_move.append(path_current_i)
    # print(folder_to_move)
    return folder_to_move


def move_folder(query, query_pre, path_move, path_move_pre, folder_to_move):

    for folder in folder_to_move:
        # find the query
        index = folder.find(query[0]) * folder.find('table')
        if index > 1:
            # print('split',folder.split('/CT')[:-1][0])
            folder_pre = folder.split('/CT')[:-1][0]
        else:
            # print('split',folder.split('/'+query[1])[:-1][0])
            folder_pre = folder.split('/'+query[1])[:-1][0]
        files = os.listdir(folder)
        for query_pre_current in query_pre:
            for file in files:
                # print('processing', folder+'/'+file )
                # find file in pre folder
                for folder_pre_current in os.listdir(folder_pre):
                    index = folder_pre_current.find(query_pre_current) * folder_pre_current.find('table')
                    if index > 1 and os.path.exists(folder_pre+'/'+folder_pre_current+'/'+file):
                        # print(folder_pre_current,file)
                        # print(query_pre_current,folder_pre_current)
                        copyfile(folder_pre+'/'+folder_pre_current+'/'+file, path_move_pre+'/'+file)
                        copyfile(folder +'/'+file, path_move+'/'+file)


def list_path():
    img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/all/'
    #  test
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TEST'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL'
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST_LABEL'
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TEST'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TEST_LABEL'
    # train
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN_LABEL'
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN_LABEL'
    # path_move_pre = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN'
    # path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN_LABEL'
    path = os.listdir(img_path)
    # print(path)
    # path_exist = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST/'
    path_exist = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/portal_new_temp/PORTAL_TRAIN_LABEL_NEW/'
    # path_exist = ''
    filename = get_filename(path_exist)
    path_list = []
    for path_name in path:
        path_name_idx = path_name.split(' ')[0]
        # if path_name_idx not in filename:
        if path_name_idx not in filename:
            path_list.append(os.path.join(img_path,path_name))
    # print(path_list)
    query = ['POR','middle']
    # query = ['ART','late']
    # query = ['GEC','late']

    query_pre = ['pre','PRE']
    folder_to_move = get_folder_query(query, img_path,path_list)
    move_folder(query,query_pre,path_move,path_move_pre,folder_to_move)
    # print(folder_portal)
    # print(path_list)
# list_path()

import cv2
# uniform the size of the input image to 512x512
def resize_image():
    path_move = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN/'
    path_save = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN_LABEL_REDO/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    files = os.listdir(path_move)
    for file in files:
        # print(file)
        image = io.imread(path_move+file)
        if image.shape[0] != 512:
            print(file)
            image = transform.resize(image,(512,512))
            # cv2.imwrite(path_move+file,np.uint8(image))

def list_case():
    # path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN/'
    # path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN/'
    path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN/'
    # path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/portal_new_temp/PORTAL_TRAIN_LABEL_NEW/'
    files = [file.split('_')[0] for file in os.listdir(path)]
    cases = set(files)
    print('The number of the cases are:',len(cases))
    print('The total number of the slices are: ',len(os.listdir(path)))

def check_full_cases():
    path  = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/DICOM/all/'
    count = 0
    folders = os.listdir(path)
    for folder in folders:
        files = os.listdir(path+folder)
        for file in files:
            print(file)
            file_final = [file_current for file_current in os.listdir(path+folder+'/'+file) if file_current.endswith('_table')]
            if len(file_final)>3:
                file_len = os.listdir((path+folder+'/'+file+'/'+file_final[0]))
                count += len(file_len)
    print(count)


# list_case()
# create_path()
# list_path()
# resize_image()
check_full_cases()

