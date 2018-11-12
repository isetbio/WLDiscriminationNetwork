import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.data.preprocess import resizeTensors, normalizeByIndividualMean, cropBlockResize, extractNiftiFromZipArchive, getArrayFromNifti
import os


def predictFromArchive(archivPath, net, wantedShape=(41, 53, 38, 6), crop=(slice(4, 28), slice(20, 44), slice(7, 31)), resizeFactor = 2):
    tensorFile = extractNiftiFromZipArchive(archivPath)
    tensor, name = getArrayFromNifti(tensorFile)
    os.remove(tensorFile)
    tensor = [tensor]
    tensor = resizeTensors(tensor, wantedShape)
    tensor = cropBlockResize(tensor, resizeFactor, crop)
    tensor = normalizeByIndividualMean(tensor)
    tensor = np.stack(tensor)
    tensor = torch.from_numpy(tensor).type(torch.float32)
    tensor = Variable(tensor).view(-1, net.fc1.in_features)

    net_out = net(tensor)
    prediction = net_out.max(1)[1]

    predictionStringArrOld = ["no axis is flipped", "the x axis is flipped", "the y axis is flipped", "the z axis is flipped", "it has no idea what's happening"]


    predictionStringArrProfessional = ["you can process the data as is",
                                       "you should flip the x axis in the bvec",
                                       "you should flip the y axis in the bvec",
                                       "you should flip the z axis in the bvec",
                                       "You should check this subject manually"]
    for i, name in enumerate([name]):
        predCertainty = F.softmax(net_out[i], dim=0)[prediction[i]].detach().numpy()*100
        pred = prediction[i]
        if predCertainty < 99:
            print(f"{predictionStringArrProfessional[4]} for {name[i]}. ({100-predCertainty}% unsure")
        else:
            print(f"I am {predCertainty:.3f}% pseudo sure that {predictionStringArrProfessional[pred]} for {name[i]}.")
        print(f"[Pseudo certainty is at {predCertainty}% for {predictionStringArrOld[pred]}]")
