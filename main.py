import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from made_classes import*
from made_classes_input import*
from made_classes_layer import*

#참고한 사이트들은 주석처리 되어 코드 내에 적혀 있습니다. 

def make_dataloader(itemuser_matrix, movie_k, user_k):
    itemuser_matrix_org = np.copy(itemuser_matrix)
    train_x = []
    train_y = []
    for i in range(len(movie_k)):
        for j in range(len(user_k[0])):
            if itemuser_matrix_org[i][j] != 0:
                new_train = np.zeros((k*3, ))
                new_train[:k] = np.multiply(movie_k[i, :], user_k[:, j])
                new_train[k : 2*k] = movie_k[i, :]
                new_train[2*k : ] = user_k[:, j]

                train_x.append(new_train)
                train_y.append([itemuser_matrix_org[i][j]])
    
    #https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
    np_train_x = np.asarray(train_x)
    np_train_y = np.asarray(train_y)
    tensor_train_x = torch.Tensor(np_train_x)
    tensor_train_y = torch.Tensor(np_train_y)


    dataset_train = TensorDataset(tensor_train_x, tensor_train_y)
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)

    return dataloader_train


def train_fn(dataloader_train, device, optimizer, model, loss_fn):
    #https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    avg_loss = 0
    last_loss = 0
    for i, data in enumerate(dataloader_train):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs[:, k:], inputs[:, :k])
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss

    avg_loss = avg_loss/(i+1)
    last_loss = loss
    
    return avg_loss, last_loss

def train_loop(epochs, dataloader_train, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = all_neumf()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.MSELoss()

    model.to(device)

    min_loss = 1000
    count_nodecrease = 0
    avg_loss_list = []
    for i in range(epochs):
        print("Epochs: ", i)
        avg_loss, last_loss = train_fn(dataloader_train, device, optimizer, model, loss_fn)
        print("Average Loss: ", avg_loss)
        avg_loss_list.append(avg_loss)
        print("Last Loss: ", last_loss, "\n")

        
        if min_loss > avg_loss:
            min_loss = avg_loss
            count_nodecrease = 0
        else:
            count_nodecrease += 1

        if count_nodecrease == 5:
            break
    
    #https://eehoeskrap.tistory.com/618
    torch.save(model.state_dict(), "./saved_models/" + model_name)
    return avg_loss_list
    
def test_rating(userId, movieId, dict_userId_idx, dict_movieId_idx, device, model, k):
    curr_useridx = dict_userId_idx[userId]
    curr_movieidx = dict_movieId_idx[movieId]

    test_input = np.zeros((3*k, ))
    test_input[:k] = np.multiply(movie_k[curr_movieidx], user_k[:, curr_useridx])

    test_input[k : k*2] = movie_k[curr_movieidx]
    test_input[k*2 : ] = user_k[:, curr_useridx]
    test_input = np.array([test_input])

    test_input = torch.Tensor(test_input)
    test_input = test_input.to(device)
    
    return model(test_input[:, k:], test_input[:, :k])



if __name__ == '__main__':
    df_imgurl = pd.read_csv('./data/movies_w_imgurl.csv')
    df_rating_train = pd.read_csv('./data/ratings_train.csv')
    df_rating_val = pd.read_csv('./data/ratings_val.csv')
    df_tags = pd.read_csv('./data/tags.csv')

    all_movieId = []
    all_userId = []

    for i in range(len(df_rating_train)):
        all_movieId.append(df_rating_train["movieId"][i])
        all_userId.append(df_rating_train["userId"][i])

    all_movieId = set(all_movieId)
    all_userId = set(all_userId)

    all_movieId = sorted(all_movieId)
    all_userId = sorted(all_userId)

    
    dict_movieId_idx = {}
    dict_userId_idx = {}
    dict_idx_movieId = {}
    dict_idx_userId = {}

    for i in range(len(all_movieId)):
        dict_movieId_idx[all_movieId[i]] = i
        dict_idx_movieId[i] = all_movieId[i]

    for i in range(len(all_userId)):
        dict_userId_idx[all_userId[i]] = i
        dict_idx_userId[i] = all_userId[i]
    
    # ###
    # pickle.dump(dict_movieId_idx, open("saved/dict_movieId_idx.pkl", "wb"))
    # pickle.dump(dict_idx_movieId, open("saved/dict_idx_movieId.pkl", "wb"))
    # pickle.dump(dict_userId_idx, open("saved/dict_userId_idx.pkl", "wb"))
    # pickle.dump(dict_idx_userId, open("saved/dict_idx_userId.pkl", "wb"))
    ###

    #https://velog.io/@yookyungkho/210114-TIL-Today-I-Learned
    #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    itemuser_matrix = np.zeros((len(all_movieId), len(all_userId)))
    for i in range(len(df_rating_train)):
        curr_movieidx = dict_movieId_idx[df_rating_train["movieId"][i]]
        curr_useridx = dict_userId_idx[df_rating_train["userId"][i]]

        itemuser_matrix[curr_movieidx][curr_useridx] = df_rating_train["rating"][i]


    ###----------------MATRIX FACTORIZATION-------------------------------------###
    itemuser_matrix_ncf = np.copy(itemuser_matrix)
    #Prj2 pdf 참고
    movie_avg = np.zeros((len(itemuser_matrix_ncf)))

    for i in range(len(itemuser_matrix_ncf)):
        movie_sum = 0
        movie_not_zero = 0
        for j in range(len(itemuser_matrix_ncf[0])):
            movie_sum += itemuser_matrix_ncf[i][j]
            if itemuser_matrix_ncf[i][j] != 0:
                movie_not_zero += 1
        movie_avg[i] = movie_sum/movie_not_zero

    for i in range(len(itemuser_matrix_ncf)):
        for j in range(len(itemuser_matrix_ncf[0])):
            if itemuser_matrix_ncf[i][j] == 0:
                itemuser_matrix_ncf[i][j] = movie_avg[i]

    movie, mu, user = np.linalg.svd(itemuser_matrix_ncf)

    ###
    # np.save("saved/movie_f.npy", movie)
    # np.save("saved/mu_f.npy", mu)
    # np.save("saved/user_f.npy", user)
    ###

    k = 400
    movie_k = movie[:, :k]
    mu_k = mu[:k]
    user_k = user[:k]
    ###----------------MATRIX FACTORIZATION-------------------------------------###


    ###----------------TRAIN---------------------------------------------------###
    # dataloader_train = make_dataloader(itemuser_matrix, movie_k, user_k)
    # epochs = 30
    # model_name = 'neuCF_30_lr0.0001_m0.9_input512_layer2'
    # avg_loss_list = train_loop(epochs, dataloader_train, model_name)
    # #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
    # pickle.dump(avg_loss_list, open('./saved_loss/avg_loss_list_'+ model_name, "wb"))
    ###----------------TRAIN---------------------------------------------------###
    

    ###----------------INFERENCE------------------------------------------------###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf_model = all_neumf()
    inf_model.load_state_dict(torch.load('param.data'))
    inf_model.to(device)

    #https://www.w3schools.com/python/python_file_open.asp
    f = open("input.txt", "r")

    input_info = []
    for f_line in f:
        split_info = f_line.split(",")
        input_info.append([int(split_info[0]), int(split_info[1])])
    
    output_info = []
    for i in input_info:
        inf_rating = test_rating(i[0], i[1], dict_userId_idx, dict_movieId_idx, device, inf_model, k)
        inf_rating = inf_rating.cpu().detach().numpy()
        #https://jsikim1.tistory.com/226
        inf_rating = format(inf_rating[0][0], ".8f")
        output_info.append([i[0], i[1], inf_rating])
    
    #https://www.pythontutorial.net/python-basics/python-write-text-file/
    with open('output.txt', "w") as f_o:
        for i in output_info:
            f_o.write(str(i[0]) + "," + str(i[1]) + "," + i[2] + "\n")



    ###----------------INFERENCE------------------------------------------------###