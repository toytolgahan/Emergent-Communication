import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt

#CIFAR-10 DATA
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        images = images.reshape(-1,3,32,32)
        images = images.transpose(0,2,3,1)
    return images
def load_cifar10(data_dir):
    train_images = []
    for i in range(1,6):
        file_path = os.path.join(data_dir, 'data_batch_' + str(i))
        batch_images = load_batch(file_path)
        train_images.append(batch_images)
    train_images = np.vstack(train_images)
    test_images = load_batch(os.path.join(data_dir, 'test_batch'))
    return train_images, test_images
data_dir = "cifar-10-batches-py"

cifarData = load_cifar10(data_dir)

#ENCODER, DECODER, EYE
class Encoder(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(Encoder,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size,hidden_size)
    def forward(self,input,hidden):
        output = input.view(1,1,-1)
        _, hidden = self.gru(output,hidden)
        return hidden
    def initHidden(self):
        return torch.zeros(self.hidden_size).view(1,1,-1)

class Decoder(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(Decoder,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,vocab_size)
    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output,hidden = self.gru(output,hidden)
        probs = self.out(output)
        return output,hidden,probs

class Eye(nn.Module):
    def __init__(self,hidden_size):
        super(Eye,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        #Fully connected layers
        self.fc1 = nn.Linear(in_features=128*2*2, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=hidden_size)
    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(x)
    
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.relu(x)
       
        
        x = self.pool(x)
        
        x =  x.reshape(1,-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#HYPERPARAMETERS - INITIALIZE AGENTS - IMAGES

MAX_LENGTH = 20
NUMBER_OF_AGENTS = 10
hidden_size = 120
vocab_size = 10
sos_token = 1
eos_token = 2
criterion = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=-1)
learning_rate = 1e-8

agents = []
optimizers = []
for i in range(NUMBER_OF_AGENTS):
    agent = []
    agent_optimizer = []
    eye = Eye(hidden_size)
    encoder = Encoder(vocab_size,hidden_size)
    decoder = Decoder(vocab_size,hidden_size)
    agent.append(eye)
    agent.append(decoder)
    agent.append(encoder)
    agents.append(agent)
    
    eye_optimizer = optim.SGD(eye.parameters(),lr=learning_rate)
    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
    agent_optimizer.append(eye_optimizer)
    agent_optimizer.append(decoder_optimizer)
    agent_optimizer.append(encoder_optimizer)
    optimizers.append(agent_optimizer)

#LOAD PARAMETERS
for i in range(NUMBER_OF_AGENTS):
    param_name = 'eye' + str(i) + '_parameters.pth'
    if os.path.exists(param_name):
        agents[i][0].load_state_dict(torch.load(param_name))
        param_name = 'decoder' + str(i) + '_parameters.pth'
        agents[i][1].load_state_dict(torch.load(param_name))
        param_name = 'encoder' + str(i) + '_parameters.pth'
        agents[i][2].load_state_dict(torch.load(param_name))



images = torch.from_numpy(cifarData[0]).float()

test_images = torch.from_numpy(cifarData[1]).float()



def privateTraining(agent,agent_optimizer,epochs):
    eye,decoder,encoder = agent
    eye_optimizer,encoder_optimizer,decoder_optimizer = agent_optimizer
    for epoch in range(epochs):
        targets = []
        loss = 0
        for n in range(len(images)):
            
            image = images[n].unsqueeze(0)
            hidden = eye(image).view(1,1,-1)
            target = hidden
            targets.append(target)
            
            #DISTRACTOR IMAGES
            distractor_images = []
            for m in range(4):
                if len(targets) >10:
                    distractor_image = targets[random.randint(0,len(targets)-1)-1]
                else:
                    distractor_image = torch.zeros(hidden.shape)
                distractor_images.append(distractor_image)
            
            decoder_outputs = torch.zeros(MAX_LENGTH, hidden_size)
            input = torch.tensor([0])
            text = []
            word_length = 0
            for di in range(MAX_LENGTH):
                output, hidden, probs = decoder(input,hidden)
                decoder_outputs[di] = output
                topv, topi = probs.topk(1)
                input = topi.detach()
                
                if input.item() == eos_token:
                    break
                word_length += 1
                text.append(input.item())
            for ei in range(word_length):
                input = decoder_outputs[ei]
                hidden = encoder(input,hidden)
            loss += 1 - cos(hidden,target)
            for distractor in distractor_images:
                loss += cos(hidden,distractor)
        
        if epoch%20==0:
            print("at epoch {} Loss is {}".format(epoch,loss.item()))
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        eye_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        eye_optimizer.step()

def privateSocialTraining(agent,agent_optimizer,images,text,epochs):
    text = text.long()
    eye,decoder,_ = agent
    eye_optimizer,decoder_optimizer,_ = agent_optimizer
    
    for epoch in range(epochs):
        loss = 0
        accurate = 0
        for n in range(len(images)):
            image = images[n].unsqueeze(0)
            hidden = eye(image).view(1,1,-1)
            decoder_outputs = torch.zeros(MAX_LENGTH,vocab_size)
            input = torch.tensor([0])
            output_text = torch.zeros(text[n].shape).int()
            for di in range(MAX_LENGTH):
                output,hidden,probs = decoder(input,hidden)
                decoder_outputs[di] = probs
                topv, topi = probs.topk(1)
                input = topi.detach()
                if input.item() == eos_token:
                    break
                
                output_text[di] = input.item()
                loss += criterion(probs[0],text[n][di].unsqueeze(0))
            accurate += (text[n]==output_text).all()
            #print(text[n])
            #print(output_text)
        accuracy = accurate/len(images)
        eye_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        eye_optimizer.step()
        decoder_optimizer.step()
        print("at epoch {} loss is {} and the accuracy is {} ".format(epoch, loss.item(), accuracy))
 

def get_data(agent,images, num_of_images):
    eye, decoder,_ = agent
   
    text_data = []
    image_data = torch.zeros((num_of_images,)+(images.shape[1:]))
    new_order = torch.randperm(images.size(0))
    shuffled_images = images[new_order]
    for n in range(num_of_images):
        image = shuffled_images[n].unsqueeze(0)
        image_data[n] = image
        hidden = eye(image).view(1,1,-1)
        target = hidden
        decoder_outputs = torch.zeros(MAX_LENGTH, hidden_size)
        input = torch.tensor([0])
        text = [0 for i in range(MAX_LENGTH)]
        for di in range(MAX_LENGTH):
            output, hidden, probs = decoder(input,hidden)
            decoder_outputs[di] = output
            topv, topi = probs.topk(1)
            input = topi.detach()
            if input.item() == eos_token:
                    break
            text[di] = input.item()
        text_data.append(text)
    return image_data, text_data

#INTERSUBJECTIVE SPACE

from pyTsetlinMachine.tm import MultiClassTsetlinMachine, MultiClassConvolutionalTsetlinMachine2D
from pyTsetlinMachine.tools import Binarizer
import numpy as np

nth_words = lambda data, n: [i[n] for i in data]
multiTMs = []
for i in range(MAX_LENGTH):
    tm = MultiClassConvolutionalTsetlinMachine2D(150,60,3.9,(2,2),boost_true_positive_feedback=0)
    multiTMs.append(tm)

def trainTM(image_data,text_data):
    total = 0
    for n, tm in enumerate(multiTMs):
        tm.fit(image_data,nth_words(text_data,n),epochs=20)
        accurate = (tm.predict(image_data) == nth_words(text_data,n)).sum()
        print(" {} / {} accurate: {}".format(n, len(multiTMs),accurate))
        total += accurate
    print("accuracy: ",total/((n+1)*len(text_data)))


#TRAINING

#TRAIN AGENTS PRIVATELY
print("Training agents privately")
for i in range(len(agents)):
    print("AGENT{}:".format(i))
    privateTraining(agents[i],optimizers[i],2000)


#CONSTRUCT PRIVATE LANGUAGE DATA
print("Emergence of private languages")
image_data, text_data = get_data(agents[0],images,500)
for i in range(1,len(agents)):
    image_data = torch.cat((image_data, get_data(agents[i],images,500)[0]),dim=0)
    text_data += get_data(agents[i],images,500)[1]
    
#TRAIN THE COMMUNITY
print("private to rule based through Tsetlin Machine")
trainTM(image_data,text_data)

#CONSTRUCT PUBLIC LANGUAGE DATA
print("emergence of the public language")
social_text = torch.zeros(len(text_data),MAX_LENGTH)
for n, tm in enumerate(multiTMs):
    social_text[:,n] = torch.from_numpy(tm.predict(image_data).astype(float))
    
#TRAIN WITH COMMUNITY SUPERVISION
print("train agents to learn the public language")
for i in range(len(agents)):
    privateSocialTraining(agents[i],optimizers[i],image_data,social_text,3000)

# SAVE THE PARAMETERS
for i in range(NUMBER_OF_AGENTS):
    param_name = 'eye' + str(i) + '_parameters.pth'
    torch.save(agents[i][0].state_dict(), param_name)
    param_name = 'decoder' + str(i) + '_parameters.pth'
    torch.save(agents[i][1].state_dict(), param_name)
    param_name = 'encoder' + str(i) + '_parameters.pth'
    torch.save(agents[i][2].state_dict(), param_name)



#TEST
print("test if agents are able to communicate in this new language")

for i in range(len(test_images)):
    accurate = 0
    a1 = random.randint(0,len(agents)-1)
    eye1,decoder1,encoder1 = agents[a1]
    
    a1_hidden = eye1(test_images[i].unsqueeze(0)).view(1,1,-1)
    input = torch.tensor([sos_token])
    text = torch.zeros(MAX_LENGTH).int()
    word_length = 0
    for di in range(MAX_LENGTH):
        output, a1_hidden, probs = decoder1(input, a1_hidden)
        topv, topi = probs.topk(1)
        input = topi.detach()
        if input.item() == eos_token:
            break
        text[di] = input.item()
        word_length += 1
    #another random agent as listener
    a2 = random.randint(0,len(agents)-1)
    eye2,decoder2,encoder2 = agents[a2]
    a2_hidden = encoder2.initHidden()
    for ei in range(word_length):
        input = decoder2.embedding(text[ei])
        a2_hidden = encoder2(input,a2_hidden)
    
    pick_image = torch.zeros(11)
    
    #Compare hidden state produced by text to hidden states produced by distractor images and the orignal image
    for n in range(10):
        random_image = test_images[random.randint(0,len(test_images)-1)]
        eye_hidden = eye2(random_image.unsqueeze(0)).view(1,1,-1)
        pick_image[n] = cos(a2_hidden, eye_hidden)
    #similarity to the original image
    eye_hidden = eye2(test_images[i].unsqueeze(0)).view(1,1,-1)
    pick_image[10] = cos(a2_hidden, eye_hidden)
    if torch.argmax(pick_image).item() == 10:
        accurate += 1
print("test accuracy is ",accurate/(len(test_images)))


