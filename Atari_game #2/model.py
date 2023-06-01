import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import os 
import numpy as np 


class Linear_QNet(nn.Module):

	def __init__(self, n_inputs, n_hidden, n_outputs):
		super(Linear_QNet, self).__init__()
		self.layer1 = nn.Linear(n_inputs, n_hidden)
		self.layer2 = nn.Linear(n_hidden, n_outputs)


	def forward(self, x):
		x = F.relu(self.layer1(x))
		return self.layer2(x)


	def save(self, file_name="model.pth"):
		model_folder_path = "./model"
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)



class QTrainer: 

	def __init__(self, model, lr, gamma): 
		self.lr = lr 
		self.gamma = gamma 
		self.model = model 
		self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
		self.criterion = nn.MSELoss() 

	def train_step(self, state, action, reward, next_state, done):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float) 
		# (n, x)

		if len(state.shape) == 1: 
			# (1, x)
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			done = (done, )

		# current Q 
		pred = self.model(state)
		target = pred.clone()

		# new Q 
		for idx in range(len(done)):
			Q_new = reward[idx]
			if not done[idx]:
				Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

			target[idx][torch.argmax(action).item()] = Q_new


		self.optimizer.zero_grad()
		loss = self.criterion(target, pred)
		loss.backward() 

		self.optimizer.step()

