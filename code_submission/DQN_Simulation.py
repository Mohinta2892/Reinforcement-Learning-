'''This is a Deep Q network simulation for a spatial navigation experiment
Through this we try to understand how animals learn to navigate to goals when exposed to a new 
environment,. Developed as part of MSc dissertation.'''


from __future__ import print_function #Importing required libraries, neural network uses keras
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
#%matplotlib inline


# Initializing the maze, 0 means blocked paths, 1 means walkable
maze =  np.array([
    [ 0,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0,  0.,  1.,  0.,  0.,  0.]
])


visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5      # The current rat cell will be painted by gray 0.5


LEFT = 0  #actions
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.5

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
	
def return_decayed_value(starting_value, global_step, decay_step):
        """Returns the decayed value.
        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param starting_value the value before decaying
        @param global_step the global step to use for decay (positive integer)
        @param decay_step the step at which the value is decayed
        """
        decayed_value = starting_value * np.power(0.1, (global_step/decay_step))
        return decayed_value

# rat = (row, col) initial rat position (defaults to (0,3))

class Qmaze(object):

    def __init__(self, maze, rat=(0,3)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (3,0)  # target cell where the "cheese"
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)
		
    def set_target(self,row,column):
        self.target=(row,column)

    def mark_task(self, task_type):
        self._maze[0,0]=task_type #0 allocentric and 1 egocentric
		
    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = 0 * self.maze.size  #assumed extrinsic minimum reward is 0
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):

        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self, flag):

        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if flag=='S1AN' and rat_row == 3 and rat_col ==0:
            return 1.0
        elif flag=='S1AS' and rat_row == 3 and rat_col ==0:
            return 1.0
        elif flag=='S2AN' and rat_row == 3 and rat_col ==6:
            return 1.0
        elif flag=='S2AS' and rat_row == 3 and rat_col ==6:
            return 1.0
        elif flag=='S3EN' and rat_row == 3 and rat_col ==0:
            return 1.0
        elif flag=='S3ES' and rat_row == 3 and rat_col ==6:
            return 1.0
        elif flag=='S4EN' and rat_row == 3 and rat_col ==6:
            return 1.0
        elif flag=='S4ES' and rat_row == 3 and rat_col ==0:
            return 1.0
        else:
            return 0

    def act(self, action, flag):
        self.update_state(action)
        reward = self.get_reward(flag)
        self.total_reward += reward
        status = self.game_status(flag)
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape

        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0

        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self,flag):

        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape

        if flag=='S1AN' and rat_row == 3 and rat_col == 0:
            return 'win'
        elif flag=='S1AS' and rat_row == 3 and rat_col == 0:
            return 'win'
        elif flag=='S2AN' and rat_row == 3 and rat_col == 6:
            return 'win'
        elif flag=='S2AS' and rat_row == 3 and rat_col == 6:
            return 'win'
        elif flag=='S3EN' and rat_row == 3 and rat_col == 0:
            return 'win'
        elif flag=='S3ES' and rat_row == 3 and rat_col == 6:
            return 'win'
        elif flag=='S4EN' and rat_row == 3 and rat_col == 6:
            return 'win'
        elif flag=='S4ES' and rat_row == 3 and rat_col == 0:
            return 'win'
        elif flag=='S3EN' and rat_row == 3 and rat_col == 6:
            return 'lose'
        elif flag=='S3ES' and rat_row == 3 and rat_col == 0:
            return 'lose'
        elif flag=='S4EN' and rat_row == 3 and rat_col == 0:
            return 'lose'
        elif flag=='S4ES' and rat_row == 3 and rat_col == 6:
            return 'lose'
        elif flag=='S1AN' and rat_row == 3 and rat_col == 6:
            return 'lose'
        elif flag=='S1AS' and rat_row == 3 and rat_col == 6:
            return 'lose'
        elif flag=='S2AN' and rat_row == 3 and rat_col == 0:
            return 'lose'
        elif flag=='S2AS' and rat_row == 3 and rat_col == 0:
            return 'lose'

        return 'not_over'

    def valid_actions(self, cell=None):
        '''Get all valid actions for moving in the maze'''

        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

def show(qmaze):
    '''Displays the maze layout with marked occupied and unoccupied cells'''

    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[3, 0] = 0.9 # cheese cell
    img =plt.imshow(canvas, interpolation='none', cmap=plt.get_cmap('gray')) 
    plt.show()

    #return img




def play_game(model, qmaze, rat_cell,flag):
    '''change of states happen here'''

    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate

        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action, flag)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False
        
def completion_check(model, qmaze, flag):
    '''Checks if the maze traversal is complete and the agent has 
learned to go to required goal points'''

    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell, flag):
            return False
    return True

def qtrain(model, maze, **opt):

    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    '''If you want to continue training from a previous model,
    just supply the h5 file name to weights_file option'''

    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)

    qmaze = Qmaze(maze)

    #show(qmaze)	#Display the maze if required

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    allo_cnt=[]        # Win/lose count of allocentric task
    ego_cnt=[]         # Win/lose count of allocentric task
    track_history=[]   #track changes

    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0

        #Divide trials into allocentric(1,2)/egocentric tasks(3,4)
        if(epoch<20):
            rule_chc=1
        elif(epoch>=20 and epoch<40):
            rule_chc=3
        elif(epoch>=40 and epoch<=50): 
            rule_chc=2
        elif(epoch>50 and epoch <=65):
            rule_chc=4
        elif(epoch>65 and epoch <=85):
            rule_chc=1
        else:
            rule_chc=2
		
        if(rule_chc==1):   #Allocentric
            chc=np.random.choice([1,2],1) #p=[0.4,0.6])
            qmaze.mark_task(0)
            if(chc%2==0):
                flag='S1AN'
                rat_cell=(0,3)
                qmaze.set_target(3,0)
            else:
                flag='S1AS'
                rat_cell=(6,3)
                qmaze.set_target(3,0)
				
        elif(rule_chc==2):  #Allocentric
            chc=np.random.choice([1,2],1) #p=[0.4,0.6])
            qmaze.mark_task(0)
            if(chc%2==0):
                flag='S2AN'
                rat_cell=(0,3)
                qmaze.set_target(3,6)
            else:
                flag='S2AS'
                rat_cell=(6,3)
                qmaze.set_target(3,6)
				
        elif(rule_chc==3):   #Egocentric
            chc=np.random.choice([1,2],1) #p=[0.4,0.6])
            qmaze.mark_task(1)
            if(chc%2==0):
                flag='S3EN'
                rat_cell=(0,3)
                qmaze.set_target(3,0)
            else:
                flag='S3ES'
                rat_cell=(6,3)
                qmaze.set_target(3,6)
		
        elif(rule_chc==4):    #Egocentric
            chc=np.random.choice([1,2],1) #p=[0.4,0.6])
            qmaze.mark_task(1)
            if(chc%2==0):
                flag='S4EN'
                rat_cell=(0,3)
                qmaze.set_target(3,6)
            else:
                flag='S4ES'
                rat_cell=(6,3)
                qmaze.set_target(3,0)

	#random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action, flag)


            #Check status of game : win/lose/midway
            if game_status == 'win' and flag in ['S1AN','S1AS', 'S2AN', 'S2AS']:
                allo_cnt.append([epoch,1])
                win_history.append(1)
                track_history.append([flag,epoch, game_status])
                game_over = True
            elif game_status == 'win' and flag in ['S3EN', 'S3ES', 'S4EN', 'S4ES']:
                ego_cnt.append([epoch,1])
                win_history.append(1)
                track_history.append([flag,epoch, game_status])
                game_over = True
            elif game_status == 'lose' and flag in ['S1AN','S1AS', 'S2AN', 'S2AS']:
                allo_cnt.append([epoch,0])
                win_history.append(0)
                track_history.append([flag,epoch,game_status])
                game_over = True
            elif game_status == 'lose' and flag in ['S3EN', 'S3ES', 'S4EN', 'S4ES']:
                ego_cnt.append([epoch,0])
                win_history.append(0)
                track_history.append([flag,epoch,game_status])
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            #allow only 100 timesteps in a trial, if goal not reached, treat trial as lost
            if(n_episodes>=100 and game_over==False):
                if(flag in ['S1AN','S1AS', 'S2AN', 'S2AS']):
                     allo_cnt.append([epoch,0])
                elif(flag in ['S3EN', 'S3ES', 'S4EN', 'S4ES']):
                     ego_cnt.append([epoch,0])
                break

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0
            )
            loss = model.evaluate(inputs, targets, verbose=0)  #calc cost func loss

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
    
        #Print output from each trial
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))

        epsilon=return_decayed_value(epsilon, epoch, decay_step=25) #decay exploration rate exponentially
 

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)


    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    print("track history",track_history)
    print("win_history",win_history)
    print("allo_cnt",allo_cnt)
    print("ego_cnt",ego_cnt)
    print("len win_history",len(win_history))
    print(len(allo_cnt), len(ego_cnt))

    #np.savetxt('win_history_simulswitching.txt',win_history)
    #np.savetxt('win_historyall_simulswitching.txt',win_history_all)


    '''Printing graphs'''
    legend_properties = {'weight':'bold','size': 12}
    trial_num=[]
    reward_val1=np.zeros(100)
    reward_val=[]
    for i in allo_cnt:
       trial_num.append(i[0])
       reward_val.append(i[1])
       reward_val1[i[0]]=i[1]


    trial_num_ego=[]
    reward_val_ego1=np.zeros(100)
    reward_val_ego=[]
    for i in ego_cnt:
       trial_num_ego.append(i[0])
       reward_val_ego.append(i[1])
       reward_val_ego1[i[0]]=i[1]


    fig=plt.figure(figsize=[30,5])
    ax=fig.add_subplot(111)


    #Allo Plot
    ax.plot(np.arange(1,21,1),reward_val1[:20],color='r',linestyle='--',linewidth=0.4)

    ax.plot(np.arange(40,51,1),reward_val1[40:51],color='r',linestyle='--',linewidth=0.4)

    ax.plot(np.arange(66,86,1),reward_val1[66:86],color='r',linestyle='--',linewidth=0.4)

    ax.plot(np.arange(86,100,1),reward_val1[86:],color='r',linestyle='--',linewidth=0.4, label='Allocentric')
    ax.set_ylim((-0.1,1.1))


    y_av_20=movingaverage(reward_val1[:20],6)
    y_av_40=movingaverage(reward_val1[40:51],6)
    y_av_66=movingaverage(reward_val1[66:86],6)
    y_av_86=movingaverage(reward_val1[86:],6)

    ax.plot(np.arange(1,21,1),y_av_20, color='r',linestyle='solid', linewidth=2,label='Moving Avg. (Allocentric)')
    ax.plot(np.arange(40,51,1),y_av_40, color='r',linestyle='solid', linewidth=2,)
    ax.plot(np.arange(66,86,1),y_av_66, color='r',)
    ax.plot(np.arange(86,100,1),y_av_86, color='r',)


    #Ego plot
    ax.plot(np.arange(20,40,1),reward_val_ego1[20:40],color='b',linestyle='--',linewidth=0.4, label='Egocentric')
    ax.plot(np.arange(50,66,1),reward_val_ego1[50:66],color='b',linestyle='--',linewidth=0.4,)


    y_av_ego40=movingaverage(reward_val_ego1[20:40],6)
    y_av_ego66=movingaverage(reward_val_ego1[50:66],6)

    ax.plot(np.arange(20,40,1),y_av_ego40, color='b',linestyle='solid', linewidth=2,label='Moving Avg. (Egocentric)')

    ax.plot(np.arange(50,66,1),y_av_ego66, color='b',linestyle='solid', linewidth=2,)
    
    ax.set_xticks(np.arange(1, 125,25))

    ax.set_xlabel('Trials',fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Accuracy',fontsize=14, fontweight='bold')
    ax.set_ylim((-0.1,1.1))
    fig.suptitle("Continual learning of strategies across days in simulation",fontsize=16, fontweight='bold')
    ax.hlines(0.5,1,100,linestyles='dashed',linewidth=1, color='k', label='Chance Level')

    #Target lines : mean performance
    target_avg_allo=(sum(reward_val1[:20])+sum(reward_val1[40:51])+sum(reward_val1[66:86])+sum(reward_val1[86:]))/65.0
    target_avg_ego=(sum(reward_val_ego1[20:40])+sum(reward_val_ego1[50:66]))/35.0


    ax.hlines(target_avg_allo,1,100,linestyles='-.',linewidth=2.2, color='#8b0000', label='Allocentric Perf. Mean')
    ax.hlines(target_avg_ego,1,100,linestyles='-.',linewidth=2.2,color='#003366', label='Egocentric Perf. Mean')

    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(direction='in')

    lgd=ax.legend(handles, labels,loc='lower center', bbox_to_anchor=(0.5, -0.24),
          ncol=8, fancybox=True, shadow=True,prop=legend_properties)

    #Save vector graphs
    #plt.savefig('simulation_switching.pdf', bbox_inches='tight', dpi=200)
    #plt.savefig('simulation_switching.svg', bbox_inches='tight', dpi=200)
    plt.show()

    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

class Experience(object):

    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        '''remembers timesteps'''
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate

            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)

            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
    
def build_model(maze, lr=0.1):
    '''Creating the model in using keras Sequential, lr=learning rate'''

    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model(maze) #get the model
qtrain(model, maze, n_epoch=100, max_memory=8*maze.size, data_size=32) #start the network training


