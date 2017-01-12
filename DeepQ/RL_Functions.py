# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:24:03 2016

Originally inspired by
http://outlace.com/Reinforcement-Learning-Part-3/

@author: jesseclark
"""
from keras.models import Sequential
from keras.optimizers import RMSprop,SGD,adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import model_from_json

import os
import copy
import numpy as np
import random
from IPython.display import clear_output
import time

import GameEnv


def create_model(img_channels, img_rows, img_cols, n_conv1=32, n_conv2=64,
                      n_conv3=64, n_out1=512, n_out2=-1, lr=.001, 
                      n_actions=4, loss='mse'):
    """ 
        Make a keras CNN model.  
    """
    
    model = Sequential()
    model.add(Convolution2D(n_conv1, 5, 5, border_mode='same',subsample=(2,2),
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(PReLU()) 
    
    model.add(Convolution2D(n_conv2, 3, 3, border_mode='same',subsample=(2,2)))
    model.add(PReLU()) 
    
    model.add(Convolution2D(n_conv3, 3, 3, border_mode='same'))
    model.add(PReLU())
    
    model.add(Flatten())
    model.add(Dense(n_out1))
    model.add(PReLU()) 
    
    model.add(Dense(n_actions))
    model.add(Activation('linear'))
    
    # try clipping or huber loss
    model.compile(loss=loss, optimizer=adam(lr=lr))

    return model


def create_dueling_net(img_channels, img_rows, img_cols, n_conv1=32, n_conv2=64,
                      n_conv3=64, n_out1=512, n_out2=-1, lr=.001,
                      n_actions=4, loss='mse', use_perm_drop=False, drop_o=.25):

    def make_output(x):
        x = -K.mean(x, axis=1, keepdims=True)
        x = K.tile(x, 4)
        return x

    def make_output_shape(input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def perm_drop(x):
        return K.dropout(x, .25)

    # input for the netwrok
    input = Input(shape=(img_channels, img_rows, img_cols))

    # conv layers - shared by both netwroks
    conv1 = Convolution2D(n_conv1, 5, 5, border_mode='same',subsample=(2,2))(input)
    prelu1 = PReLU()(conv1)

    conv2 = Convolution2D(n_conv2, 3, 3, border_mode='same',subsample=(2,2))(prelu1)
    prelu2 = PReLU()(conv2)

    conv3 = Convolution2D(n_conv2, 3, 3, border_mode='same')(prelu2)
    prelu3 = PReLU()(conv3)

    flatten = Flatten()(prelu3)

    # A(s,a)
    dense11 = Dense(n_out1)(flatten)

    if not use_perm_drop:
        prelu31 = PReLU()(dense11)
    else:
        prelu310 = PReLU()(dense11)
        prelu31 = Lambda(perm_drop, output_shape=make_output_shape)(prelu310)

    dense21 = Dense(n_actions)(prelu31)
    out1 = Activation('linear')(dense21)

    # V(s)
    dense12 = Dense(n_out1)(flatten)
    if not use_perm_drop:
        prelu32 = PReLU()(dense12)
    else:
        prelu320 = PReLU()(dense12)
        prelu32 = Lambda(perm_drop, output_shape=make_output_shape)(prelu320)

    dense22 = Dense(1)(prelu32)
    out2 = Activation('linear')(dense22)
    out2 = RepeatVector(n_actions)(out2)
    out2 = Reshape((n_actions,))(out2)

    # - E[ A(s,a) ]
    out3 = Lambda(make_output, output_shape=make_output_shape)(out1)

    output = merge([out1, out2, out3], mode='sum', concat_axis=1)

    model = Model(input=input,output=output)
    model.compile(loss=loss, optimizer=adam(lr=lr))

    return model


def save_model(model, m_name):
    """Save keras model to json and weights to h5. """
    
    json_string = model.to_json()
    open(m_name+'.json', 'w').write(json_string)
    model.save_weights(m_name+'.h5')


def load_model(m_name, loss='mse', optimizer='adam'):
    """Load keras model from json and h5."""

    model_l = model_from_json(open(m_name+'.json').read())
    model_l.load_weights(m_name+'.h5')
    model_l.compile(loss=loss,optimizer=optimizer)
    
    return model_l


def transfer_dense_weights(model1, model2):
    """    
     Transfer weights for dense layers between keras models.
     transfer model1 to model2
    """
    for ind in range(len(model2.layers)):
        if 'dense' in model2.layers[ind].get_config()['name'].lower():
            try:            
                print('*')
                weights = copy.deepcopy(model1.layers[ind].get_weights())
                model2.layers[ind].set_weights(weights)
            except:
                print('!')
    return model2   


def transfer_conv_weights(model1, model2):
    """    
     Transfer weights for conv layers between keras models.        
     Transfer model1 to model2.
    """
    
    for ind in range(len(model2.layers)):
        if 'convolution' in model2.layers[ind].get_config()['name'].lower():
            try:            
                weights = copy.deepcopy(model1.layers[ind].get_weights())
                model2.layers[ind].set_weights(weights)
            except:
                print('!')
    return model2


def transfer_all_weights(model1,model2):
    """ Transfer all weights between keras models"""
    
    for ind in range(len(model2.layers)):
        try:            
            weights = copy.deepcopy(model1.layers[ind].get_weights())
            model2.layers[ind].set_weights(weights)
        except:
            print('!')
            
    return model2
    

def create_duplicate_model(model):
    """Create a duplicate keras model."""    
    
    new_model = Sequential.from_config(model.get_config())    
    new_model.set_weights(copy.deepcopy(model.get_weights()))
    new_model.compile(loss=model.loss,optimizer=model.optimizer)
            
    return new_model


def add_to_replay(replay, state, action, reward, new_state, replay_buffer, n_times=1):
    # append to replay
    [replay.append((state.copy(), action, reward, new_state.copy())) for ind in range(n_times)]

    [replay.pop(0) for ind in range(n_times) if len(replay) > replay_buffer]

def sample_minibatch(replay, minibatch_size, priority=False):
    # priority replay minibatch sampling

    if priority:
        _,_,_,_,_,dq = zip(*replay)
        probs = np.array(dq)/sum(dq)
        inds = np.random.choice(range(len(replay)),minibatch_size,p=probs)
        return [replay[ind] for ind in inds]
    else:
        return random.sample(replay, minibatch_size)

def train(parameters):
    """
        Train the Q network using RL.
    """
    
    # keep track of some paramters
    parameters['best_score'] = 0
    frame_number = 0

    # ideally we set these outside
    # number of items and normalised number of items collected
    parameters['n_items_collected'] = []
    parameters['norm_items_collected'] = []
    
    # moves per game
    parameters['n_moves_made'] = []

    # why did the game terminate
    parameters['term_reasons'] = []
    parameters['dq_errors'] = []

    dq_errors_added = []

    # iterate over the games
    for i in range(parameters['n_games']):

        time_start = time.time()

        Game = GameEnv.WhGame(parameters['nx'],parameters['ny'],n_fixes=parameters['n_items'],
                                        n_players=parameters['n_players'],wall_loc=parameters['wall_loc'], 
                                        term_on_collision=parameters['term_on_collision'], n_frames=parameters['n_frames'])

        # set this ti False for 2D only, will need to adjust the input for the netwrok accordingly
        Game.RGB = True
        Game.init_game()

        # do nothing frame at the start - 4 is the do nothing move number (anything > 3 will do nothing)
        _ = [Game.frame_step(4, ind) for ind in range(parameters['n_players']) for dind in range(parameters['n_frames'])]

        status = 1
        moves_game = 0

        #while game still in progress
        while(status == 1 and moves_game < parameters['max_moves']):

            # game move counter
            moves_game +=1
            # total frame numbers 
            frame_number += 1

            # get one of the current states
            cur_ind = (moves_game % parameters['n_players'])

            # don't continue if no result after max_moves - only terminal 
            # condition we set outside the game
            # although we don't adjust the Q update with this one
            if moves_game >= parameters['max_moves']:
                print("## MAX MOVES ##")
                Game.terminal_reason = 'maximum moves'
                status = 0

            # get the current concatenated game state (constructed within GameEnv,
            # could be done here)
            state = Game.game['s_t'][cur_ind].copy().reshape((1,parameters['img_channels'],Game.ny,Game.nx))

            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions
            qval = parameters['model'].predict(state,batch_size=1)

            # choose random action, 4 is for up/down/left/right -
            # the number of possible moves
            if (random.random() < parameters['epsilon']) or frame_number < parameters['observe']: 
                action = np.random.randint(0,parameters['n_actions'])
            else:
                # choose best action from Q(s,a) values
                action = (np.argmax(qval))

            # Take action, observe new state S' and get terminal
            # terminal - all items collected, hit wall, hit boundry, repeated actions
            # still an edge case in multiplayer that needs to be addressed
            x_t, reward, terminal = Game.frame_step(action, cur_ind)

            # make the state histroy for player cur_ind
            new_state = Game.game['s_t'][cur_ind].copy().reshape((1,parameters['img_channels'],Game.ny,Game.nx))

            # dq error for priority replay
            qval_new = parameters['model'].predict(new_state,batch_size=1)
            dq_error = abs(qval[0][action]-qval_new[0][action]-reward)
            parameters['dq_errors'].append(dq_error)

            # experience replay
            exp_tuple = (state.copy(), action, reward, new_state.copy(), terminal, dq_error)
            parameters['replay'].append(exp_tuple)
            # check replay size
            if len(parameters['replay']) > parameters['replay_buffer']:
                parameters['replay'].pop(0)

            # # add in a hacky way of prioritising replay
            # # ideally use a priority queue
            # if frame_number > parameters['observe']:
            #     # leave some room for the dist width
            #     parameters['dq_errors'].pop(0)
            #     # could also do based on std or prob and cdf
            #     if dq_error >= 1.1*np.mean(parameters['dq_errors'][-1000:-1]):
            #         parameters['replay'].append(exp_tuple)
            #         dq_errors_added.append(dq_error)
            #
            #         if len(parameters['replay']) > parameters['replay_buffer']:
            #             parameters['replay'].pop(0)

            # are we done observing?
            if frame_number > parameters['observe']:        
                # randomly sample the exp replay memory (could add better choice here)
                # minibatch = random.sample(parameters['replay'], parameters['batch_size'])
                minibatch = sample_minibatch(parameters['replay'], parameters['batch_size'], priority=True)

                X_train, y_train = process_minibatch(parameters['model'],minibatch,model_target=parameters['model_target'],
                                                                   n_actions=parameters['n_actions'], 
                                                                   dbldqn=parameters['dqn'])

                model_temp = parameters['model'].fit(X_train, y_train, 
                                batch_size=parameters['batch_size'], nb_epoch=1, verbose=0)
                parameters['loss'].append(model_temp.history['loss'][0])


            # update target network
            if (frame_number % parameters['update_target']) == 0:
                print('** Update target **')
                parameters['model_target'] = transfer_all_weights(parameters['model'],
                                                    parameters['model_target'])

            # stop playing if we are terminal
            if terminal:
                status = 0

        # store the best model - use previous scores, ideally we would intermittently play the game
        if np.mean(parameters['norm_items_collected'][-5:]) > parameters['best_score']:
            parameters['model_best'] = transfer_all_weights(parameters['model'],parameters['model_best'])
            parameters['best_score'] = np.mean(parameters['norm_items_collected'][-5:])
            print('^^ Updated best ^^')

        # decrement epsilon over games
        if parameters['epsilon'] > parameters['epsilon_min']: 
            parameters['epsilon'] -= (1./parameters['epsilon_stop'])
        else:
            parameters['epsilon'] = parameters['epsilon_min']

        # metrics to keep track of game learning progress
        parameters['norm_items_collected'].append(1.0*Game.game['n_fixes_collected']/Game.game['n_fixes_start'])
        parameters['n_moves_made'].append(moves_game)      
        parameters['n_items_collected'].append(Game.game['n_fixes_collected'])

        clear_output(wait=True)

        # display some params during
        print("Game #: %s" % (i,))
        print("Moves this round %s" % moves_game)
        print("Items collected %s" % Game.game['n_fixes_collected'])
        print(Game.terminal_reason)
        if frame_number > parameters['observe']:
            print("Loss %s" % parameters['loss'][-1])
            print("Avg. items %s" % np.mean(parameters['n_items_collected'][-20:]))
            print("Avg. score %s" % np.mean(parameters['norm_items_collected'][-20:]))
            print("Epsilon %s" % parameters['epsilon'])

        print("Game time %s"% (time.time()-time_start) )

        # keep track of terminal reasons - good for debugging
        parameters['term_reasons'].append(Game.terminal_reason)

    print("Finished")

    return parameters


def process_minibatch(model, minibatch, model_target=[], gamma=0.9,
                         n_actions=4, dbldqn=False):
                             
    """Process the random minibatches and get the update."""

    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        
        y = np.zeros((1,n_actions))
        
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m, terminal_m = memory
        # Get prediction on old state s with new params w.
        Q_s_w = model.predict(old_state_m, batch_size=1)
        
        # get prediction of new state sd with old params wd
        Q_sd_wd = model_target.predict(new_state_m, batch_size=1)

        if dbldqn:
            # get action for new state sd, new params w            
            Q_sd_w = model.predict(new_state_m, batch_size=1)
            max_action = (np.argmax(Q_sd_w))
            # get Q for new state with max action Q_sd_maxa old params
            maxQ = Q_sd_wd[0][max_action]
        else:
            # Get max over actions
            maxQ = np.max(Q_sd_wd)

        # old q
        y[:] = Q_s_w[:]

        if not terminal_m:
            update = (reward_m + (gamma * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
    
        x_temp = np.squeeze(old_state_m)

        X_train.append(x_temp)
        y_train.append(y.reshape(n_actions,))
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train      


def play(parameters):
    """
        Play using the Q network.
    """
    
    # moves played in the game
    parameters['played_frames'] = []
    
    Game = GameEnv.WhGame(parameters['nx'],parameters['ny'],n_fixes=parameters['n_items'],
                                    n_players=parameters['n_players'],wall_loc=parameters['wall_loc'], 
                                    term_on_collision=parameters['term_on_collision'], n_frames=parameters['n_frames'])

    Game.RGB = True
    Game.init_game()

    # do nothing frame at the start - 4 is the do nothing move number (anything > 3 will do nothing)
    _ = [Game.frame_step(4, ind) for ind in range(parameters['n_players']) for dind in range(parameters['n_frames'])]

    status = 1
    moves_game = 0

    # while game still in progress
    while(status == 1 and moves_game < parameters['max_moves']):

        # game move counter
        moves_game +=1
        
        # get one of the current states
        cur_ind = (moves_game % parameters['n_players'])

        # don't continue if no result after max_moves - only terminal 
        # condition we set outside the game
        # although we don't adjust the Q update with this one
        if moves_game >= parameters['max_moves']:
            print("## MAX MOVES ##")
            Game.terminal_reason = 'maximum moves'
            status = 0

        # get the current concatenated game state (constructed within GameEnv,
        # could be done here)
        state = Game.game['s_t'][cur_ind].copy().reshape((1,parameters['img_channels'],Game.ny,Game.nx))

        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        qval = parameters['model'].predict(state,batch_size=1)

        # choose random action, 4 is for up/down/left/right -
        # the number of possible moves
        if (random.random() < parameters['epsilon_min']): 
            action = np.random.randint(0,parameters['n_actions'])
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))

        # Take action, observe new state S' and get terminal
        # terminal - all items collected, hit wall, hit boundry, repeated actions
        x_t, reward, terminal = Game.frame_step(action, cur_ind)

        # store the time step
        parameters['played_frames'].append(x_t)

        # stop playing if we are terminal
        if terminal:
            status = 0

    clear_output(wait=True)
    parameters['n_moves_played'] = moves_game
    # display some params during
    print("Moves this round %s" % moves_game)
    print("Items collected %s" % Game.game['n_fixes_collected'])
    print(Game.terminal_reason)
    
    print("Finished")
    
    return parameters
