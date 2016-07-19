"""
Created on Thu Jul 14 13:24:03 2016

@author: jesseclark

The game environemt for the 'warehouse'.  Its a bit of a mess but it works.

"""

import numpy as np
import scipy
from collections import defaultdict


def rand_pair(s,ny,nx):
    """ Rand pair of numbers for a rectangle """
    return np.random.randint(s,ny), np.random.randint(s,nx)


def update_available(available_loc, unavailable_loc):
    """Return a list of available locations"""
    return [loc for loc in available_loc if loc not in unavailable_loc]


def choose_from_available(available_loc, n_loc=1):
    """Randomly choose from the available locations"""
    inds = np.random.choice(len(available_loc),n_loc,replace=False)

    return [ available_loc[ind] for ind in inds ]


def check_bounds(state,loc):
    """
    Check if the loc is within the bounds of the mxn state.
    """    
    nn = state.shape
    
    # check if a loc lies inside the array
    if loc[0] >= nn[0] or loc[0] < 0:
        return False
    elif loc[1] >= nn[1] or loc[1] < 0:
        return False
    else:
        return True  


def init_multi_game(nx,ny,n_players=2,n_fixes=5, wall_loc=[]):
    n_fixes_total = n_players*n_fixes
    
        #1.
    available_loc = [[y,x] for x in range(nx) for y in range(ny)]
    
    # update what is available
    available_loc = update_available(available_loc, wall_loc)
    # get player loc
    player_loc = choose_from_available(available_loc, n_loc=n_players)
    
    available_loc = update_available(available_loc, player_loc)
    
    fix_loc = np.array(choose_from_available(available_loc, n_fixes_total)).reshape(n_players,n_fixes,2)
    
        #2. populate
    
    # add in defualts
    game = defaultdict(dict)
    keys = ['player','wall','fix']
    for key in keys:
        for player in range(n_players):
            game[key][player] = np.zeros((ny,nx))
            
    # players
    for ind,loc in enumerate(player_loc):
        game['player'][ind][loc[0],loc[1]] = 1        
    
    # walls
    for ind in range(n_players):
        for loc in wall_loc:
            game['wall'][ind][loc[0],loc[1]] = 1        
    
    # fixes
    for ind in range(n_players):
        locs = fix_loc[ind,:,:]
        for loc in locs:
            game['fix'][ind][loc[0],loc[1]] = 1        
    
    # check the number of fixes
    game['n_fixes_start'] = np.sum([game['fix'][ind].sum() for ind in range(n_players)])
    
    # store some stuff   
    game['n_players'] = n_players
    game['n_fixes'] = n_fixes    
    
    # store history
    game['action_history'] = { ind:[] for ind in range(n_players)}
    game['n_moves'] = 0
    game['n_moves_reward'] = 0
    game['n_fixes_collected'] = 0
    
    game['x_t'] = { ind:[] for ind in range(n_players)}    
    
    game['pass_thru_fix'] = False    
    
    return game   


def update_alternate_and_locs(game):
    """Creates the view for the players"""
    game['wall_alt'] = [game['wall'][ind].copy() for ind in range(game['n_players'])]
    
    if not game['pass_thru_fix']:
        # make the combined wall for the players
        for ind1 in range(game['n_players']):
            for ind2 in range(game['n_players']):
                if ind1 != ind2:
                    game['wall_alt'][ind1] += game['player'][ind2].copy()+game['fix'][ind2].copy()

    # update the locations of the fixes and players
    for ind in range(game['n_players']):
        (y,x) = np.where(game['player'][ind] == 1)
        game['player_loc'][ind] = [[y[ind1],x[ind1]] for ind1 in range(int(game['player'][ind].sum()))]
        
    for ind in range(game['n_players']):
        (y,x) = np.where(game['fix'][ind] == 1)
        game['fix_loc'][ind] = [[y[ind1],x[ind1]] for ind1 in range(int(game['fix'][ind].sum()))]
    
    for ind in range(game['n_players']):
        (y,x) = np.where(game['wall_alt'][ind] == 1)
        game['wall_loc'][ind] = [[y[ind1],x[ind1]] for ind1 in range(int(game['wall_alt'][ind].sum()))]
   
    return game   
   
   
def get_new_loc(player_loc, action):
    """ Create new coordinates based on an action.
    """
    new_loc = player_loc # allows for a do nothing action`
    #up (row - 1)
    if action==0:
        new_loc = [player_loc[0] - 1, player_loc[1]]
            
    #down (row + 1)
    elif action==1:
        new_loc = [player_loc[0] + 1, player_loc[1]]
            
    #left (column - 1)
    elif action==2:
        new_loc = [player_loc[0], player_loc[1] - 1]
            
    #right (column + 1)
    elif action==3:
        new_loc = [player_loc[0], player_loc[1] + 1]

    return new_loc


def set_loc_value(state,loc,value):
    state[loc[0],loc[1]] = value
    
    return state


def frame_step(game, state_ind, action):
   
    # make a move and return
    player_loc = game['player_loc'][state_ind]
    # check bounds, check wall, check fix
    # get new location    
    new_loc = get_new_loc(player_loc[0],action)
    # check bounds
    status = 'none'
    
    game['n_moves'] += 1                
    game['n_moves_reward'] += 1      
    
    if new_loc != player_loc[0]:
        if check_bounds(game['player'][state_ind], new_loc):
            if new_loc not in game['wall_loc'][state_ind]:
                if new_loc not in game['fix_loc'][state_ind]:
                    status = 'ok'
                else:
                    status = 'hit fix'
                    game['n_moves_reward'] = 0
                    game['n_fixes_collected'] += 1
                
            else:
                status = 'hit wall'
            
        else:
            status = 'hit boundry'

    else:
        status = 'no move'

    # set the new player pos and remove fix if necessary
    if status in ['ok','hit fix']:
        # remove old one
        game['player'][state_ind] = set_loc_value(game['player'][state_ind],player_loc[0],0)               
        game['player'][state_ind] = set_loc_value(game['player'][state_ind],new_loc,1)
        
        if status == 'hit fix':
            # remove the fix
            game['fix'][state_ind] = set_loc_value(game['fix'][state_ind],new_loc,0)
            
    # store the action
    game['action_history'][state_ind].append(action)                
                            
    return game,status   
   
   
def convert_RGB_to_Y(array):
    """Convert RGB image"""    
    # assumes fast axis is first
    nn = array.shape
    im_out = (0.299*array[0,:,:] + 0.587*array[1,:,:] + 0.114*array[2,:,:])
        
    return im_out.reshape(1,nn[1],nn[2])   


def convert_states_RGB_to_Y(game, state_ind=[]):
    """Convert the players states """
    for ind in range(game['n_players']):
        game['out'][ind] = convert_RGB_to_Y(np.stack((game['player'][ind].copy(),game['wall_alt'][ind].copy(),game['fix'][ind].copy())))
    
    if state_ind == []:
        for ind in range(game['n_players']):
            game['x_t'][ind].append(game['out'][ind].copy())
    else:
        game['x_t'][state_ind].append(game['out'][state_ind].copy())
        
    return game


def convert_RGB_to_RGB(array):
    nn = array.shape
    
    return array.reshape(1,nn[0],nn[1],nn[2])      


def convert_states_RGB_to_RGB(game, state_ind=[]):
    """Convert the states to RGB"""
    for ind in range(game['n_players']):
        game['out'][ind] = convert_RGB_to_RGB(np.stack((game['player'][ind].copy(),game['wall_alt'][ind].copy(),game['fix'][ind].copy())))
    
    if state_ind == []:
        for ind in range(game['n_players']):
            game['x_t'][ind].append(game['out'][ind].copy())
    else:
        game['x_t'][state_ind].append(game['out'][state_ind].copy())
        
    return game    
    
   
def action_map(action):   

    if action == 0:
        return 'up'
    if action == 1:
        return 'down'
    if action == 2:
        return 'left'
    if action == 3:
        return 'right'
    
    return 'no move'


def check_action_history(game, max_actions, state_ind):
    """Check for repeated moves using max_actions previous steps."""    
    # return terminal condition for max actions
    if type(game) != type([]):
        action_history = np.array(game['action_history'][state_ind][-max_actions:])
    else:
        action_history = np.array(game[-max_actions:])

    # default return value
    result = False
    # action history
    if len(action_history) >= max_actions:    
        # vectorize
        action_pairs = [ [0,1],[1,0],[2,3],[3,2],[4,4] ]
        for pair in action_pairs:
            
            n_m = np.array(pair*(max_actions/2))     
            diff = (abs(n_m - action_history)).sum()
            
            if diff == 0:
                result = True
        
        return result      
    else:
        return False


def get_st(game, state_ind, n_frames=4):
    """Create the phi(state) from the history using n previous frames."""
    
    game['s_t'][state_ind] = np.concatenate(game['x_t'][state_ind][-n_frames:])

    return game
        

class WhGame:
    """
        Class for the warehouse game.  It keeps track of some of the 
        things necessary for RL, like s_t and the action history.
    """
    
    def __init__(self, nx, ny, n_fixes=5, n_players=1, wall_loc=0, 
                 pass_thru_fix=False, RGB=True, term_on_collision = True, n_frames=4):
        
        self.nx = nx
        self.ny = ny
        self.n_fixes = n_fixes    # number of fixes for the game
        self.n_players = n_players
        
        
        # set reward structure
        self.reward_fix = 1.5
        self.reward_wall = -1.5
        self.reward_move = -.1
        self.reward_no_move = -1
        self.verbose = True
        self.status = 'none'
        
        # set end-game conditions 
        self.terminate_wall = term_on_collision     
        self.terminate_boundry = term_on_collision
        self.max_actions = 10
        self.n_frames = n_frames
        self.RGB = RGB
        self.pass_thru_fix = pass_thru_fix   

        self.wall_loc = wall_loc     
        self.terminal_reason = 'none'
        self.reward = 0

        # default obstacle positions
        if self.wall_loc == 0:
            self.wall_loc = [[2,2],[2,3],[2,6],[2,7],[5,2],[5,3],[5,6],[5,7],[8,2],[8,3],[8,6],[8,7]]

    def init_game(self):
        
        self.game = init_multi_game(self.nx, self.ny, n_players=self.n_players,
                                    wall_loc=self.wall_loc, n_fixes=self.n_fixes)
       
        # create alternate views, the walls for the other players
        self.game = update_alternate_and_locs(self.game)  
        if self.RGB:
            self.game = convert_states_RGB_to_RGB(self.game)
        else:
            self.game = convert_states_RGB_to_Y(self.game)
        
        # can other players pass through other player items?
        self.game['pass_thru_fix'] = self.pass_thru_fix
        
        # check reward and terminal
        self.terminal_reason = 'none'

    def init_game_custom(self,fix_loc=[],player_loc=[]):
        """Custom game. Pass through the positions
        1 player only at the mo."""

        self.game = init_multi_game(self.nx, self.ny, n_players=self.n_players, 
                                    wall_loc=self.wall_loc, n_fixes=self.n_fixes)

        # overwrite the posns
        self.game['fix'][0] *= 0
        for fx in fix_loc:        
            self.game['fix'][0][fx[0],fx[1]] = 1
            
        self.game['player'][0] *= 0
        for plyr in player_loc:        
            self.game['player'][0][plyr[0],plyr[1]] = 1
            
        # create alternate views, the walls for the other players
        self.game = update_alternate_and_locs(self.game)  
        if self.RGB:
            self.game = convert_states_RGB_to_RGB(self.game)
        else:
            self.game = convert_states_RGB_to_Y(self.game)
        
        self.game['pass_thru_fix'] = self.pass_thru_fix
        
        # check reward and terminal
        self.terminal_reason = 'none'

    def get_reward(self):

        if self.status == 'ok':
            self.reward = self.reward_move
        if self.status == 'hit fix':
            self.reward = self.reward_fix
        if self.status == 'hit wall':
            self.reward = self.reward_wall
        if self.status == 'hit boundry':
            self.reward = self.reward_wall
        if self.status == 'no move':
            self.reward = self.reward_no_move

    def check_terminal(self):
        
        self.terminal = False
        self.terminal_reason = 'none'

        # all fixes collected
        if self.game['n_fixes_collected'] >= self.game['n_fixes_start']:
            self.terminal = True
            self.terminal_reason = 'all fixes collected'

        # check wall
        if not self.terminal:                        
            if self.status == 'hit wall':
                self.terminal = self.terminate_wall
                self.terminal_reason = 'hit wall'

        # check boundry
        if not self.terminal:
            if self.status == 'hit boundry':
                self.terminal = self.terminate_boundry
                self.terminal_reason = 'hit boundry'

        # check action history
        if not self.terminal:
            # repeated actions
            self.terminal = check_action_history(self.game, self.max_actions, self.state_ind)
            if self.terminal:
                self.terminal_reason = 'repeated actions'

    def frame_step(self, a_t, state_ind):
        """Move the player and update the states."""        
        
        self.action = a_t
        self.state_ind = state_ind
        # update state
        self.game, self.status = frame_step(self.game, self.state_ind, self.action)
        # update ocations
        self.game = update_alternate_and_locs(self.game)
        # convert to output
        if self.RGB:
            self.game = convert_states_RGB_to_RGB(self.game, state_ind)
        else:
            self.game = convert_states_RGB_to_Y(self.game, state_ind)

        # check reward and terminal
        self.get_reward()        
        self.check_terminal()
        self.game = get_st(self.game, state_ind,n_frames=self.n_frames)

        return self.game['out'][state_ind],self.reward,self.terminal


def output_sequence_RGB(statess, sdir, n_players = 3):
    """
        Conver the game to RGB for saving.
    """    
    # create a single (or rgb) array from mult agents
    ind=0
    xx = range(0,len(statess),n_players)

    # colors for the different players
    colors = {}
    colors['player'] = [[0,0,225],[205,0,0],[0,128,0],[75,0,130]]
    colors['fix'] = [[0,195,255],[240,128,128],[0,255,0],[153,50,204]]

    for qq,ind in enumerate(xx):
        state = {}
        player = {}
        fix = {}
        wall = 1.0
        
        for ind1 in range(n_players):        
            try:
                state[ind1] = statess[ind+ind1].squeeze()
            except:
                state[ind1] = statess[xx[qq-2]+ind1].squeeze()
                
            wall *= state[ind1][1,:,:]
            
            player[ind1] = state[ind1][0,:,:]
            fix[ind1] = state[ind1][2,:,:]
           
        ny,nx = wall.shape
        rgb = np.zeros((ny,nx,3))
       
        # make the wall a color
        rgb[:,:,0] += wall*255
        rgb[:,:,1] += wall*255
        rgb[:,:,2] += wall*255


        for indc in range(3):
            for indp in range(n_players):
                rgb[:,:,indc] += player[indp]*colors['player'][indp][indc]
                rgb[:,:,indc] += fix[indp]*colors['fix'][indp][indc]

        rgb = scipy.misc.imresize(rgb,(ny*100,nx*100,3),interp='nearest')
        ii = sdir+'rgb-'+str(ind)+'.jpg'
        scipy.misc.toimage(rgb).save(ii)
