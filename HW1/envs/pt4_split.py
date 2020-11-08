import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def cmp(a, b):
        return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
#deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(self, hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]
    
    
class FooEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, natural=False):      
        self.action_space = spaces.Discrete(6) # 2 + 2 + 1 + 1
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2),
            
            spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2),
            spaces.Box(-22.0, +22.0, shape=(1,1), dtype=np.float32)
        ))
        
        # =============== EDITED ===============
        self.deck = [1, 2, 3, 4, 
                     5, 6, 7, 8, 9, 
                     10, 10, 10, 10] * 4
        
        self.card_points = {
            1: -1, 2: 0.5, 3: 1,
            4: 1, 5: 1.5, 6: 1,
            7: 0.5, 8: 0, 9: -0.5,
            10: -1,
        }
        
        self.card_counter = 0.0
        # =======================================
        
        self.seed()
        
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()
        
    
    # =============== EDITED ===============
    def draw_card(self, np_random):
        card = self.deck.pop(np_random.randint(0, len(self.deck)))
        self.card_counter += self.card_points[card]
        return int(card)

    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]

    # =======================================
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, hnd, k):
        rew = 0
        if is_bust(hnd):
            rew -= self.coef[k]
        elif self.natural and is_natural(hnd):
            assert self.coef[k] == 1.0
            rew += cmp(score(hnd), score(self.dealer)) * 1.5
        else:
            rew += cmp(score(hnd), score(self.dealer)) * self.coef[k]

        return rew
    
    def is_done(self, flg):
        if flg == 1:
            return is_bust(self.player) or self.coef[0] == 2.0
        else:
            return len(self.second_player) == 0 or is_bust(self.second_player)\
                                              or self.coef[1] == 2.0
            
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        # default actions
        if action <= 3:
            player_num = action // 2
            reward = 0.0
            done = (self.is_done(1),
                    self.is_done(2))

            if player_num == 0 or (player_num == 1 and len(self.second_player) != 0):
                chsn_hand = None
                if player_num == 0:
                    chsn_hand = self.player
                else:
                    chsn_hand = self.second_player
                chsn_player_action = action % 2
                
                # chsn_player_action = 0 hit: add a card to players hand 
                if chsn_player_action == 0:
                    reward = 0.0
                    
                    if (not is_bust(chsn_hand)) and self.coef[player_num] == 1.0:
                        chsn_hand.append(self.draw_card(self.np_random)) #add elem to chosen hand
                    
                    done = (self.is_done(1), 
                            self.is_done(2))
                    
                    if done[0] == done[1] == True:
                        if not (is_bust(self.player) == is_bust(self.second_player) == True):
                            while sum_hand(self.dealer) < 17:
                                self.dealer.append(self.draw_card(self.np_random))
                                
                        reward += self.get_reward(self.player, 0)
                        
                        if len(self.second_player) != 0:
                            reward += self.get_reward(self.second_player, 1)
                
                # chsn_player_action = 1 double
                elif chsn_player_action == 1:
                    reward = 0.0
                    
                    if (not is_bust(chsn_hand)) and self.coef[player_num] == 1.0:
                        chsn_hand.append(self.draw_card(self.np_random)) #add elem to chosen hand
                    
                    self.coef[player_num] = 2.0 #double
                    
                    done = (self.is_done(1),
                            self.is_done(2))
                    
                    if done[0] == done[1] == True:
                        if not (is_bust(self.player) == is_bust(self.second_player) == True):
                            while sum_hand(self.dealer) < 17:
                                self.dealer.append(self.draw_card(self.np_random))
                        
                        reward += self.get_reward(self.player, 0)
                        
                        if len(self.second_player) != 0:
                            reward += self.get_reward(self.second_player, 1)
        # stand analogue
        elif action == 4:
            done = (True, True)
            reward = 0.0
            
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            
            # hand 1
            reward += self.get_reward(self.player, 0)

            # check hand 2 exists or not
            if len(self.second_player) != 0:
                reward += self.get_reward(self.second_player, 1)
        # split
        elif action == 5:
            done = (False, False)
            reward = 0.0
            
            if len(self.player) == 2 and self.player[0] == self.player[1]\
                                     and len(self.second_player) == 0:
                self.second_player.append(self.player.pop(1))
                
                self.player.append(self.draw_card(self.np_random))
                self.second_player.append(self.draw_card(self.np_random))
                
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), sum_hand(self.second_player), 
                self.dealer[0], usable_ace(self.player), 
                usable_ace(self.second_player), self.card_counter)

    
    def reset(self):
        if len(self.deck) < 20:
            # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
            self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
            self.card_counter = 0.0
        
        self.dealer = self.draw_hand(self.np_random)
        self.player = self.draw_hand(self.np_random)
        
        self.second_player = []
        self.coef = [1.0, 1.0]
        
        return self._get_obs()
    
    #===================== EDITED =====================
    def check_action(self, action):
        if action == 5:
            if len(self.player) == 2 and self.player[0] == self.player[1]\
                                     and len(self.second_player) == 0:
                    return True
            return False
        elif action == 4:
            return True
        elif action == 2 or action == 3:
            if len(self.second_player) != 0 and not is_bust(self.second_player)\
                                          and self.coef[1] == 1.0:
                return True
            return False
        else:
            if len(self.player) != 0 and not is_bust(self.player)\
                                     and self.coef[0] == 1.0:
                return True
            return False
    
    
    def random_action(self):
        action = self.action_space.sample()
        while not self.check_action(action):
            action = self.action_space.sample()
        return action

    #=================== EDITED END ===================
